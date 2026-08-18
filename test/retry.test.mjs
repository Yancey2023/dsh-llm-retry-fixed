/**
 * Unit tests for dsh-llm-retry-fixed — the fixed-schedule quota retry
 * executor on the agent/request-error recovery extension point.
 *
 *   QUOTA failure        -> immediate retry decision, durable llm/retry events
 *   non-QUOTA failure    -> passed through to next() untouched
 *   retry counting       -> schedule consumed in order, then give up
 *   restart durability   -> prior retries counted from the session log
 *   abort during wait    -> no retry decision (loop aborts the turn)
 *   config validation    -> empty schedule rejected
 */
import assert from 'node:assert/strict'
import { apply, DEFAULT_CODES, DEFAULT_DELAYS_MS } from '../lib/index.js'

function makeCtx() {
  let handler = null
  const ctx = {
    on(event, fn) {
      assert.equal(event, 'agent/request-error')
      handler = fn
      return () => {
        if (handler === fn) handler = null
      }
    },
    effect() {
      return () => {}
    },
    logger: { warn() {} },
  }
  ctx._dispatch = (payload, next) => handler(payload, next)
  return ctx
}

function makeAgent() {
  const events = []
  return { session: { events, append(type, data) { events.push({ type, data }) } } }
}

function quotaFailure(message = 'Allocated quota exceeded, please increase your quota limit.') {
  return { message, code: 'QUOTA' }
}

function nextCaller() {
  let called = 0
  const next = () => {
    called += 1
    return undefined
  }
  next.called = () => called
  return next
}

// ---------------------------------------------------------------------------
// QUOTA failure: immediate retry with durable events and the fixed schedule.
// ---------------------------------------------------------------------------
{
  const ctx = makeCtx()
  apply(ctx, {})
  const agent = makeAgent()
  const next = nextCaller()
  const signal = new AbortController().signal
  const result = await ctx._dispatch({ agent, turn: 3, step: 2, provider: 'relayfor', failure: quotaFailure(), signal }, next)
  assert.deepEqual(result, { kind: 'retry' }, 'QUOTA failure retries')
  assert.equal(next.called(), 0, 'next not consulted for QUOTA')
  const retry = agent.session.events.find((e) => e.type === 'llm/retry')
  assert.ok(retry, 'durable llm/retry event appended')
  assert.equal(retry.data.retry, 1)
  assert.equal(retry.data.delayMs, 0, 'first retry is immediate')
  assert.equal(retry.data.maxRetries, DEFAULT_DELAYS_MS.length, 'maxRetries = schedule length')
  assert.equal(retry.data.mode, 'normal')
  assert.equal(retry.data.turn, 3)
  assert.equal(retry.data.step, 2)
  assert.equal(retry.data.provider, 'relayfor')
  assert.equal(retry.data.failure.code, 'QUOTA')
  assert.equal(agent.session.events.filter((e) => e.type === 'llm/retry-started').length, 1, 'retry-started appended after the wait')
}

// ---------------------------------------------------------------------------
// RATE_LIMIT failure (the code the harness actually emits for the user's
// "rpm exhausted" / quota_exceeded_error 429): retried on the fixed schedule
// too — the harness maps that error to RATE_LIMIT (isQuotaExceededError never
// matches the glued "quota_exceeded_error" wording), so without RATE_LIMIT
// coverage the stock policy would own it with its 2-retry budget and break
// the turn. See the regression test below for the exact real failure.
// ---------------------------------------------------------------------------
{
  const ctx = makeCtx()
  apply(ctx, {})
  const agent = makeAgent()
  const next = nextCaller()
  const signal = new AbortController().signal
  const result = await ctx._dispatch({ agent, turn: 1, step: 1, provider: 'p', failure: { message: '429: {"message":"rpm exhausted"}', code: 'RATE_LIMIT' }, signal }, next)
  assert.deepEqual(result, { kind: 'retry' }, 'RATE_LIMIT retries with the fixed schedule')
  assert.equal(next.called(), 0, 'next not consulted for RATE_LIMIT')
  const retry = agent.session.events.find((e) => e.type === 'llm/retry')
  assert.equal(retry.data.delayMs, 0, 'first retry immediate')
  assert.equal(retry.data.failure.code, 'RATE_LIMIT')
  assert.equal(retry.data.maxRetries, 9, 'full fixed schedule (9 retries, 10 attempts)')
}

// ---------------------------------------------------------------------------
// REGRESSION: the user's exact error — DeepSeek/sensenova 429
//   {"message":"rpm exhausted","type":"quota_exceeded_error","code":"8"}
// The harness classifies it as RATE_LIMIT (isQuotaExceededError does not
// match "rpm exhausted" nor "quota_exceeded_error" with its \b requirement).
// It must retry on the LONG fixed schedule (not the stock 2-retry budget).
// ---------------------------------------------------------------------------
{
  const ctx = makeCtx()
  apply(ctx, {})
  const agent = makeAgent()
  const next = nextCaller()
  const signal = new AbortController().signal
  const rpmExhausted = {
    message: '429: {"message":"rpm exhausted","type":"quota_exceeded_error","code":"8"}',
    code: 'RATE_LIMIT',
  }
  const result = await ctx._dispatch({ agent, turn: 5, step: 1, provider: 'sensenova2', failure: rpmExhausted, signal }, next)
  assert.deepEqual(result, { kind: 'retry' }, 'rpm-exhausted 429 retries')
  assert.equal(next.called(), 0, 'fixed schedule owns this failure, stock policy never consulted')
  const retry = agent.session.events.find((e) => e.type === 'llm/retry')
  assert.equal(retry.data.failure.message.includes('rpm exhausted'), true)
  assert.equal(retry.data.failure.code, 'RATE_LIMIT')
  assert.equal(retry.data.maxRetries, 9, '9 fixed retries — the conversation no longer breaks after 2')
}

// ---------------------------------------------------------------------------
// Listener ordering: the fixed executor's agent/request-error listener MUST
// be registered with prepend, so for matched codes it decides before the
// stock policy (which is registered first by dsh-base and would otherwise
// burn its own 2-retry budget for RATE_LIMIT and veto the chain).
// ---------------------------------------------------------------------------
{
  const calls = []
  const ctx = {
    on(event, fn, options) {
      if (event !== 'agent/request-error') throw new Error('unexpected event ' + event)
      calls.push({ prepend: options?.prepend === true, fn })
      return () => {}
    },
    effect() { return () => {} },
    logger: { warn() {} },
  }
  // Simulate dsh-base's stock listener registering FIRST (no prepend).
  ctx.on('agent/request-error', () => {}, undefined)
  // The fixed plugin registers AFTER but with prepend: true.
  apply(ctx, {})
  // verify: two listeners registered; the fixed one is prepended, the stock is not.
  assert.equal(calls.length, 2, 'stock + fixed = 2 listeners')
  const stock = calls.find((c) => c.prepend === false)
  const fixed = calls.find((c) => c.prepend === true)
  assert.ok(stock !== undefined, 'stock listener registered without prepend')
  assert.ok(fixed !== undefined, 'fixed listener registered with prepend')
}

// ---------------------------------------------------------------------------
// Raw provider code (insufficient_quota) is retried too — the exact 429 case.
// ---------------------------------------------------------------------------
{
  const ctx = makeCtx()
  apply(ctx, {})
  const agent = makeAgent()
  const next = nextCaller()
  const signal = new AbortController().signal
  const result = await ctx._dispatch(
    { agent, turn: 4, step: 1, provider: 'relayfor', failure: { message: 'Allocated quota exceeded, please increase your quota limit.', code: 'insufficient_quota' }, signal },
    next,
  )
  assert.deepEqual(result, { kind: 'retry' }, 'raw insufficient_quota code retries with the fixed schedule')
  assert.equal(next.called(), 0)
  const retry = agent.session.events.find((e) => e.type === 'llm/retry')
  assert.equal(retry.data.delayMs, 0, 'first retry immediate')
  assert.equal(retry.data.maxRetries, 9)
  assert.equal(retry.data.failure.code, 'insufficient_quota')
}

// ---------------------------------------------------------------------------
// Non-matched failure (SERVER): passed through untouched — the stock policy
// still owns every code outside the fixed set.
// ---------------------------------------------------------------------------
{
  const ctx = makeCtx()
  apply(ctx, {})
  const agent = makeAgent()
  const next = nextCaller()
  const signal = new AbortController().signal
  const result = await ctx._dispatch({ agent, turn: 2, step: 1, provider: 'p', failure: { message: 'upstream 502', code: 'SERVER' }, signal }, next)
  assert.equal(result, undefined, 'no retry decision for non-matched codes')
  assert.equal(next.called(), 1, 'next() consulted -> stock policy owns SERVER')
  assert.equal(agent.session.events.length, 0, 'no events appended')
}

// ---------------------------------------------------------------------------
// Schedule order and give-up: 3 retries (0/1/2ms), 4th failure gives up.
// ---------------------------------------------------------------------------
{
  const ctx = makeCtx()
  apply(ctx, { codes: ['QUOTA'], delaysMs: [0, 1, 2] })
  const agent = makeAgent()
  const seen = []
  const signal = new AbortController().signal
  for (let i = 0; i < 3; i += 1) {
    const next = nextCaller()
    const result = await ctx._dispatch({ agent, turn: 5, step: 1, provider: 'p', failure: quotaFailure(), signal }, next)
    assert.deepEqual(result, { kind: 'retry' }, `retry ${i + 1} decides retry`)
    assert.equal(next.called(), 0)
    const retry = agent.session.events.filter((e) => e.type === 'llm/retry')[i]
    seen.push(retry.data.delayMs)
  }
  assert.deepEqual(seen, [0, 1, 2], 'delays consumed in schedule order')
  assert.equal(agent.session.events.filter((e) => e.type === 'llm/retry').length, 3)
  // 4th failure: schedule exhausted -> next() -> the turn fails.
  const next = nextCaller()
  const result = await ctx._dispatch({ agent, turn: 5, step: 1, provider: 'p', failure: quotaFailure(), signal }, next)
  assert.equal(result, undefined)
  assert.equal(next.called(), 1, 'gives up after the schedule is exhausted')
  assert.equal(agent.session.events.filter((e) => e.type === 'llm/retry').length, 3, 'no further retry events')
}

// ---------------------------------------------------------------------------
// Restart durability: prior retries counted from the session log, so a fresh
// executor instance continues the same chain (and the same retry id).
// ---------------------------------------------------------------------------
{
  const ctx = makeCtx()
  apply(ctx, { codes: ['QUOTA'], delaysMs: [0, 1, 2, 3] })
  const agent = makeAgent()
  const signal = new AbortController().signal
  // One retry through the first executor instance.
  await ctx._dispatch({ agent, turn: 7, step: 1, provider: 'p', failure: quotaFailure(), signal }, () => {})
  // A SECOND executor instance (process restart) sees the prior llm/retry.
  const ctx2 = makeCtx()
  apply(ctx2, { codes: ['QUOTA'], delaysMs: [0, 1, 2, 3] })
  const next = nextCaller()
  const result = await ctx2._dispatch({ agent, turn: 7, step: 1, provider: 'p', failure: quotaFailure(), signal }, next)
  assert.deepEqual(result, { kind: 'retry' })
  assert.equal(next.called(), 0)
  const retries = agent.session.events.filter((e) => e.type === 'llm/retry')
  assert.equal(retries.length, 2, 'continued the chain, not restarted it')
  assert.equal(retries[1].data.retry, 2, 'retry counter continues')
  assert.equal(retries[1].data.delayMs, 1, 'next schedule slot used')
  assert.equal(retries[1].data.retryId, retries[0].data.retryId, 'same retry chain id across restarts')
}

// ---------------------------------------------------------------------------
// Abort during the wait: durable event written, but no retry decision.
// ---------------------------------------------------------------------------
{
  const ctx = makeCtx()
  apply(ctx, { codes: ['QUOTA'], delaysMs: [5000] })
  const agent = makeAgent()
  const controller = new AbortController()
  const pending = ctx._dispatch({ agent, turn: 9, step: 1, provider: 'p', failure: quotaFailure(), signal: controller.signal }, () => {
    throw new Error('next must not be called during an aborted wait')
  })
  // Give the append + wait a beat, then abort.
  await new Promise((resolve) => setTimeout(resolve, 10))
  controller.abort()
  const result = await pending
  assert.equal(result, undefined, 'aborted wait yields no retry decision')
  assert.equal(agent.session.events.filter((e) => e.type === 'llm/retry').length, 1, 'durable event survives the abort')
  assert.equal(agent.session.events.filter((e) => e.type === 'llm/retry-started').length, 0, 'no retry-started after abort')
}

// ---------------------------------------------------------------------------
// Config: empty schedule is rejected; defaults match the requested table.
// ---------------------------------------------------------------------------
{
  assert.throws(() => apply(makeCtx(), { delaysMs: [] }), /delaysMs must be a non-empty array/)
  assert.deepEqual(DEFAULT_CODES, ['QUOTA', 'insufficient_quota', 'RATE_LIMIT'])
  assert.deepEqual(DEFAULT_DELAYS_MS, [0, 1000, 5000, 10000, 30000, 60000, 120000, 300000, 600000], 'immediately + 1s/5s/10s/30s/1m/2m/5m/10m')
}

console.log('llm-retry-fixed test: all assertions passed')
