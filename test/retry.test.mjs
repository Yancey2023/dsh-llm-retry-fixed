/**
 * Unit tests for dsh-llm-retry-fixed — the fixed-schedule quota retry
 * executor on the agent/request-error recovery extension point.
 *
 *   QUOTA failure        -> immediate retry decision, durable llm/retry events
 *   non-QUOTA failure    -> passed through to next() untouched
 *   retry counting       -> schedule consumed in order, then give up
 *   restart durability   -> prior retries derived from the session log
 *                           through the OFFICIAL session-projection API
 *   abort during wait    -> no retry decision (loop aborts the turn)
 *   config validation    -> empty schedule rejected
 *   NEW Session API      -> `session.events` is gone (undefined); the
 *                           executor must never touch it (the historical
 *                           "Cannot read properties of undefined (reading
 *                           'findLast')" crash) and counts retries via
 *                           `ctx.sessionProjections.stateOf(...)` instead
 */
import assert from 'node:assert/strict'
import { apply, DEFAULT_CODES, DEFAULT_DELAYS_MS, PROJECTION_KEY } from '../lib/index.js'

/**
 * Minimal in-memory projection registry with the REAL contract shape of the
 * current DSH `sessionProjections` service: `register({key, stateVersion,
 * stateSchema, init, apply})`; `stateOf(session, key)` returns the fold of
 * `init` over `session.snapshotEvents()` (lazily materialized on read —
 * exactly like the real cells, which fold the in-memory log on first touch).
 */
function makeProjections() {
  const units = new Map()
  return {
    register(def) {
      units.set(def.key, def)
      return () => units.delete(def.key)
    },
    stateOf(session, key) {
      const def = units.get(key)
      if (def === undefined) return undefined
      let state = def.init()
      for (const event of session.snapshotEvents()) state = def.apply(state, event)
      return state
    },
  }
}

function makeCtx() {
  let handler = null
  const ctx = {
    sessionProjections: makeProjections(),
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

/**
 * Fake agent whose session mirrors the CURRENT DSH Session surface: events
 * are append-only and read through `snapshotEvents()`; there is NO `.events`
 * property — exactly like the real 0.1.3-alpha.2 Session, where the removed
 * property is `undefined` and reading it was the historical findLast crash.
 */
function makeAgent() {
  const log = []
  return {
    session: {
      events: undefined, // the removed property — the executor must not read it
      append(type, data) {
        log.push({ type, data })
        return { type, data, seq: log.length - 1 }
      },
      snapshotEvents() {
        return log.map((e) => ({ ...e }))
      },
    },
  }
}

const retriesOf = (agent) => agent.session.snapshotEvents().filter((e) => e.type === 'llm/retry')
const startedOf = (agent) => agent.session.snapshotEvents().filter((e) => e.type === 'llm/retry-started')

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
// NEW-API REGRESSION: the current Session has no `.events` — the executor
// recover() path must work through the projection, not session.events.
// ---------------------------------------------------------------------------
{
  const ctx = makeCtx()
  apply(ctx, {})
  const agent = makeAgent()
  assert.equal(agent.session.events, undefined, 'the removed session.events property is absent (real 0.1.3-alpha.2 Session shape)')
  const next = nextCaller()
  const signal = new AbortController().signal
  const result = await ctx._dispatch({ agent, turn: 3, step: 2, provider: 'relayfor', failure: quotaFailure(), signal }, next)
  assert.deepEqual(result, { kind: 'retry' }, 'QUOTA failure retries')
  assert.equal(next.called(), 0, 'next not consulted for QUOTA')
  const retry = retriesOf(agent)[0]
  assert.ok(retry, 'durable llm/retry event appended')
  assert.equal(retry.data.retry, 1)
  assert.equal(retry.data.delayMs, 0, 'first retry is immediate')
  assert.equal(retry.data.maxRetries, DEFAULT_DELAYS_MS.length, 'maxRetries = schedule length')
  assert.equal(retry.data.mode, 'normal')
  assert.equal(retry.data.turn, 3)
  assert.equal(retry.data.step, 2)
  assert.equal(retry.data.provider, 'relayfor')
  assert.equal(retry.data.failure.code, 'QUOTA')
  assert.equal(startedOf(agent).length, 1, 'retry-started appended after the wait')
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
  const result = await ctx._dispatch({ agent, turn: 1, step: 1, provider: 'p', failure: { message: '429 rpm exhausted', code: 'RATE_LIMIT' }, signal }, next)
  assert.deepEqual(result, { kind: 'retry' }, 'RATE_LIMIT failure retries')
  assert.equal(next.called(), 0)
  const retry = retriesOf(agent)[0]
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
  const retry = retriesOf(agent)[0]
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
    sessionProjections: makeProjections(),
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
  const retry = retriesOf(agent)[0]
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
  assert.equal(agent.session.snapshotEvents().length, 0, 'no events appended')
}

// ---------------------------------------------------------------------------
// Schedule order and give-up: 3 retries (0/1/2ms), 4th failure gives up.
// Prior-retry counting goes through the projection state, derived from the
// session log via snapshotEvents() — never through session.events.
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
    const retry = retriesOf(agent)[i]
    seen.push(retry.data.delayMs)
  }
  assert.deepEqual(seen, [0, 1, 2], 'delays consumed in schedule order')
  assert.equal(retriesOf(agent).length, 3)
  // 4th failure: schedule exhausted -> next() -> the turn fails.
  const next = nextCaller()
  const result = await ctx._dispatch({ agent, turn: 5, step: 1, provider: 'p', failure: quotaFailure(), signal }, next)
  assert.equal(result, undefined)
  assert.equal(next.called(), 1, 'gives up after the schedule is exhausted')
  assert.equal(retriesOf(agent).length, 3, 'no further retry events')
}

// ---------------------------------------------------------------------------
// A new turn after retries: the projection key embeds (provider, policyKey)
// — the OLD turn's retries stay out of the NEW turn's count, and the new-turn
// failure starts a fresh chain. (The fold itself resets on step/start and
// turn/end exactly like the stock llmRetry unit.)
// ---------------------------------------------------------------------------
{
  const ctx = makeCtx()
  apply(ctx, { codes: ['QUOTA'], delaysMs: [0, 1, 2] })
  const agent = makeAgent()
  const signal = new AbortController().signal
  // Two retries in turn 5.
  for (let i = 0; i < 2; i += 1) {
    await ctx._dispatch({ agent, turn: 5, step: 1, provider: 'p', failure: quotaFailure(), signal }, () => {})
  }
  assert.equal(retriesOf(agent).length, 2, 'two retries in turn 5')
  // Turn 6 (after turn/end + turn/start): fresh chain — retry 1 again.
  agent.session.append('turn/end', { turn: 5, reason: { kind: 'completed' } })
  agent.session.append('turn/start', { turn: 6 })
  const next = nextCaller()
  const result = await ctx._dispatch({ agent, turn: 6, step: 1, provider: 'p', failure: quotaFailure(), signal }, next)
  assert.deepEqual(result, { kind: 'retry' }, 'new turn still retries')
  const all = retriesOf(agent)
  assert.equal(all[2].data.retry, 1, 'new turn starts at retry 1 (old turn retries do not carry over)')
  assert.equal(all[2].data.turn, 6)
}

// ---------------------------------------------------------------------------
// Restart durability: prior retries derived from the session log — a fresh
// executor instance (process restart) continues the same chain, same id.
// The projection is materialized from snapshotEvents(), so no process-local
// memory is involved (exactly like the real dsh-session-projection cells).
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
  const retries = retriesOf(agent)
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
  assert.equal(retriesOf(agent).length, 1, 'durable event survives the abort')
  assert.equal(startedOf(agent).length, 0, 'no retry-started after abort')
}

// ---------------------------------------------------------------------------
// Config: empty schedule is rejected; defaults match the requested table.
// ---------------------------------------------------------------------------
{
  assert.throws(() => apply(makeCtx(), { delaysMs: [] }), /delaysMs must be a non-empty array/)
  assert.deepEqual(DEFAULT_CODES, ['QUOTA', 'insufficient_quota', 'RATE_LIMIT'])
  assert.deepEqual(DEFAULT_DELAYS_MS, [0, 1000, 5000, 10000, 30000, 60000, 120000, 300000, 600000], 'immediately + 1s/5s/10s/30s/1m/2m/5m/10m')
  assert.equal(PROJECTION_KEY, 'llmRetryFixed', 'own projection key (never collides with the stock llmRetry unit)')
}

console.log('llm-retry-fixed test: all assertions passed')