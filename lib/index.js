/**
 * dsh-llm-retry-fixed — provider-routed LLM request retry with a FIXED delay
 * schedule, for quota-exhaustion failures that the stock exponential policy
 * would give up on.
 *
 * The stock `dsh-llm-retry` retries only its `retryableCodes` with
 * exponential backoff (initialDelayMs × 2^(retry−1), jittered) and a default
 * `maxRetries` of 2 — so a quota/rate-limit failure (`QUOTA` mapped by the
 * harness, the raw provider `insufficient_quota`, or a 429 that the harness
 * classifies as `RATE_LIMIT`) dies after 2 fast retries and breaks the
 * turn. This executor hooks the SAME agent-loop recovery extension point
 * (`agent/request-error`) — as a PREPENDED listener, so it decides before
 * the stock policy for its matched codes — and retries those failures on a
 * fixed schedule:
 *
 *   immediately, 1s, 5s, 10s, 30s, 1min, 2min, 5min, 10min
 *
 * i.e. 9 retries (delays [0, 1000, 5000, 10000, 30000, 60000, 120000,
 * 300000, 600000] ms) — 10 attempts in total — and only then is the turn
 * allowed to fail. The retries are durable and cancellable exactly like the
 * stock policy's (the same `llm/retry` / `llm/retry-started` session events,
 * so the conversation UI shows the retry chain), prior retries are counted
 * from the session log (survives restarts), and every non-matched failure
 * (SERVER / TIMEOUT / TRANSPORT / …) is passed through untouched to the
 * stock policy.
 *
 * Mount: host composition of the web profile, via the bundle channel
 * (`dsh plugin --profile web add <this package>`); the bundle patch inserts
 * the row.
 */

import { randomUUID } from 'node:crypto'

export const name = 'llm-retry-fixed'
export const inject = ['agents']

/** Failure codes this executor retries. The harness maps 429 quota errors
 * ("Allocated quota exceeded …" / code "insufficient_quota") to "QUOTA" via
 * classifyPiAiError; the raw "insufficient_quota" code is matched too. And a
 * 429 that the harness cannot positively identify as quota — e.g. the
 * DeepSeek/sensenova "rpm exhausted" / type "quota_exceeded_error" body,
 * whose glued wording "quota_exceeded_error" and "rpm exhausted" never match
 * isQuotaExceededError's regexes — lands on the harness code "RATE_LIMIT".
 * That class is matched as well, so quota-type 429s are retried on the long
 * fixed schedule no matter which code surface they arrive on. */
export const DEFAULT_CODES = ['QUOTA', 'insufficient_quota', 'RATE_LIMIT']
/** Fixed retry delays in ms: immediately, 1s, 5s, 10s, 30s, 1min, 2min, 5min, 10min. */
export const DEFAULT_DELAYS_MS = [0, 1000, 5000, 10000, 30000, 60000, 120000, 300000, 600000]

/**
 * Install fixed-schedule quota retry on the agent loop's request recovery
 * extension point.
 * @param ctx - plugin context that owns the listener and active waits.
 * @param config - retry codes and delay schedule (defaults as above).
 */
export function apply(ctx, config = {}) {
  const codes = new Set(config.codes ?? DEFAULT_CODES)
  const delays = config.delaysMs ?? DEFAULT_DELAYS_MS
  if (codes.size === 0) throw new Error('llm-retry-fixed: codes must not be empty')
  if (!Array.isArray(delays) || delays.length === 0 || delays.some((ms) => !Number.isFinite(ms) || ms < 0)) {
    throw new Error('llm-retry-fixed: delaysMs must be a non-empty array of non-negative finite numbers')
  }
  const lifetime = new AbortController()
  const active = new Set()

  function track(operation) {
    const tracked = operation.finally(() => active.delete(tracked))
    active.add(tracked)
    return tracked
  }

  function policyKey() {
    return JSON.stringify(['fixed', [...codes].sort(), delays])
  }

  /** Wait `ms` (0 = immediately), resolving false when aborted. */
  function cancellableDelay(ms, signal) {
    if (ms <= 0) return Promise.resolve(true)
    return new Promise((resolve) => {
      if (signal.aborted) {
        resolve(false)
        return
      }
      const timer = setTimeout(() => {
        signal.removeEventListener('abort', onAbort)
        resolve(true)
      }, ms)
      function onAbort() {
        clearTimeout(timer)
        resolve(false)
      }
      signal.addEventListener('abort', onAbort, { once: true })
    })
  }

  async function backoff(agent, turn, step, failure, provider, retry, retryId, delayMs, signal) {
    const fusedSignal = AbortSignal.any([signal, lifetime.signal])
    if (fusedSignal.aborted) return undefined
    agent.session.append('llm/retry', {
      retryId,
      turn,
      step,
      provider,
      mode: 'normal',
      policyKey: policyKey(),
      retry,
      maxRetries: delays.length,
      delayMs,
      failure,
    })
    if (!(await cancellableDelay(delayMs, fusedSignal))) return undefined
    agent.session.append('llm/retry-started', { retryId, turn, step, retry })
    return { kind: 'retry' }
  }

  async function recover({ agent, turn, step, provider, failure, signal }, next) {
    if (failure === undefined || failure === null || !codes.has(failure.code)) return next()
    const key = policyKey()
    const prior = agent.session.events.findLast(
      (event) =>
        event.type === 'llm/retry' &&
        event.data.turn === turn &&
        event.data.step === step &&
        event.data.provider === provider &&
        event.data.policyKey === key,
    )
    const previousRetry = prior?.data.retry ?? 0
    if (previousRetry >= delays.length) {
      ctx.logger.warn(
        `llm-retry-fixed: provider "${provider}" ${failure.code} persisted through ${delays.length} fixed retries; giving up`,
      )
      return next()
    }
    const retry = previousRetry + 1
    const retryId = prior?.data.retryId ?? randomUUID()
    return backoff(agent, turn, step, failure, provider, retry, retryId, delays[retry - 1], signal)
  }

  const disposeListener = ctx.on(
    'agent/request-error',
    (payload, next) => {
      if (lifetime.signal.aborted) return Promise.resolve(undefined)
      return track(recover(payload, next))
    },
    // Prepended: for the matched codes this executor must decide BEFORE the
    // stock dsh-llm-retry listener. In the request-error waterfall a listener
    // that returns a retry decision without calling next() vetoes the rest of
    // the chain — if the stock reporter (registered earlier by dsh-base) ran
    // first it would burn its own 2-retry exponential budget for RATE_LIMIT
    // and only pass through after giving up. Prepend makes the fixed schedule
    // own QUOTA / insufficient_quota / RATE_LIMIT from the very first failure;
    // every other code still falls through to next() -> stock policy.
    { prepend: true },
  )
  ctx.effect(
    () => async () => {
      disposeListener()
      lifetime.abort(new Error('llm-retry-fixed plugin disposed'))
      await Promise.allSettled([...active])
    },
    'llm-retry-fixed: abort and drain active recovery',
  )
}
