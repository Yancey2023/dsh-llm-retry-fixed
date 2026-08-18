# dsh-llm-retry-fixed

DSH host plugin: retry LLM **quota-exhaustion / rate-limit** failures on a
**fixed schedule** before giving up, instead of the stock policy's default
2 fast retries that break long-running turns (the recurring `429` /
`rpm exhausted` / `quota_exceeded_error` on rate-limited relays).

The stock `@deepseek-ai/dsh-llm-retry` policy (mounted by `dsh-base`) retries
only its `retryableCodes` (RATE_LIMIT / SERVER / TIMEOUT / TRANSPORT / …)
with exponential backoff and a default `maxRetries` of **2** — so a
`RATE_LIMIT` failure (which is how the harness classifies a 429 whose wording
`isQuotaExceededError` cannot confirm as quota, e.g.
`{"message":"rpm exhausted","type":"quota_exceeded_error","code":"8"}`) dies
after two ~500ms/~1s retries and the turn breaks, and a `QUOTA`/`insufficient_quota`
failure dies immediately. This plugin hooks the **same** agent-loop recovery
extension point (`agent/request-error`, **prepended** so it decides before the
stock policy for its codes) and retries those failures on the schedule:

| retry | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| delay after failure | 立刻 | 1s | 5s | 10s | 30s | 1min | 2min | 5min | 10min |

Ten attempts in total; only after the 10-minute retry fails does the turn
fail. Everything is durable and cancellable exactly like the stock policy:
the same `llm/retry` / `llm/retry-started` session events (the conversation
UI renders the retry chain as model-retry notices), prior retries counted
from the session log so the schedule survives restarts, and aborts during a
wait end the turn without a retry. Every failure outside the matched code
set falls through untouched to the stock policy.

## Install (web profile)

```bash
pnpm install          # devDeps (schemastery) for the test
pnpm test             # retry-schedule unit tests

dsh plugin --profile web add /absolute/path/to/dsh-llm-retry-fixed
# restart the web app:
dsh --profile web
```

The bundle patch inserts the host row with the schedule in
`cordis.patch.yml` (codes `QUOTA` + `insufficient_quota` + `RATE_LIMIT`,
delays 0/1000/5000/10000/30000/60000/120000/300000/600000 ms). Tune either
value there (this is the config the running instance actually uses).

## Uninstall

```bash
dsh plugin --profile web remove dsh-llm-retry-fixed
# restart — the stock exponential policy alone handles retries again.
```

## Behavior notes

- Matched codes:
  - `QUOTA` — the harness-mapped 429 quota code (via `isQuotaExceededError`
    in `dsh-llm-pi-ai`), e.g. "Allocated quota exceeded…";
  - `insufficient_quota` — the raw provider code of
    `{"code":"insufficient_quota", …}`;
  - `RATE_LIMIT` — the harness code for any other 429. This is the crucial
    one in practice: DeepSeek/sensenova rate-limited relays return
    `429: {"message":"rpm exhausted","type":"quota_exceeded_error","code":"8"}`,
    whose glued `quota_exceeded_error` wording and "rpm exhausted" message
    never match `isQuotaExceededError`'s regexes, so the harness classifies
    it as `RATE_LIMIT` and the stock policy would burn its 2-retry budget on
    it. Matching `RATE_LIMIT` is what keeps those turns alive on the long
    fixed schedule.
  Every other failure falls through to the stock policy / default error path.
- The listener is registered with `prepend`, so for its codes it decides
  before the stock policy's listener, which is registered earlier by
  `dsh-base`. In the request-error waterfall a listener that returns a retry
  decision without calling `next()` vetoes the chain — prepending prevents
  the stock policy from consuming the first failures with its own 2-retry
  exponential budget before this schedule can take over.
- The retry chain is keyed by `(turn, step, provider, policyKey)`, so
  different turns/steps retry independently and a process restart continues
  the same chain instead of restarting the schedule.
- Retries happen while the agent's turn is still open; the conversation UI
  shows each scheduled retry as a model-retry row with the remaining delay.
- The provider's own `Retry-After` header is NOT honored (fixed schedule
  wins; stock policy still honors it for its codes).
