# The Relay

All numbers come from real calls to Groq's free tier (`openai/gpt-oss-20b`) and can be recomputed from
`results.csv` (214 rows) and the files named in brackets. No row comes from a stand-in.

## 1 What I built and why

A Python pipeline that answers questions only from the three papers. (1) Route the question to the papers it
names and set an answer-length budget. (2) Exact-match cache. (3) **Tool call** `retrieve_passages`: BM25 + dense
search fused with RRF, reference lists and figure-axis noise filtered, cross-encoder re-rank, one search per paper
in parallel for cross-paper questions. (4) **Branch**: semantic-cache hit / daily budget gone / compact mode under
token pressure / full answer. (5) Stream the answer through a **sentence gate**. (6) **Loop**: one repair call if
nothing verifiable came back.

**The decision that shaped it most: tokens, not requests, are the scarce resource.** The per-day token limit
(200K) runs out after ≈140 answers, while the request limit (1,000/day) is never close. So the system optimises
tokens per answer: extractive compression (context 1,317 → 733 tokens, 91% of key facts kept), a lean prompt,
short `[E1]` citation tags, length caps by question type, and admission control that paces to Groq's refilling
token bucket. Second decision: **stream, but show only verified sentences.** A sentence reaches the person only
once it cites a retrieved passage and every number in it appears in that passage. The cost is that the first
visible token is the end of the first verified sentence, not the first raw token.

## 2 The real service and my stand-in

**Table 1: Limits found in the real service**

| Limit | Value | How I found it | Confidence |
|---|---|---|---|
| Requests per minute | 30 | Groq docs; never reached (my pacing stays below) | Medium |
| Tokens per minute | 8,000 | `x-ratelimit-limit-tokens`; 429 body "Limit 8000" | High |
| Requests per day | 1,000 | `x-ratelimit-limit-requests` | High |
| Tokens per day | 200,000 | Real 429 body: "Limit 200000, Used 199159, Requested 1810" | High |
| Requests in flight | No cap up to 8; TTFT p50 ≈450 → 596 ms at 8 | Bursts of 1/2/4/8 | Medium |
| What it returns | HTTP 429, JSON `type: tokens`, "Please try again in 525ms"; `retry-after` rounded up to seconds; `x-should-retry: false` on the daily limit | 17 live captures (`data/calibration/live_429_*.json`) | High |
| Time until it recovers | Continuous refill, not a window: 133 tok/s (TPM/60), 2.31 tok/s per day (200K/86,400), 1 request per 86.4 s, so it depends on the next request's size | `reset-tokens` 41.602 s = 5,547 used ÷ 133.3; daily "try again in 6m58.6s" = 969 missing ÷ 2.315; `reset-requests` 59m2.4s = 41 × 86.4 s | High |

**The model changed under me.** `llama-3.1-8b-instant`, my first choice, began returning 404 `model_not_found`
(Groq retired it for free tier on 16 Aug 2026). I moved to `gpt-oss-20b`, a reasoning model whose hidden reasoning
tokens come before the first visible token. **Prompt caching** is documented for gpt-oss, but 7 identical
1.3K–2.5K-token test prompts got `cached_tokens: 0` and were charged in full; only 2 of 83 load-test calls got 256
cached tokens. So I dropped a longer few-shot prompt (≈250 tokens billed every call) for a lean one.

**Stand-in.** Fitted to 24 paced real requests (`config/calibration_live.json`): first token = 400 ms + 0.095 ms
per prompt token, noise σ 167 ms, 0.64 ms per output token. Measured TTFT p50 was 403 / 466 / 519 ms at 169 /
845 / 1,969 prompt tokens. Groq's own `queue_time` is ≈300 ms at every size and dominates; prompt processing is
only 10 / 53 / 123 ms. **Least like the real service:** Groq's queue time is heavy-tailed (61–1,058 ms) and shared
with other users, while the stand-in draws Gaussian noise. Also, the real daily token limit is invisible in the
headers until its 429 arrives.

## 3 Results

All real calls, setting R5, the person's deadline 12 s. Percentiles are over `status == ok` rows. First token =
first verified text the person sees, measured from arrival, so cache hits count as answers.

**Table 2a**

| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |
|---|---|---|---|---|---|---|
| steady | real | 17 | 116.0 | 8.3 | 1099 / 1939 / 2028 (n=17) | 1106 / 1940 / 2031 (n=17) |
| burst | real | 8 | 3.0 | 142.0 | 1290 / 2336 / 2606 (n=8) | 1304 / 2347 / 2609 (n=8) |
| sustained_overload | real | 50 | 109.5 | 26.9 | 1135 / 8314 / 9638 (n=45) | 1149 / 8314 / 9665 (n=45) |
| rpm_halved | real | 19 | 86.6 | 12.5 | 2300 / 7357 / 7809 (n=12) | 2300 / 7374 / 7812 (n=12) |
| daily_budget | real | 16 | 67.5 | 13.3 | 986 / 1515 / 1671 (n=8) | 1009 / 1551 / 1721 (n=8) |
| cache_repeat | real | 17 | 77.3 | 12.4 | 911 / 1487 / 1820 (n=17) | 911 / 1487 / 1820 (n=17) |
| naive_retry | real | 27 | 56.4 | 27.7 | 1053 / 1946 / 2543 (n=23) | 1156 / 1948 / 2543 (n=23) |

**Table 2b**

| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers |
|---|---|---|---|---|---|---|---|
| steady | 100.0 | 0.0 | 0.0 | 0.00 | 0 | 1337 | $0.0874 |
| burst | 100.0 | 0.0 | 0.0 | 0.00 | 1 | 1043 | $0.0894 |
| sustained_overload | 90.0 | 10.0 | 0.0 | 0.00 | 3 | 820 | $0.0460 |
| rpm_halved | 63.2 | 36.8 | 0.0 | 0.00 | 3 | 968 | $0.0699 |
| daily_budget | 50.0 | 50.0 | 0.0 | 0.00 | 0 | 1279 | $0.0768 |
| cache_repeat | 100.0 | 0.0 | 0.0 | 0.00 | 0 | 1323 | $0.0611 |
| naive_retry | 85.2 | 14.8 | 0.0 | 0.37 | 1 | 1026 | $0.0651 |

Cost uses Groq's list price for gpt-oss-20b; the free tier actually charged $0. Groq returned **zero 429s** in
every scenario where my admission control was on. *daily_budget* uses an app-level cap of 5 model calls to trigger
the same path as Groq's real per-day 429, which I also hit for real during the eval. *naive_retry* turns admission
control off and retries quickly.

**A number I could not have guessed:** the binding daily limit is tokens, not requests. At ≈1,400 tokens per
answer, 200,000 tokens/day ≈ **140 answers/day**, although the request limit says 1,000.

**Answer quality:** 20 questions I wrote (16 answerable, 4 not covered by the papers), graded by an LLM judge
(`gpt-oss-120b`) plus a required citation to the right paper. I graded the 20 R5 answers myself: I agree with the
judge on **19 of 20**, versus 14 of 20 for a keyword-matching rule, which wrongly failed answers that left out
gold details the question never asked for. The one disagreement shows the judge is lenient: on "did learned
positional embeddings beat sinusoidal ones, and why sinusoidal?", it marked an answer correct while its own reason
said the answer had wrongly claimed the paper gives no rationale. I graded it wrong.

## 4 Where the time goes

Slow request `eval_S1_256_3_no_q_catB_01` (`data/live_v1/traces.jsonl`): total 1,469 ms, first token 1,449 ms.

| Waiting: my queue | **Waiting: Groq queue** | Working: retrieval + compression | Working: Groq prompt | Working: Groq generation (56 of 124 tokens hidden reasoning) | Network (remainder) | Retrying |
|---|---|---|---|---|---|---|
| 0.1 ms | **1,058 ms** | 6 ms | 156 ms | 136 ms | ≈113 ms | 0 |

72% of the time was waiting inside Groq, and **the delay was all before the first token**. I know because Groq
reports its own queue, prompt and completion times in `usage`, and they add up with my `perf_counter` spans to
within ≈110 ms of network time. The slowest latency-probe request had the same shape: 949 of 1,245 ms was Groq queue.

## 5 When a limit is hit

**Per minute.** Admission control models Groq's limits as refilling buckets, fitted exactly to its reset headers.
A request that cannot start before the person's 12 s deadline is **dropped at once** ("at capacity, retry in a
minute") instead of waiting for nothing; one that can start in time is **delayed**. Above 75% bucket use,
**everyone gets a weaker answer**: 2 passages instead of 4 and a smaller `max_tokens`. The reason, measured: after
a 429 Groq said `retry-after: 14` s, but a small request was accepted after **1.4 s**, so shrinking the request is
≈10× faster than waiting to resend it. In sustained overload (≈27/min), 45 of 50 were answered, 5 dropped, and 0
got a 429. The longest wait that still got an answer was 8.5 s, and 19 of the 50 were repeats served from cache in
0 ms. With the budget halved, 7 of 19 were dropped, several in under 0.7 s because their wait would have passed
the deadline. **Cost of compact mode:** 13 of 18 compact calls ended in "the papers don't say", versus 2 of 8
full calls. Under pressure, people are told "not covered" more often, rather than given a wrong answer.

**Per day.** Groq's headers do not report daily tokens, so the system learns at the 429 (`x-should-retry: false`).
It never retries. Every later request gets an immediate, **zero-token** reply with the wait from `retry-after` and
the top passages, and the system re-opens after `retry-after` because Groq refills continuously. This happened for
real in my eval: 12 requests in a row were answered this way in 1–87 ms. In *daily_budget*, 8 of 16 got that
reply in ≈0.5 s, while 3 repeat questions were still **answered from the cache** with no budget.

## Retrieval: quality versus cost

Default PDF extraction glued words ("newstate-of-the-artBLEUscoreof28.4"). Lowering pdfplumber `x_tolerance` from
3 to 1.5 raised the words extracted from the Attention paper's first 12 pages from 1,703 to 5,755 and cut glued
words from 310 to 5 (`scripts/check_extraction.py`). Everything below uses the clean index.

**Table 3**

| Setting | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |
|---|---|---|---|---|---|---|---|
| R5: 450 tok, 10+10→4 RRF, routed, re-rank yes, compressed | 20 | 50.0 | 25.0 | 20.0 | 1326 | 1219 / 1673 | 1258 / 1673 |
| S3: 450 tok, 8→4 via RRF, re-rank no | 20 | 55.0 | 20.0 | 20.0 | 1664 | 768 / 1570 | 768 / 1691 |
| S1: 256 tok, 3 pieces, re-rank no | 20 | 55.0 | 5.0 | 20.0 | 1619 | 692 / 1451 | 707 / 1514 |

The remaining share in each row is "refused wrongly": 5% for R5 and S3, 20% for S1. Retrieval alone (no model
calls): hit@k 0.625 / 0.562 / 0.500 and MRR 0.594 / 0.406 / 0.385 for R5 / S3 / S1.

**Choice: R5.** S3 and S1 are one question better (within noise at n=20) and ≈0.5 s faster to the first token,
because re-ranking costs CPU time. But R5 uses **20% fewer tokens per answer** (1,326 vs 1,664), and tokens are
the binding limit, so R5 serves ≈25% more people per minute and per day. **Near the limit** I switch to the
2-passage compact mode (cost measured in §5). I did **not** build a "refuse without calling the model" threshold:
the lowest-scoring answerable question (re-rank score −2.68) scored below the highest unanswerable one (−0.73),
so any threshold would refuse real questions.

## How this library will mislead a naive retriever

1. **Glued PDF text.** For *"What BLEU did the big Transformer get on EN-DE?"*, the keyword "BLEU" no longer
   matched, and my own number check rejected the correct 28.4 because it sat inside a glued word. Fix: a tighter
   `x_tolerance`, measured before rebuilding.
2. **Identifiers vanish.** The BM25 tokenizer drops tokens containing `_`, so *"What are d_model, h, d_k and d_v
   in the base Transformer?"* searched for almost nothing, and all three settings refused it in my first run.
   Fix: expand to "d model dmodel".
3. **Numbers-only and reference chunks.** Figure axes ("0 10 20 30 40 50") and reference lists look relevant to
   numeric or name-heavy questions. For the d_model question, S1 retrieved two Moshi chart-axis chunks (pages
   62–63) for an Attention-paper question. Fix: filter references and numbers-only prose; route by paper name.

## 6 Retries

Per-minute 429: wait Groq's `retry-after` only if the answer can still arrive before the person's deadline (at
most 2 retries). Per-day 429: never retry (Groq itself says `x-should-retry: false`). 5xx or timeout before the
first token: 1 retry with jitter. Never resend after tokens have streamed, which would bill them twice.
**Where retrying made it worse:** with admission off and quick retries (*naive_retry*, ≈28/min), Groq returned
**14 real 429s**. 5 requests retried twice each and **4 of those 5 still failed**, so 10 extra calls (counted
against the daily request limit) bought 1 extra answer. At the same rate with admission control on, there were 0
429s and 0 retries. (Turning the budget off also pushed that run into compact prompts, so its token figures are
not comparable.)

## 7 What the person sees

At 3pm the daily budget is gone. The waiting person gets this at once, with no model call and no tokens (wait and
clock time are filled from Groq's `retry-after`; the values here are an example):

> Live answers are paused: today's model budget is used up. It frees up again in about 0h 07m (≈15:10 UTC, from
> Groq's retry-after; capacity refills continuously); after that answers return slowly (about one every 10
> minutes, as Groq refills ~2.3 tokens/s). Please ask again then. Meanwhile, these are the most relevant passages
> (not checked by the model):
> - [arXiv:2410.00037v1 p.10 §3.3.1 Architecture] "…"

"About one every 10 minutes" comes from the measured refill: ≈1,400 tokens per answer ÷ 2.31 tokens/s ≈ 600 s.

## 8 What I'm unsure about, what I left out, what I learned

**Least certain:** that Groq's limits stay as measured. They already changed under me once, when the model was
retired mid-project. If the daily refill rate or the model's reasoning-token use changes, "≈140 answers/day" and
the compact-mode threshold are wrong. The system would then hit the daily wall earlier with no warning, because
the headers do not expose daily tokens. **Left out on purpose:** a persisted daily-token ledger that would see the
daily wall coming (the headers don't provide one, and it needs state across restarts); the "refuse without the
model" threshold (measured: no safe value); and reliance on prompt caching (measured: almost no hits).
**New to me:** a rate limit can be a bucket that refills continuously rather than a counter that resets. So "when
does it recover" has no single answer: it depends on the size of the next request, and sending a smaller request
can beat waiting for the advertised `retry-after`.



## How I worked with the assistant

i worked with antigravity and cluade. The prompts that mattered most: asking
whether my first version was ready to submit (the review found my API key inside the zip, a synthetic 429 sample
presented as measured, and Table 3 graded on stand-in echoes); "no mock or simulated results, real API calls only";
and sharing my raw eval answers and traces, which led to the four retrieval fixes above. I caught it being wrong
twice: it said a test fix was already in its copy, but the next zip brought the failing test back; and its
default model had been retired, which my first real call exposed with a 404. Where I overruled it: it said the
overload rows could come from the stand-in; I required every reported number to be real, so all seven load shapes
ran against Groq behind admission control. The measurements, my 20 human grades, and the decisions they drove
(switching model, lean prompt, choosing R5, no abstain threshold) are mine.
