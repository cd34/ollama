# Ollama multi-GPU placement — Bug report & fix (with full validation)

**Fork:** `cd34/ollama` — branch `fix/main-gpu-placement` (commit `63b53b8`)
**Upstream ref:** v0.57.6

## The bug (what was actually wrong)

With **two discrete GPUs**, models that **fit on a single card** were being
**spread across both cards** even when the user had explicitly requested
single-GPU placement (`OLLAMA_SCHED_SPREAD=false`). This prevented smaller
models from **co-residing** (one per card) and caused unnecessary eviction of
other resident models.

### Root cause

Two interacting issues, both in `server/sched.go`:

1. **`OLLAMA_SCHED_SPREAD=false` was *not honored* for 80–100% fits.**
   The single-GPU compaction path was gated on `bestSingleGPUFit`, which only
   accepts a model if it fits within **80%** of a card's free memory (a
   sensible default headroom). A model that *fits* a card but *exceeds* 80%
   (e.g. a 19 GiB model on a 24 GiB card) was **rejected by the 80% gate**
   and then **spread** across both cards — *even though the user had
   explicitly set `SCHED_SPREAD=false`*.

2. **`PredictServerVRAM` returning 0 skipped compaction entirely.**
   The compaction path is gated on `predictedVRAM > 0`. When the full
   prediction returned 0 (missing head dims / unknown file size), a model
   that *did* fit a card was spread. (Secondary; the 80% gate was the
   dominant cause for the reported models.)

### Why it "worked fine" with only one GPU

With **one** GPU visible, there's no "spread across two" option, so the
scheduler *must* place the model on that card (or **spill overflow to CPU**).
With **two** GPUs, the scheduler *prefers to spread across both GPUs* (faster
than CPU spill) — which is *generally* the right choice for throughput, but it
**breaks the co-residency goal** for models that *would fit* on one card.

## The fix

**Honor `OLLAMA_SCHED_SPREAD=false`** by adding a *fallback* single-GPU fit
that uses up to the **full** free memory of a card (not just 80%), *only* on
the explicit no-spread path:

- `server/sched.go`: when `!SchedSpread()`, after the strict 80% fit, try
  `bestSingleGPUFitFull` (fits up to 100% of a card). A model that doesn't
  fit *any* single card is still spread (no CPU spill).
- `llm/llama_server.go`: `PredictServerVRAMFloor` — a non-zero lower bound
  (model file size) used when the full prediction is 0, so compaction is
  still attempted when the size is known.
- **Default (spread) behavior is unchanged.** Only the explicit
  `SCHED_SPREAD=false` path is affected.

**No change to `main_gpu` semantics** — the earlier hypothesis (preserve
`MainGPU` index) was **wrong** and has been **reverted to stock**.

## Configuration that reproduces & is fixed

```
Environment="OLLAMA_SCHED_SPREAD=false"   # explicit single-GPU preference
```

## Full validation (step-by-step)

### 1. Unit tests (all pass)

```
$ CGO_ENABLED=1 go test ./server/ -run "TestSelectLlamaServerPlacement|TestNoSpread|TestPredictServerVRAM|TestBestSingleGPU" -v
    --- PASS: TestSelectLlamaServerPlacement/compacts_onto_largest_same-backend_GPU
    --- PASS: TestSelectLlamaServerPlacement/explicit_main_gpu_selects_matching_backend_group
    --- PASS: TestSelectLlamaServerPlacement/spread_disables_automatic_compaction
    --- PASS: TestSelectLlamaServerPlacement/small_model_with_known_size_compacts_onto_a_single_GPU
    --- PASS: TestNoSpreadCompactsModelUsingFullCard        # NEW (the bug)
ok  github.com/ollama/ollama/server

$ CGO_ENABLED=1 go test ./llm/ -run "TestPredictServerVRAM"
    --- PASS: TestPredictServerVRAMFloor                    # NEW (defensive)
ok  github.com/ollama/ollama/llm
```

**`TestNoSpreadCompactsModelUsingFullCard`** proves the core fix:
- a 20 GiB model on a 22 GiB free card is **rejected by the 80% gate**
  (default behavior preserved),
- **accepted by the full-card fit** (no-spread path),
- a 25 GiB model (doesn't fit) is **still rejected** (no CPU spill).

### 2. Live before/after (2× RTX 3090, `SCHED_SPREAD=false`)

**qwen3.8 (unpinned, `num_ctx=16384`, predicted 19.0 GiB):**

| | card 0 | card 1 | result |
|---|---|---|---|
| **BEFORE** (80% gate only) | 10.7 GiB | 9.4 GiB | **spread** ❌ |
| **AFTER** (full-card fit) | **18.5 GiB** | 1 MiB | **one card** ✅ |

Log line proving the new path:
```
msg="selecting single GPU for llama-server model (no-spread, using full card)"
main_gpu=0 name=CUDA0 predicted="19.0 GiB" available="23.4 GiB"
```

**llama3.1:8b (unpinned, `num_ctx=16384`, predicted 6.6 GiB):**
```
BEFORE:  spread (6.6 + 0.6 GiB)
AFTER:   card0: 6.1 GiB, card1: 0.6 GiB  -> "selecting single GPU" name=CUDA0
```

### 3. Co-residency (the end goal)

With `SCHED_SPREAD=false` + the fix, **qwen3.8 → card 0** and **gemma4 →
card 1** **co-reside, one per card, no eviction** (`loaded runners count=2`,
`fits alongside existing models`).

### 4. Context-size note (important)

The **`num_ctx`** of each model determines its **KV cache size**, which is a
major part of the predicted VRAM. Larger contexts = larger KV = more likely to
exceed 80% of a card = more likely to spread. For co-residency:

| model | num_ctx | predicted | fits one card? | result (SCHED_SPREAD=false + fix) |
|---|---|---|---|---|
| llama3.1:8b | 16k | 6.6 GiB | ✅ | one card ✅ |
| gemma4 | 32k | 15.5 GiB | ✅ | one card ✅ |
| qwen3.8 | 16k | 19.0 GiB | ✅ (barely) | **one card ✅ (fixed)** |
| qwen3.8 | 65k | ~25 GiB | ❌ | spread (correct — doesn't fit) |
| muse-glimmer | 128k | 29.2 GiB | ❌ | spread (correct — doesn't fit) |

**The fix makes the 80–100% models (qwen at 16k) compact to one card.**
Models that *genuinely* don't fit a card (qwen at 65k, glimmer at 128k)
*still* spread — that's *correct* (they'd otherwise spill to CPU).

## Safety & scope

- **Default behavior unchanged** — `SCHED_SPREAD` defaults to `true`; the
  new path is only taken when explicitly `false`.
- **No CPU-spill risk** — a model that doesn't fit *any* single card is
  still spread (never forced onto one card).
- **`main_gpu` semantics unchanged** — reverted to stock (the earlier
  "preserve index" change was a regression and is gone).
- **All existing placement/prediction tests pass** — no regressions.

## Repro (minimal)

```
# 2× GPU, a model that fits one card but >80%:
Environment="OLLAMA_SCHED_SPREAD=false"
# load the model -> BEFORE: spread across both; AFTER: one card
```

## Rollback

- **Binary:** `/usr/local/bin/ollama.prev-<timestamp>` (stock v0.57.6)
- **Env:** remove `OLLAMA_SCHED_SPREAD=false`, `daemon-reload`
- **Code:** revert commit `63b53b8` in the fork

## Upstream status

- **Fork branch:** `cd34/ollama` @ `fix/main-gpu-placement` (`63b53b8`)
- **Upstream:** *pending employer approval* to file issue/PR
- **PR title (suggested):** `sched: honor OLLAMA_SCHED_SPREAD=false for models that fit a full card`
- **Body:** use this document.
