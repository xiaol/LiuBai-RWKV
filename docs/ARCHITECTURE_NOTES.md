# RWKV-7 state vs self-attention — architecture notes

Discussion notes, 2026-10-07. Formulas are plain text (no LaTeX) so they read in a terminal.
Numbers for "our 3B" use the stage-1 model: 32 layers, width 2560, head size 64 → 40 heads per layer.

## 1. What the "state" is, compared with the KV cache

| | Self-attention | RWKV-7 |
|---|---|---|
| Memory between tokens | KV cache: every past token's key and value, in every layer | one fixed 64×64 matrix per head per layer (plus a one-token shift) |
| Size, our 3B | 2 × 2560 × 32 = 164k numbers per token → about 2.7 GB (bf16) at 8k context, growing with length | 32 × 40 × 64 × 64 = 5.2M numbers ≈ 10 MB (bf16), at any length |
| Write | append (k, v), never forget | decay the old state, erase what is stored along key k, write v at k (delta rule) |
| Read | softmax(q·K)·V: exact lookup of any past token by content | S·r: read back a compressed summary; whatever was overwritten is gone |
| Cost per generated token | grows with history | constant |
| Training cost | O(T²) (flash attention) | O(T) (chunked parallel kernel) |
| Position | explicit (RoPE) | implicit, through the decay |

Scale check: the whole RWKV state holds as many numbers as the KV cache of about 32 tokens
(5.2M / 164k). It is a compressed memory, not a store of 32 tokens, but that is the budget.

In short: attention keeps everything losslessly and looks it up; RWKV compresses everything into a fixed
memory and has to decide what to keep.

## 2. Is "token interaction" a disadvantage for the state?

Yes, in one specific way: **the state has to decide what to keep before it knows what will be asked.**

RWKV-7 is not free of token interaction. Its read unrolls to roughly

```
y_t  ≈  Σ_i  (decay from i to t) · v_i · (k_i · r_t)
```

a dot product between the current query and every past key, like attention. Two things differ:

1. **When the interaction is computed.** Attention keeps every token and compares the query against all of them
   at read time — it decides relevance once the question is known. RWKV folds each token into S on arrival;
   what the decay and the delta-rule erase removed is gone before anyone asks. A fixed state cannot losslessly
   hold arbitrary amounts of information; exact recall of N items needs memory that grows with N.
   (Formalised as the recall gap: Arora et al., Zoology / Based; Jelassi et al., "Repeat After Me".)
2. **How sharp the read is.** Softmax is nonlinear: it can pick one token out of 100k if its logit stands out.
   The state read is linear superposition: one head with 64-dim keys stores only about 64 nearly orthogonal
   key-value pairs cleanly; beyond that they interfere. The delta rule keeps the memory cleaner than older
   linear attention but does not raise that capacity.

Where the state wins:

- **State tracking.** A fixed-depth transformer cannot follow an evolving state over arbitrary length (parity,
  permutation composition, variable tracking): its circuit depth does not grow with the input (TC⁰ bound).
  RWKV-7's generalised delta rule recognises all regular languages with a constant number of layers (RWKV-7
  paper). Attention is good at lookup and weak at serial update; the state is the reverse.
- **Cost.** Constant per token, fixed memory at any length.
- **Recency and order** are built in through the decay.

Our own evidence of the recall gap:

- RWKV-Long-ROSA: frozen RWKV-7 scored 0 on needle-in-a-haystack at 32k–64k; a 0.1M-param pointer head that
  re-reads the input restored 0.95.
- WKVM: the routed span bank recovered recall only by keeping spans outside the state.

In both cases the reasoning was there; the information had been overwritten before it was needed.

Usual fix: keep the state for computation, add a small exact-retrieval path — a few attention layers
(Qwen3.5-style hybrids, about 3:1 delta-rule to attention), retrieval such as ROSA (RWKV-8), cross-attention to
a bounded token set such as the prompt, or repeating the prompt so the state reads it again knowing what to keep.

## 3. Pairwise vs triple interaction

**One layer: pairwise is provably weaker on some tasks. Across layers the gap mostly closes.**

- Standard attention is pairwise: score = q_t · k_i.
- Triple ("2-simplicial") attention scores the query against two tokens at once:

  ```
  score(t, i, j) = Σ_c  q_t[c] · k_i[c] · k'_j[c]        output mixes  v_i ⊙ v'_j
  ```

Theory (Sanford, Hsu, Telgarsky, NeurIPS 2023):

- Match2 — is there i with x_t + x_i = 0? One attention layer solves it easily.
- Match3 — is there a pair i, j with x_t + x_i + x_j = 0? One standard attention layer needs width or head
  count polynomial in sequence length; one triple-attention layer solves it at small size.

Empirics: Meta, "Fast and Simplex: 2-Simplicial Attention in Triton" (Roy et al., 2025) — under a fixed token
budget, better scaling exponent on math, code and reasoning; little gain on plain language modelling. Origin:
Clift et al., "Logic and the 2-Simplicial Transformer" (ICLR 2020).

Why depth compensates: layer 1 writes information about token j into token i's residual; layer 2 pairs t with
that enriched i. Induction heads work this way (two pairwise layers, three positions). Roughly, L layers reach
interactions of up to ~2^L tokens, at the cost of layers and residual width.

| Interaction | Training cost per layer | In practice |
|---|---|---|
| Pairwise (attention) | O(n²) | standard |
| Triple | O(n³) | windowed (second key local) or a few layers only |
| RWKV-7 state | O(n) | see below |

Where RWKV-7 sits: the write v_i k_iᵀ is pairwise, but every later token transforms it (section 4), so token
i's contribution at time t is shaped by all tokens in between. It is a many-token interaction, ordered and
sequential (hence state tracking), but fixed at write time — the read query cannot choose which tokens to
combine; it gets whatever survived.

| | Selects by content at read time | Higher-order across time | Capacity |
|---|---|---|---|
| Attention (pairwise) | yes | only by stacking layers | unbounded (cache grows) |
| Triple attention | yes, over pairs of tokens | yes, within one layer | unbounded, O(n³) cost |
| RWKV-7 state | no: pre-compressed | yes, through the chain of transitions | fixed |

Conclusion: pairwise-vs-triple is not the state's main weakness. The state is already beyond pairwise along
time; its problem is that it cannot select by content after the fact. A hybrid (state + small retrieval path)
targets the failure we actually measured (long-range recall); triple attention targets a compositional gap we
have not observed.

## 4. The chain of transitions, and looped transformers

Per layer and head, each token t does a read and an update:

```
read:    y_t = S_t · r_t
update:  S_t = S_{t-1} · A_t  +  v_t · k_tᵀ
         A_t = diag(w_t)  −  κ̂_tᵀ · (a_t ⊙ κ̂_t)
```

- `v_t k_tᵀ` is the write: store value v under key k.
- `A_t` is a 64×64 matrix built from the current token; it multiplies the whole existing memory.
  `diag(w_t)` decays each channel at its own rate; the rank-1 term erases what is stored along κ̂_t,
  by an amount set by a_t (delta rule).

Unrolled:

```
S_t = Σ_i  v_i · k_iᵀ · A_{i+1} · A_{i+2} · … · A_t
```

What token i wrote reaches token t only after every token in between has transformed it — that product is the
chain. Attention has no such chain: q_t·k_i links t to i directly.

Why order matters:

- Diagonal transitions commute (RWKV-6, Mamba-style decay): the product only records total decay per channel,
  so nothing order-dependent can be tracked.
- RWKV-7's rank-1 term makes A_t non-diagonal; A₁A₂ ≠ A₂A₁, so the product can rotate and mix the state.
- Example, three cups being swapped: each "swap 1 and 2" token applies a permutation; the product is the
  current arrangement. One sequential step per swap — exactly what the chain provides, and what a fixed-depth
  transformer lacks.
- How far this goes depends on the transition's eigenvalues; parity, for instance, needs eigenvalues near −1
  (Grazzi et al., ICLR 2025, "Unlocking state-tracking in linear RNNs through negative eigenvalues"). Check the
  RWKV-7 paper's construction before relying on a specific case.

Comparison with a looped transformer:

| | Looped transformer | RWKV-7 chain |
|---|---|---|
| What loops | the whole block, k times over depth | one transition per token, over time |
| Serial depth | number of loops, chosen at inference, can be adaptive | equal to sequence length, built in |
| The step | nonlinear (attention + MLP), same weights each loop | linear in S, matrix depends on the token |
| What each step sees | all tokens again, through attention | one new token plus the compressed past |
| More compute on the same input? | yes, loop more | no: each token is read once per layer |
| Training | loops run serially | parallel (chunked scan), because the update is linear in S |

A looped transformer thinks longer by re-reading everything; RWKV thinks while reading. Looped-transformer
results (Saunshi et al. 2025, "Reasoning with Latent Thoughts") show k loops behave roughly like k steps of
chain-of-thought. RWKV gets one serial step per token: depth grows with input length for free, but cannot be
added without adding tokens. The nonlinearity lives between layers (FFN) and in how A_t is built from the token,
not in the state update — that linearity is what makes parallel training possible.

The chain is also the recall problem: each A_j can erase part of token i's memory, and the later read cannot
undo it. Attention never edits the past; it only looks it up.

Practical consequences:

- More compute for RWKV → more tokens (chain-of-thought, repeating the prompt).
- More serial depth for attention → loop it or add chain-of-thought.
- A hybrid (state chain + small attention/retrieval path) gets both — the direction of the ROSA and WKVM work.

## 5. Relevance to the song LM

Lyric following needs retrieval of the right lyric line at each audio frame. In the whole-song format (G3) the
lyrics come thousands of high-entropy audio tokens before the frames that sing them; every audio token writes into
the same small state, consistent with G3's near-zero binding (shuffled − matched 0.10 nats). The section format
(R3.1) shortens that distance. This is not proof the architecture is the cause: YuE/YuE2 are transformers and
also needed section interleaving. A natural hybrid here: RWKV for the audio stream plus cross-attention to the
lyric/prompt tokens only (~500 tokens, bounded cache), keeping constant memory, streaming and state snapshots.

## 6. Closing the recall gap with the state, for music post-training

Caveat first: in the line-level format (R3.2) the line being sung was read 2–6 s earlier (50–150 frames). A 64×64
state per head holds one line over that distance easily, and R3.1 bound weakly (permuted gap 0.106 nats) even
with the stanza right before its audio. So the remaining problem may be *using* the text (learning text → sung
sound), not *keeping* it. Those need different fixes; measure first.

**Step 0 — diagnose (≈ 1 GPU-hour).** Linear probe on frozen hidden states: at each audio frame, predict the
lyric line (or phonemes) currently being sung, labels from the forced alignment (`data/lyric_align/fa`).
Plot accuracy against distance from the text, whole-song vs line format.

- Probe fails → the state does not hold the lyrics → recall problem → options 2–4.
- Probe succeeds but generation ignores the lyrics → usage problem → options 5–6.

Options, cheapest first:

1. **Put the text where it is needed (format only).** Line-level interleaving already does this; at inference the
   current line can always be local. Variants: insert the next line's text shortly before its frames, or repeat
   the whole lyrics in the header ("Just Read Twice", Arora et al. 2024). Costs a few hundred tokens.

2. **>>> RECOMMENDED FIRST: train the state to keep the lyrics — lyric-cursor auxiliary loss. <<<**
   See section 7 for the full design.

3. **Lyric memory heads: gate writes by token type.** In a few heads per layer, audio tokens do not write
   (k = 0, no erase) and decay is held near 1; only text tokens write, and SOS clears them. A protected lyric
   store the audio stream can read but not overwrite. Mask on k and a for the chosen heads, fine-tuned from the
   existing checkpoint. Capacity: 4 heads × 32 layers × 64×64 vs ~500 lyric tokens.

4. **State bank addressed by pointer.** Lyric recall is monotonic (line j after line j−1): snapshot the state after
   each lyric line; while singing line j, read snapshot j plus the live audio state. Pointer = the cursor from
   option 2, or SOS boundaries. WKVM / Multi-state RWKV idea applied to an ordered memory. More engineering.

5. **Distill from YuE2.** Same semantic token space, so YuE2's AR (3B transformer, exact attention to the lyrics)
   gives per-frame logits on the same documents; train RWKV on them with KL. Dense lyric-conditioned supervision
   on every frame; attention-to-state distillation (LoLCATs, Mamba-in-the-Llama) without a vocabulary mismatch.
   Cost: one teacher pass per document (~24 GB), prompt built in YuE2's format.

6. **RL on lyric accuracy (R4).** GRPO with a phoneme-error-rate reward (HeartTranscriptor on rendered 30 s
   excerpts) plus KL to the SFT model. Adds no capacity; makes the model use what it has. GRPO pipeline exists
   in RNN-StateTuning.

7. **If all fail: gated cross-attention to the lyrics.** A few cross-attention layers over lyric/prompt tokens
   only, zero-initialised gate (Flamingo-style) so the checkpoint is unchanged at start. Bounded cache (~500
   tokens), streaming and constant memory survive.

Not useful here: **state tuning** (learned initial state). It is one constant for all songs, so it cannot carry a
given song's lyrics, and it does not change the gates that decide what is kept. Helps format/style, not recall.

Suggested order:

| Step | What | Cost |
|---|---|---|
| 1 | Linear probe on R3.1 / R3.2a hidden states | ~1 GPU-hour |
| 2 | **Line-level format + lyric-cursor loss, continued from R3.1 (option 2)** | one R3.2-sized run |
| 3 | If the probe shows decay: lyric memory heads (option 3) | small patch, same run |
| 4 | YuE2 logit distillation (option 5) | teacher passes, 2 GPUs |
| 5 | GRPO with phoneme-error reward (option 6) | planned R4 |

Options 2 and 3 can share one run.

## 7. ★ Option 2 in detail: lyric-cursor auxiliary loss ★

**Why this one first.** It is the most direct way to *train the state* for recall: no architecture change, fits
continued training of R3.1/R3.2, and the labels already exist. Mothersuperior found the same kind of cursor
necessary to stop lyric drift when fine-tuning YuE2's AR — and lyric adherence (PER) is exactly where Suno leads.

**What it does.** A small head reads the hidden state at every audio frame and must say *which lyric unit is being
sung right now*. To answer, the state must still contain the lyrics at that frame. The gradient of this loss
flows back into the RWKV-7 gates that decide what the state keeps:

```
w  (decay)       how fast old content fades
a  (in-context learning rate / erase strength)
k  (write direction)
```

so the model learns a write policy that protects lyric content from being overwritten by the stream of
high-entropy audio tokens. The main LM loss alone gives this signal only weakly (0.1-nat binding in G3/R3.1).

**Targets (from data we already have).**

- MMS forced alignment (`tools/lyric_align_fa.py`, `data/lyric_align/fa`) gives per-line and per-word times.
- 25 Hz frames → per-frame label. Choices, from coarse to fine:
  - line index within the song (cheap, but positional),
  - word id / word index inside the current line,
  - phoneme or character sequence of the current line, scored with CTC (best: forces content, not position).
- Frames with no vocal (intro, instrumental breaks) → blank / ignore.
- For the real-song corpus, LRC timestamps give line times for free (see `tools/ingest_lrc_corpus.py`).

**Head and loss.**

```
h_t        = hidden state after the last block (before ln_out/head), audio positions only
cursor_t   = Linear(2560 → n_units)(h_t)                # or a 2-layer MLP, ≤ 5 M params
L_total    = L_LM  +  λ · L_cursor                       # λ ≈ 0.05–0.1
L_cursor   = CE(cursor_t, label_t)                       # word/line target
          or CTC(cursor over frames of a line, phonemes of that line)
```

- A second variant attaches the head to a middle layer too (e.g. layer 16), so lower layers are pushed to keep the
  lyrics, not only the top.
- Content, not position: the target must require the *words*. A pure line-index target can be solved by counting
  frames; use word/phoneme targets, or at least word identity, so the only way to win is to remember the text.

**Training.** Continue from `out/stage1_sec/rwkv-final.pth` (R3.1) on the line-level corpus
(`data/binidx/yue2_line_stage1_*`), same LR schedule as R3.2a (3e-5 → 3e-6). Needs: per-document frame labels
stored next to the binidx (a side array of int16 per audio token), a data-loader change to return them, and the
extra head and loss term in `training_step`. The head is dropped at inference — or kept, since it gives a free
live lyric cursor for section edits and streaming UI.

**Gates (all already measurable).**

1. Cursor accuracy on held-out songs, and its decay with distance from the text (should stay flat across a line).
2. Teacher-forced binding: `scripts/rwkv_eval_loss_sec.py` permuted − matched must clearly exceed R3.1's 0.106.
3. Audio: 20 shard-2 prompts rendered and scored — WER vs lyrics below R3.1's 1.05, transcript lyric-word overlap
   well above 10 % (originals 80 %).

**Risks.** λ too high costs audio LM quality (watch code-token CE on val); alignment noise on screamed / non-English
songs (filter by alignment score); the head can learn timing shortcuts unless targets are content-based.

## 8. New axes for the WKV state

Today: one 64×64 matrix per head per layer. Our 3B: 32 layers × 40 heads → 5.2M numbers (≈ 10 MB). Each added
axis targets a different weakness.

| Axis | Idea | Fixes | State size | Fit for music |
|---|---|---|---|---|
| Slot | M memories S[m]; a router (or SOS) picks which slot a token writes; reads mix slots | capacity, ordered recall | ×M | **best** |
| Modality | separate S_text / S_audio; audio never writes the text state | audio overwriting lyrics | ×2 | **best, simplest** |
| Time-scale | O(log T) states, Fenwick tree over time (log-linear attention, Guo/Yang 2025) | far recall, e.g. a returning chorus | ×log T | good, second stage |
| Micro-step | n Householder transitions per token (DeltaProduct, Siems/Grazzi 2025) | state tracking | ×1 (n× compute) | low |
| Order | 3-tensor state, write v⊗k⊗k' | triple interaction | ×64 at head 64 | skip |
| Nonlinear | state = MLP weights, one gradient step per token (TTT, Titans) | capacity per number | varies | later research |

Formulas:

```
slot:      S_t[m] = S_{t-1}[m] · A_t + v_t k_tᵀ      only for the slot(s) the router picks
           y_t    = Σ_m g_t[m] · S_t[m] · r_t
modality:  S_text  written by text tokens only, decay ≈ 1 during audio
           S_audio written by audio tokens, normal decay
           y_t = S_text · r_t + S_audio · r_t
time:      y_t = Σ_level λ_t[level] · S_t[level] · r_t
micro:     A_t = H_t,1 · H_t,2 · … · H_t,n
```

Prior work for the slot axis: MoM "Mixture-of-Memories" (Du et al. 2025), Multi-state RWKV, the WKVM span bank.

### 8.1 Prototype: slot + modality axes as protected heads (implemented)

Recommendation: combine modality + slot + the lyric-cursor loss (section 7). Text writes into a protected state
indexed by lyric line; audio keeps its own state; the cursor head says which line is being sung. "Remember all the
lyrics in 4,096 numbers per head" becomes "read the current line from its own slot". Memory stays bounded.

First prototype needs **no kernel change**: slots are existing heads.

- `src/model.py` (tools/RWKV-LM, recorded in tools/patches): `RWKV_SLOT_HEADS=K`, `RWKV_SLOT_M=M`.
  The last K heads of every layer are protected lyric memory, split into M slots of K/M heads.
- Hold rule, computed from token ids (`slot_hold_mask`), per token and head:

  ```
  hold = True  →  k = 0, erase term (kka) = 0, decay w = 1     (state passes through unchanged)
  audio tokens (codes, SOA, YUE2CODEC, EOA), header, EOD, separator 0  →  hold all K slot heads
  line text (SOS ... up to SOA)                                         →  write only slot (#SOS so far) mod M
  ```

- Reads are untouched: every token reads every head, so audio frames read the protected lyric heads.
- M = 1 → modality axis only. K = 0 → stock RWKV-7.
- The hold is implemented by masking the kernel inputs (k, kka, w), so the stock `rwkv7_clampw` kernel is reused
  and gradients flow normally through the non-held positions.
- Limits of this prototype: slot assignment is round-robin by line (line j and j+M share a slot), there is no learned
  router, and the protected heads are taken out of the general-purpose budget. A dedicated multi-slot kernel
  (state [H, M, N, N]) would remove the last limit.

Unit test `scripts/test_state_axes.py` (2026-10-09, K = 4, M = 2):

- mask: slot heads write only on their own line's text, hold on header / audio / markers; stock heads never hold;
- kernel: a held head gives identical output at two held steps with the same r (|Δy| = 0 vs 4.4 on a free head);
- 2-layer model forward + backward with the patch on.

### 8.2 Experiment plan (236M A/B setup, `scripts/train_arch_ab.sh`, line format)

| Run | Command | Question |
|---|---|---|
| baseline | `bash scripts/train_arch_ab.sh rwkv line GPU` | stock state |
| modality | `SLOT_HEADS=4 SLOT_M=1 bash scripts/train_arch_ab.sh rwkv line GPU` | does protecting text from audio help? |
| slot + modality | `SLOT_HEADS=4 SLOT_M=4 bash scripts/train_arch_ab.sh rwkv line GPU` | do per-line slots add to it? |
| attention reference | `bash scripts/train_arch_ab.sh attn line GPU` | how far from exact recall? |

Score with `scripts/arch_ab_eval.py --fmt line [--slot-heads K --slot-m M]`: code CE, binding gaps (shuffled − matched,
permuted − matched), first-50-frame CE. Success = permuted − matched clearly above the baseline at equal code CE.
If positive, port to the 3B (continue from R3.1 with K = 4 of 40 heads) together with the lyric-cursor loss.

### 8.3 Results (2026-10-10)

Setup: 236M (L12 D768, 12 heads of 64), from scratch, 1 epoch of the line-format corpus (144M tokens, 4,405 steps,
micro_bsz 4, ctx 8192, LR 6e-4 → 6e-5), one A100 each, ≈ 50 min (attention) / 70 min (RWKV). Scored on 100 held-out
line documents (`logs/arch_ab_eval_*.log`, `out/arch_ab/<run>/rwkv-final.eval_line.json`). Slot runs use K = 4 of 12
heads. CE in nats.

| Run | code CE matched | shuffled | permuted | no text | gap shuffled | **gap permuted** | first-50 gap permuted | lyric-text CE |
|---|---|---|---|---|---|---|---|---|
| RWKV-7 stock | **4.906** | 5.380 | 5.318 | 5.163 | **0.474** | **0.412** | **0.679** | 2.747 |
| RWKV-7 + modality (4×1) | 4.915 | 5.364 | 5.292 | 5.157 | 0.449 | 0.376 | 0.629 | 2.696 |
| RWKV-7 + slot + modality (4×4) | 4.912 | 5.366 | 5.307 | 5.155 | 0.453 | 0.394 | 0.651 | 2.759 |
| Attention (RoPE, QK-norm) | 4.978 | 5.406 | 5.259 | 5.290 | 0.428 | 0.281 | 0.483 | **1.849** |

Final train loss (mean of last 500 steps): 4.584 / 4.584 / 4.592 / 4.548.

Reading:

1. **At line level, lyric binding is not limited by the state.** Stock RWKV-7 binds *more* than attention
   (permuted gap 0.41 vs 0.28; at the start of a line 0.68 vs 0.48) and has lower code CE (4.906 vs 4.978).
   Every one of the 100 songs has a positive gap in all four runs. The line text sits 2–6 s before its audio,
   well inside what a 64×64 state holds, so the caveat in section 6 holds: the gap to close here is not recall.
2. **The state axes did not help.** Modality and slot+modality are within ±0.04 of stock on every metric, slightly
   below it on binding. Taking 4 of 12 heads out of the general budget costs about what protecting them gains.
   One seed per run, so differences of this size are not distinguishable from noise.
3. **The recall gap does show up — on copying text, not on binding audio.** The header contains the full lyrics,
   so each line's text can be copied from thousands of tokens earlier. Attention predicts that text at 1.85 nats,
   RWKV-7 at 2.70–2.76: a 0.9-nat gap, exactly the exact-recall weakness of sections 2–4. The slot heads cannot help
   there because, by design, they do not write during the header.
4. **These are teacher-forced numbers at 236M.** They say the 3B's poor lyric adherence in audio (R3.1 WER 1.05)
   is not explained by state recall at line distance. The levers are those that teach the text → sung-sound mapping:
   the lyric-cursor loss (section 7), more aligned songs, and RL on phoneme error (option 6).

Decision: do not port the slot/modality heads to the 3B. Keep the patch (off by default, `RWKV_SLOT_HEADS=0`).
Revisit the slot axis only for long-range targets where recall *is* the problem: returning choruses (melody recall
across minutes), or a whole-song format without line interleaving.
