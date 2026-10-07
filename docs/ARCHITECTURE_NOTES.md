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
