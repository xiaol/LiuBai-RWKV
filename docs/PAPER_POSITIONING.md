# Paper positioning: LiuBai-RWKV vs YuE2 (2026-10-27)

Figure: `docs/figures/liubai_arch.pdf` (source `liubai_arch.py`). YuE2 facts are checked against the arXiv 2609.33757v1
TeX source (`curl -L https://arxiv.org/e-print/2609.33757v1`, `technical_report.tex`) and the local full-text copy
`/tmp/lit/2609.33757v1.txt`.

## 1. Side by side

| | YuE2 (m-a-p, 2026-09) | LiuBai-RWKV (now) |
|---|---|---|
| Generator | 3.58B AR–NAR Mixture-of-Transformers, 28 layers, shared attention | 208M causal RWKV-7 (12 × 1024) + per-block flow head |
| Sequence memory | attention, context 24,576 positions (≈ 8 min cap) | fixed-size recurrent state, no positional limit (stateful decoding not implemented yet) |
| Audio representation | 25 Hz MERT2 semantic tokens (32,768) → NAR flow → 25 Hz × 64 VAE latent, 48 kHz | no tokens: block flow directly on a 12.5 Hz × 64 RWKV-VAE latent, 24 kHz |
| Lyric conditioning | whole lyric text as a prefix; the model must recall it through attention | monotonic letter cursor: a 24-letter window moves along the lyrics, an advance head predicts its speed |
| Lyric supervision of the latent | MERT2 tokenizer trained with lyric CTC (unreleased) | semantic-first VAE: dims 0–31 from frozen MERT-v2 L12 (bounded lookahead) + lyric CTC on z |
| Symbolic planning | ABC melody + chord score (SheetSage2 labels), improves musicality (49 vs 35 % prefs) | none |
| Data / compute | 346k h; 106k updates × batch 256 on 64 H800 | 21k h (333k NetEase songs with timed LRC); 1 × A100 per run |
| Task | text + lyrics → full song (create, edit, cover) | 4 s audio prompt + lyrics → 20 s continuation |
| Lyric metric | PER 0.0844 on WildSongBench (Qwen3-ASR phonemes; best of 2 candidates × min of 4 ASR runs) | Whisper letter CER on 96 held-out crops: 0.358 (v4b, CFG 5); VAE ceiling 0.61, original 0.55 |

The two lyric numbers are not comparable (different task, length, ASR, unit and selection protocol). We do not beat YuE2
on anything measured so far. The comparison is architectural.

## 2. Evidence we have (96 held-out crops, causal prompt, paired with 95 % bootstrap CIs)

| Model | no CFG | CFG 3 | CFG 5 |
|---|---|---|---|
| v2 latent, 25k steps | 0.82 | 0.507 | 0.399 |
| v4 latent (leaky training), 40k | 0.90 | 0.382 | 0.302 |
| **v4b** (bounded lookahead, cleaned lyrics, sem-w 3), 20k | 0.89 | 0.436 | 0.358 |
| no lyrics (v2) | 0.94 | | |

- Swapped lyrics are sung as given: v4b CFG 5 scores 0.407 against the given text and 1.27 against the crop's own.
- Learned timing beats forced-alignment timing at every CFG scale (v2: 0.507 vs 0.545 at CFG 3).
- Prefix conditioning (DiffRhythm-2 style, whole lyrics before the audio) fails in the same RNN: own 1.065.
- CFG improves quality metrics too (MERT cos to the original 0.73 → 0.79–0.80, LTAS 6.2 → 4.6 dB); CFG 5 starts to
  over-compress (clipping 0.26 %, 2 × the VAE).
- v4 latent leaks future audio into earlier frames (0.78 → 0.18 std from 1 to 100 frames before a cut) through
  bidirectional MERT + whole-crop normalisation; bounded-lookahead encoding removes it (0 beyond the bound).

## 3. Candidate novelty (literature check 2026-10-22 found no prior RWKV / linear-attention continuous song LM and no lyric pointer for songs; re-check before submission)

1. **Monotonic lyric cursor for recurrent song generation.** Lyrics never have to be recalled from the state: a moving
   letter window plus a learned advance head binds text to audio in O(1) memory. The model learns its own timing (no
   timestamps at inference, unlike DiffRhythm 1), and the prefix baseline in the same RNN fails.
2. **Tokenizer-free recurrent song LM.** Causal RWKV-7 over continuous VAE latent blocks with a per-block flow head:
   no codec tokens, no separate renderer, no KV cache. Enables unbounded length, streaming and state snapshots for
   section edits (claims need the stateful decoder, see §4).
3. **Latent readability only pays off under guidance.** v2 and v4 latents tie without CFG; with CFG the semantic-first
   latent wins clearly. Explained by CFG sharpening the model's implicit p(lyrics | z), which is better on a latent
   whose lyric content is linearly decodable. Ties into MSVAE, VA-VAE / REPA (reconstruction vs generability).
4. **Causality pitfall of SSL-aligned latents in AR generators.** Bidirectional SSL features (MERT) inside a VAE latent
   leak future content into teacher-forced context. Measured, and fixed with bounded-lookahead encoding. Relevant to any
   AR model trained on MERT/HuBERT/WavLM-aligned latents.
5. Single romanised letter vocabulary (uroman) across Chinese, Korean, Japanese and English lyrics.

## 4. What we need before a paper

Must-have
- **Text + lyrics → full song.** Caption/style conditioning (Qwen3-Omni captions exist for the corpus), no audio
  prompt, multi-minute generation. Today's results are 20 s continuations.
- **Stateful RWKV decoding** (carry WKV and token-shift states; parity test against full-prefix inference). The current
  eval replays the whole history every block, so the constant-memory claim is not yet demonstrated.
- **Standard benchmark and metrics.** WildSongBench (192 prompts, public) with YuE2's protocol: PER (Qwen3-ASR, pinyin with tones / English phonemes), SongEval,
  SongBench, AudioBox PQ, MuQ-MuLan; report single-sample and best-of-2 separately. Baselines: YuE2, DiffRhythm 2,
  ACE-Step 1.5, HeartMuLa (public weights).
- **Ablations with matched steps and ≥ 3 seeds:** v2 vs v4b latent, cursor vs prefix, sampled vs oracle timing, CFG
  scale, sem-dim weight, leaky vs bounded-lookahead encoding, RWKV-7 vs Transformer of the same size.
- **Human listening test** (A/B or MOS on lyric clarity, musicality, quality).
- **48 kHz VAE** (24 kHz drops everything above 12 kHz; LTAS gap is mostly that).

Should-have
- **Replace MERT-v2-FullSong** (YuE2-family, CC BY-NC) with our own causal RWKV semantic encoder, distilled from MERT
  L12 or trained with lyric CTC, so the whole stack is RWKV and causal.
- **Long-form demo** beyond YuE2's 8 min cap at constant memory, and a streaming real-time factor.
- **Data statement.** The NetEase corpus is copyrighted commercial music: training/release terms must be settled
  before publishing weights; the paper needs a data card.

Optional
- Symbolic planning: YuE2 shows an ABC melody/chord plan improves musicality. SheetSage2 could label our corpus as an
  offline teacher (it is a YuE2-family model, so this conflicts with the "no YuE2 components" goal unless used only
  for labels).
