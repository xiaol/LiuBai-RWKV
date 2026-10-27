#!/usr/bin/env python3
"""Architecture figure for the paper: LiuBai-RWKV = semantic-first RWKV-VAE (v4) + block-flow cursor LM.
Numbers follow out/vae/v4/config.json, out/clm/cursor_v4b/config.json and vae/{model,cursor_lm,rwkv}.py.
usage: python liubai_arch.py  ->  liubai_arch.pdf (vector, for LaTeX) + liubai_arch.png (preview)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from pathlib import Path

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "font.size": 8.5})
W, H = 1400, 925
fig = plt.figure(figsize=(W / 100, H / 100)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")

STY = dict(rwkv=("#dbe8ff", "#2f5fb3", "-", ""), mlp=("#e3f3e8", "#3c8c5a", "-", ""), frz=("#ececec", "#777777", "-", ""),
           sem=("#ffe2bf", "#c46a00", "-", ""), aco=("#d3e9ff", "#2f5fb3", "-", ""), io=("#ffffff", "#333333", "-", ""),
           cfg=("#f4ecff", "#7048b0", "-", ""), train=("#ffffff", "#888888", "--", ""), loss=("#fff6f5", "#b03a2e", "--", ""))


def box(x, y, w, h, st, title=None, lines=(), small=(), r=6):
    fc, ec, ls, hatch = STY[st]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, ls=ls, lw=1.3,
                                hatch=hatch, zorder=2))
    rows = ([(title, dict(fontweight="bold", fontsize=8.8, color="#b03a2e" if st == "loss" else "#222"))] if title else []) \
        + [(t, dict(fontsize=7.9, color="#222")) for t in lines] + [(t, dict(fontsize=7.3, color="#555")) for t in small]
    step = 15; y0 = y + h / 2 - step * (len(rows) - 1) / 2
    for i, (t, kw) in enumerate(rows):
        ax.text(x + w / 2, y0 + i * step, t, ha="center", va="center", zorder=3, **kw)


def arrow(pts, dashed=False, grey=False, head=True):
    col = "#888888" if (dashed or grey) else "#333333"
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-1],), ys, color=col, lw=1.3, ls="--" if dashed else "-", zorder=1, solid_capstyle="butt")
    if head:
        ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=4,
                    arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.28", color=col, lw=1.3, shrinkA=0, shrinkB=0))


def note(x, y, t, **kw): ax.text(x, y, t, fontsize=kw.pop("fs", 7.8), color=kw.pop("c", "#555"), va="center", zorder=3, **kw)


ax.text(20, 24, "LiuBai-RWKV: semantic-first RWKV-VAE (v4) and block-flow cursor LM", fontsize=12.5, fontweight="bold", va="center")

# ============ (a) VAE ============
ax.text(20, 58, "(a) RWKV-VAE v4: partitioned latent, dims 0–31 semantic, dims 32–63 acoustic", fontsize=11, fontweight="bold", va="center")
box(20, 175, 110, 70, "io", "Audio $x$", ["24 kHz stereo"])
arrow([(130, 200), (148, 200), (148, 125), (165, 125)]); arrow([(130, 220), (148, 220), (148, 295), (165, 295)])

box(165, 90, 190, 70, "mlp", "Conv encoder", ["WN-Conv1d + Snake, 6 stages", "strides 2·2·4·4·5·6 = 1920"])
arrow([(355, 125), (385, 125)])
box(385, 90, 160, 70, "rwkv", "Bi-RWKV-7 × 4", ["$d$ = 512", "forward + reverse WKV"])
arrow([(545, 125), (575, 125)])
box(575, 90, 120, 70, "mlp", "Linear head", [r"$\mu_a,\ \log\sigma_a^2$", "64 dims"])

box(165, 255, 200, 80, "frz", "MERT-v2-FullSong", ["frozen Conformer, layers 0–12", "3 s windows, ≤ 0.64 s lookahead",
                                                   "instance-norm, 25 Hz"])
arrow([(365, 295), (390, 295)])
box(390, 265, 155, 60, "mlp", "Semantic MLP", [r"$2{\times}1024 \rightarrow 512 \rightarrow 512$  ($h_s$)"])
arrow([(545, 295), (575, 295)])
box(575, 260, 120, 70, "mlp", "Linear head", [r"$\mu_s,\ \log\sigma_s^2$", "32 dims"])

arrow([(650, 160), (650, 215), (725, 215)]); note(656, 190, r"$\mu_a[0{:}32]$")
arrow([(695, 295), (815, 295), (815, 265)]); note(702, 285, r"$\mu_s,\ g(h_s)$")
box(722, 175, 186, 90, "mlp", "Gated correction",
    [r"$\mu[0{:}32] = \mu_s$", r"$+\,0.5\,\tanh(\mu_a[0{:}32])\,\sigma(g(h_s))$"])
arrow([(695, 110), (930, 110)]); note(770, 101, r"$\mu_a[32{:}64],\ \log\sigma_a^2[32{:}64]$")
arrow([(905, 240), (918, 240), (918, 272), (930, 272)])

ax.text(975, 80, "$z$", ha="center", va="center", fontsize=11, fontweight="bold")
box(930, 90, 92, 115, "aco", "dims 32–63", ["acoustic"], r=3)
box(930, 215, 92, 115, "sem", "dims 0–31", ["semantic"], r=3)
note(975, 345, r"$T \times 64$ at 12.5 Hz", ha="center")

arrow([(1022, 210), (1060, 210)])
box(1060, 155, 170, 110, "rwkv", "Decoder", ["Linear 64 → 512", "Bi-RWKV-7 × 4", "transposed conv, ×1920"])
arrow([(1230, 210), (1260, 210)])
box(1260, 175, 115, 70, "io", r"Audio $\hat{x}$", ["24 kHz stereo"])
box(1170, 290, 210, 60, "loss", "Audio losses", ["mel ×15 + STFT + GAN + FM ×2", r"MMS-CTC on $\hat{x}$ ×0.3"])
box(860, 360, 310, 62, "loss", "Latent losses", [r"$\cos(\mathrm{head}(z[0{:}32]),\ \mathrm{MERT\ L12})$ ×2",
                                                 r"chroma$(z[32{:}64])$ ×1, lyric CTC$(z)$ ×1, KL ×0.01"])
note(165, 385, "MERT runs only on the encoder side (training data, generation prompts).")
note(165, 401, "Generation produces z directly and decodes it with the RWKV decoder: no MERT at decode time.")
ax.plot([20, 1380], [455, 455], color="#cccccc", lw=1)

# ============ (b) cursor LM ============
ax.text(20, 488, "(b) Cursor LM: causal RWKV-7 over blocks of z, per-block flow head, monotonic lyric cursor",
        fontsize=11, fontweight="bold", va="center")
box(20, 520, 230, 56, "io", "Lyrics (LRC lines)", ["uroman → 28 letters (a–z, ', space)"])
arrow([(135, 576), (135, 600)], dashed=True)
box(20, 600, 230, 60, "train", "MMS forced alignment", [r"letter onsets $\rightarrow c_i,\ a_i$"], ["(training targets only)"])
arrow([(135, 660), (135, 690)], dashed=True)
arrow([(250, 548), (268, 548), (268, 700), (250, 700)])
box(20, 690, 230, 70, "mlp", r"Cursor window $W_i$", [r"letters $c_i-4\ \ldots\ c_i+19$", "24 × (letter emb. + rel. pos.)"])
box(20, 790, 230, 56, "io", "Prompt: 4 s real audio", ["encoder (a) → context blocks"])

ax.plot([250, 300, 300, 780], [745, 745, 775, 775], color="#333333", lw=1.3, zorder=1)
for xx in (420, 600, 780): arrow([(xx, 775), (xx, 750)])
note(470, 788, r"$W_{i-1},\ W_i,\ W_{i+1}$")
arrow([(250, 818), (400, 818), (400, 750)]); note(300, 810, "context")

for xx, t, lab in ((345, r"Lin($z_{i-2}$) + Emb($W_{i-1}$)", "token i − 1"), (525, r"Lin($z_{i-1}$) + Emb($W_i$)", "token i"),
                   (705, r"Lin($z_i$) + Emb($W_{i+1}$)", "token i + 1")):
    box(xx, 700, 150, 50, "mlp", None, [t], [lab]); arrow([(xx + 75, 700), (xx + 75, 670)])
box(330, 610, 540, 60, "rwkv", "Causal RWKV-7 × 12, $d$ = 1024", ["fixed-size recurrent state per layer (no KV cache)"])
arrow([(420, 610), (420, 594)], grey=True); note(426, 600, r"$h_{i-1}$")
arrow([(780, 610), (780, 594)], grey=True); note(786, 600, r"$h_{i+1}$")
arrow([(600, 610), (600, 585)]); note(606, 600, r"$h_i$")
box(525, 540, 150, 45, "mlp", "Advance head", [r"$a_i \in \{0,\ldots,23\}$ letters"])
arrow([(525, 562), (290, 562), (290, 712), (250, 712)]); note(330, 553, r"$c_{i+1} = c_i + a_i$")
arrow([(675, 562), (900, 562)]); note(730, 553, r"$h_i + \mathrm{emb}(a_i)$")
box(900, 520, 220, 90, "mlp", "Flow head", ["adaLN-MLP × 6, $d$ = 1024", r"condition $h_i + \mathrm{emb}(a_i)$",
                                             "rectified flow, 24 Euler steps"])
arrow([(1120, 565), (1160, 565)])
box(1160, 530, 115, 70, "sem", r"$z_i$", ["4 frames × 64", "= 0.32 s"])
arrow([(1217, 600), (1217, 625), (885, 625), (885, 725), (855, 725)]); note(1095, 640, r"$z_i$ is the input of token i + 1")
arrow([(1010, 610), (1010, 680)], dashed=True)
box(900, 680, 370, 90, "cfg", "Lyric classifier-free guidance (inference)",
    [r"$v = v_\varnothing + w\,(v_\mathrm{lyr} - v_\varnothing),\quad w = 3$–$5$", r"$\varnothing$ = null window + null advance"],
    ["(lyrics dropped with p = 0.1 in training)"])
box(900, 795, 230, 62, "loss", "Training loss", ["flow MSE (dims 0–31 weighted ×3)", r"+ 0.2 · CE($a_i$)"])
arrow([(1275, 565), (1330, 565), (1330, 780), (1217, 780), (1217, 795)])
box(1150, 795, 135, 62, "rwkv", "VAE decoder (a)", ["Bi-RWKV-7 + conv"], ["no MERT"])
arrow([(1285, 826), (1300, 826)])
box(1300, 795, 90, 62, "io", r"Audio $\hat{x}$", ["24 kHz"])

# legend
lx = 20
for st, t in (("rwkv", "trained, RWKV-7"), ("mlp", "trained, conv / MLP"), ("frz", "frozen, external"), ("sem", "semantic latent"),
              ("aco", "acoustic latent"), ("train", "training only"), ("cfg", "inference only"), ("loss", "loss")):
    fc, ec, ls, hatch = STY[st]
    ax.add_patch(Rectangle((lx, 884), 16, 13, fc=fc, ec=ec, ls=ls, lw=1.2, hatch=hatch))
    ax.text(lx + 22, 891, t, va="center", fontsize=8.5); lx += 40 + 7.0 * len(t)

out = Path(__file__).with_suffix("")
fig.savefig(f"{out}.pdf"); fig.savefig(f"{out}.png", dpi=150); print("wrote", f"{out}.pdf", f"{out}.png")
