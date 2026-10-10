#!/usr/bin/env python3
"""Architecture A/B scorer (scripts/train_arch_ab.sh): teacher-forced code-token CE and lyric binding, for RWKV-7 and
attention checkpoints alike, through the *training* model code (tools/RWKV-LM/.../src/model.py, RWKV_MIXER selects the mixer).

whole format (data/binidx/yue2_stage1_val: text EOD SOA YUE2CODEC codes EOA 0), prompts:
  matched / shuffled (another val song's caption+lyrics) / none (no text)
  + code CE by frame position (first 250 / middle / last 250) and by distance bucket from the text
sec format (data/binidx/yue2_sec_stage1_val: header EOD (SOS text SOA YUE2CODEC codes EOA)* 0), prompts as
  scripts/rwkv_eval_loss_sec.py: matched / shuffled / permuted (own stanzas rotated) / none
Binding = shuffled - matched (and permuted - matched); G3 3B whole-song had 0.10, R3.1 3B section permuted gap 0.106.
line format (data/binidx/yue2_line_stage1_val, one SOS block per lyric line): scored as sec
usage: arch_ab_eval.py --ckpt out/arch_ab/attn_whole/rwkv-final.pth --mixer attn --fmt whole [--n 100] [--slot-heads 4 --slot-m 4]
"""
import argparse, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
TT = ROOT / "tools/RWKV-LM/RWKV-v7/train_temp"
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(TT))
from yue2_layout import EOD, SOA, EOA, YUE2CODEC, SOS, N_CODES  # noqa: E402


def load_model(ckpt, mixer, n_layer, n_embd, vocab, ctx, slot_heads=0, slot_m=1):
    os.environ.update(RWKV_SLOT_HEADS=str(slot_heads), RWKV_SLOT_M=str(slot_m), RWKV_MY_TESTING="x070", RWKV_KERNEL="", RWKV_HEAD_SIZE="64", RWKV_HEAD_L2WRAP_CE_CHUNK="0",
                      RWKV_FLOAT_MODE="bf16", RWKV_JIT_ON="1", RWKV_CTXLEN=str(ctx), RWKV_MIXER=mixer)
    cwd = os.getcwd(); os.chdir(TT)                                       # kernels are loaded from cuda/ relative paths
    from src.model import RWKV
    os.chdir(cwd)
    args = SimpleNamespace(n_layer=n_layer, n_embd=n_embd, dim_att=n_embd, dim_ffn=n_embd * 4, head_size=64, vocab_size=vocab,
                           ctx_len=ctx, my_testing="x070", grad_cp=0)
    m = RWKV(args); sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    m.load_state_dict({k.replace("_forward_module.", ""): v for k, v in sd.items()}); return m.cuda().bfloat16().eval()


@torch.no_grad()
def token_ce(model, seq):
    """CE (nats) for seq[1:] given seq[:-1]; input padded to a multiple of 16 (RWKV-7 kernel chunk; causal, so padding is inert)."""
    x = list(seq[:-1]); T = len(x); pad = (-T) % 16
    idx = torch.tensor([x + [0] * pad], device="cuda")
    h = model._forward_features(idx)[0, :T]; tgt = torch.tensor(seq[1:], device="cuda"); out = torch.empty(T, device="cuda")
    for s in range(0, T, 2048):
        out[s:s + 2048] = F.cross_entropy((h[s:s + 2048] @ model.head.weight.T).float(), tgt[s:s + 2048], reduction="none")
    return out.cpu().numpy()


def eval_whole(model, ds, n, max_len, rng):
    docs = []
    for i in rng.permutation(len(ds)):
        d = [int(t) for t in np.asarray(ds[int(i)])]
        if EOD in d and SOS not in d: e = d.index(EOD); docs.append((d[:e], d[e:]))      # body = EOD SOA YUE2CODEC codes EOA 0
        if len(docs) >= n: break
    keys = ("matched", "shuffled", "none"); res = {k: [] for k in keys}; pos = {k: {"first250": [], "mid": [], "last250": []} for k in keys}
    dist = {k: {} for k in keys}; text_ce = []
    for j, (text, body) in enumerate(docs):
        other = docs[(j + 1) % len(docs)][0]
        for k, t in (("matched", text), ("shuffled", other), ("none", [])):
            full = t + body if t else body[1:]; seq = full[:max_len]; l = token_ce(model, seq)
            c0 = (len(t) + 2) if t else 1                                     # target index of the first code token
            codes = l[c0:] if len(seq) < len(full) else l[c0:-2]              # untruncated: drop the EOA and 0 targets
            res[k].append(float(codes.mean()))
            pos[k]["first250"].append(float(codes[:250].mean())); pos[k]["last250"].append(float(codes[-250:].mean()))
            pos[k]["mid"].append(float(codes[250:-250].mean()) if len(codes) > 500 else float("nan"))
            for b0 in (0, 1000, 2000, 3000, 4000, 5000):                       # frames since the song started (= distance from the lyrics)
                seg = codes[b0:b0 + 1000]
                if len(seg): dist[k].setdefault(b0, []).append(float(seg.mean()))
            if k == "matched": text_ce.append(float(l[:len(t)].mean()))
        if j % 20 == 19: print(f"  {j + 1}/{len(docs)} " + " ".join(f"{k} {np.mean(res[k]):.3f}" for k in keys), flush=True)
    d = np.array(res["shuffled"]) - np.array(res["matched"])
    return dict(n=len(docs), code_ce={k: float(np.mean(v)) for k, v in res.items()}, text_ce=float(np.mean(text_ce)),
                gap_shuffled=float(d.mean()), gap_shuffled_sd=float(d.std()), gap_frac_pos=float(np.mean(d > 0)),
                gap_none=float(np.mean(np.array(res["none"]) - np.array(res["matched"]))),
                by_pos={k: {p: float(np.nanmean(v)) for p, v in pp.items()} for k, pp in pos.items()},
                gap_by_frames={b: float(np.mean(np.array(dist["shuffled"][b][:len(dist["matched"][b])]) - np.array(dist["matched"][b][:len(dist["shuffled"][b])])))
                               for b in sorted(dist["matched"]) if b in dist["shuffled"]})


def eval_sec(model, ds, n, max_len, rng):
    from rwkv_eval_loss_sec import parse, assemble
    docs = []
    for i in rng.permutation(len(ds)):
        p = parse(np.asarray(ds[int(i)]))
        if p is not None and len(p[1]) >= 2: docs.append(p)
        if len(docs) >= n: break
    keys = ("matched", "shuffled", "permuted", "none"); res = {k: {"c": [], "c0": [], "b": []} for k in keys}; text_ce = []
    for j, (header, secs) in enumerate(docs):
        oh, osecs = docs[(j + 1) % len(docs)]
        variants = {"matched": (header, [t for t, _ in secs]),
                    "shuffled": (oh, [osecs[k % len(osecs)][0] for k in range(len(secs))]),
                    "permuted": (header, [secs[(k + 1) % len(secs)][0] for k in range(len(secs))]),
                    "none": ([], [[] for _ in secs])}
        for k, (h, texts) in variants.items():
            seq, tags = assemble(h, secs, texts); seq, tags = seq[:max_len], np.array(tags[:max_len - 1]); l = token_ce(model, seq)
            for tag in ("c", "c0", "b"):
                sel = (tags == tag) | ((tag == "c") & (tags == "c0")); res[k][tag].append(float(l[sel].mean()))
            if k == "matched": text_ce.append(float(l[tags == "t"].mean()))
        if j % 20 == 19: print(f"  {j + 1}/{len(docs)} " + " ".join(f"{k} {np.mean(res[k]['c']):.3f}" for k in keys), flush=True)
    c = {k: np.array(v["c"]) for k, v in res.items()}
    return dict(n=len(docs), code_ce={k: float(v.mean()) for k, v in c.items()},
                code_ce_first50={k: float(np.mean(v["c0"])) for k, v in res.items()},
                boundary_ce={k: float(np.mean(v["b"])) for k, v in res.items()}, text_ce=float(np.mean(text_ce)),
                gap_shuffled=float((c["shuffled"] - c["matched"]).mean()), gap_permuted=float((c["permuted"] - c["matched"]).mean()),
                gap_permuted_frac_pos=float(np.mean(c["permuted"] > c["matched"])), gap_none=float((c["none"] - c["matched"]).mean()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--mixer", required=True, choices=["rwkv", "attn"])
    ap.add_argument("--fmt", required=True, choices=["whole", "sec", "line"])
    ap.add_argument("--slot-heads", type=int, default=0); ap.add_argument("--slot-m", type=int, default=1); ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--n-layer", type=int, default=12); ap.add_argument("--n-embd", type=int, default=768)
    ap.add_argument("--max-len", type=int, default=8192); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="")
    a = ap.parse_args()
    from src.binidx import MMapIndexedDataset
    model = load_model(a.ckpt, a.mixer, a.n_layer, a.n_embd, 98816, a.max_len, a.slot_heads, a.slot_m)
    data = ROOT / {"whole": "data/binidx/yue2_stage1_val", "sec": "data/binidx/yue2_sec_stage1_val", "line": "data/binidx/yue2_line_stage1_val"}[a.fmt]
    ds = MMapIndexedDataset(str(data)); rng = np.random.default_rng(a.seed); t0 = time.time()
    print(f"{a.mixer} {a.fmt} {a.ckpt} on {data.name}, n={a.n}", flush=True)
    r = (eval_whole if a.fmt == "whole" else eval_sec)(model, ds, a.n, a.max_len, rng)
    r.update(ckpt=str(a.ckpt), mixer=a.mixer, fmt=a.fmt, slot_heads=a.slot_heads, slot_m=a.slot_m, seconds=round(time.time() - t0))
    print("RESULT " + json.dumps(r, indent=1), flush=True)
    out = Path(a.out) if a.out else Path(a.ckpt).with_suffix(f".eval_{a.fmt}.json"); out.write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
