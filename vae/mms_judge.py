"""MMS_FA (torchaudio wav2vec2, multilingual CTC over uroman letters a-z ') as an audio-domain lyric judge.

Our CTC alphabet (vae/model.LETTERS = " abcdefghijklmnopqrstuvwxyz'") maps onto the MMS_FA labels; the space has no MMS label
(MMS_FA scores word pieces without separators), so spaces are dropped from targets. Input: 24 kHz audio [B,2,S] (any grad),
resampled to 16 kHz mono inside; output log-probs [B, T, V] (frozen weights, differentiable w.r.t. the audio).
"""
import torch
import torch.nn.functional as F
import torchaudio
from .model import LETTERS


class MMSJudge(torch.nn.Module):
    def __init__(s):
        super().__init__()
        from torchaudio.pipelines import MMS_FA
        s.model = MMS_FA.get_model(with_star=False).eval().requires_grad_(False)
        labels = MMS_FA.get_labels(star=None); s.blank = 0
        d = {c: i for i, c in enumerate(labels)}
        m = torch.full((len(LETTERS) + 1,), -1, dtype=torch.long)                  # our id (1-based) -> MMS id, -1 = drop
        for i, c in enumerate(LETTERS):
            if c in d: m[i + 1] = d[c]
        s.register_buffer("map", m); s.resample = torchaudio.transforms.Resample(24000, 16000)

    def targets(s, tg, lens):
        """our flat ids + lengths -> MMS flat ids + lengths (spaces dropped)"""
        out, nl, o = [], [], 0
        for n in lens.tolist():
            t = s.map[tg[o:o + n].to(s.map.device)]; t = t[t >= 0]; out.append(t); nl.append(len(t)); o += n
        return torch.cat(out) if out else torch.zeros(0, dtype=torch.long, device=s.map.device), torch.tensor(nl)

    def forward(s, wav24):
        x = s.resample(wav24.float().mean(1))
        x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-5)       # wav2vec2 expects normalised input
        em, _ = s.model(x)
        return F.log_softmax(em.float(), -1)

    def ctc(s, lp, tg, lens, ok, cf_sec):
        """CTC of the selected samples; cf_sec = seconds of audio the targets cover (from the 50 Hz ctc_frames / 50)."""
        idx = ok.nonzero().flatten()
        if len(idx) == 0: return None
        mt, ml = s.targets(tg, lens); offs = torch.cat([torch.zeros(1, dtype=torch.long), ml.cumsum(0)])
        tgt = torch.cat([mt[offs[i]:offs[i + 1]] for i in idx.tolist()])
        fps = lp.shape[1] / 8.0                                                       # frames per second of the 8 s crop
        il = torch.clamp((cf_sec[idx] * fps).long(), max=lp.shape[1]).to(lp.device)
        return F.ctc_loss(lp[idx].transpose(0, 1), tgt, il, ml[idx].to(lp.device), blank=s.blank, zero_infinity=True)
