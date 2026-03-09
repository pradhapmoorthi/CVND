import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import Tuple, List

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================== Encoder: Spatial CNN =================
class EncoderCNN(nn.Module):
    """
    Spatial feature encoder using ResNet-50.
    - Call as: encoder = EncoderCNN(embed_size)
    - forward(images) -> Tensor of shape (B, T=Hf*Wf, embed_size)
    """
    def __init__(self, embed_size: int, train_backbone: bool = False):
        super().__init__()
        self.embed_size = embed_size

        m = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        )
        self.cnn = nn.Sequential(*list(m.children())[:-2])  # -> (B, 2048, Hf, Wf)
        self.adapt = nn.Conv2d(2048, embed_size, kernel_size=1)

        for p in self.cnn.parameters():
            p.requires_grad = train_backbone

        self._last_hw: Tuple[int, int] = (0, 0)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            seq : Tensor of shape (B, T=Hf*Wf, embed_size)
        """
        fmap = self.cnn(images)         # (B, 2048, Hf, Wf)
        fmap = self.adapt(fmap)         # (B, embed_size, Hf, Wf)
        B, C, Hf, Wf = fmap.shape
        self._last_hw = (Hf, Wf)
        seq = fmap.permute(0, 2, 3, 1).contiguous().view(B, Hf * Wf, C)  # (B, T, C)
        return seq

    def last_hw(self) -> Tuple[int, int]:
        """Return the last (Hf, Wf) seen in forward (useful for attention heatmaps)."""
        return self._last_hw


# ===============  Spatial Attention Module  ==============
class SpatialAttention(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(feat_dim, hidden_dim, bias=True)
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, feats: torch.Tensor, hidden: torch.Tensor):
        # feats: (B, T, C), hidden: (B, H)
        score = self.v(torch.tanh(self.W(feats) + self.U(hidden).unsqueeze(1)))  # (B, T, 1)
        alpha = torch.softmax(score, dim=1)                                      # (B, T, 1)
        ctx = (alpha * feats).sum(dim=1)                                         # (B, C)
        return ctx, alpha


# ==============  Decoder: Spatial-only LSTM  =============
class DecoderRNN(nn.Module):
    """
    Call as: decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    """
    def __init__(self,
                 embed_size: int,
                 hidden_size: int,
                 vocab_size: int,
                 dropout: float = 0.3,
                 teacher_forcing_ratio: float = 1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.attn  = SpatialAttention(feat_dim=embed_size, hidden_dim=hidden_size)
        self.lstm  = nn.LSTMCell(embed_size + embed_size, hidden_size)
        self.drop  = nn.Dropout(dropout)
        self.fc    = nn.Linear(hidden_size, vocab_size)

    def set_teacher_forcing_ratio(self, tfr: float):
        self.teacher_forcing_ratio = float(tfr)

    def forward(self, feats: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """
        feats: (B, T, embed_size) from EncoderCNN
        captions: (B, L), with BOS at position 0
        returns logits: (B, L-1, vocab_size)
        """
        B, L = captions.size()
        device = captions.device
        h = torch.zeros(B, self.hidden_size, device=device)
        c = torch.zeros(B, self.hidden_size, device=device)

        outputs = []
        inp = self.embed(captions[:, 0])  # BOS

        for t in range(1, L):
            ctx, _ = self.attn(feats, h)                         # (B, embed)
            h, c = self.lstm(torch.cat([inp, ctx], dim=1), (h, c))
            logits = self.fc(self.drop(h))                        # (B, vocab)
            outputs.append(logits.unsqueeze(1))                   # (B, 1, vocab)

            if random.random() < self.teacher_forcing_ratio:
                inp = self.embed(captions[:, t])
            else:
                inp = self.embed(logits.argmax(dim=-1))

        return torch.cat(outputs, dim=1)

    @torch.no_grad()
    def greedy_decode(self, feats, bos_id, eos_id, max_len=20):
        B = feats.size(0)
        device = feats.device
        h = torch.zeros(B, self.hidden_size, device=device)
        c = torch.zeros(B, self.hidden_size, device=device)
        x = torch.full((B,), bos_id, dtype=torch.long, device=device)
        emb = self.embed(x)
        seqs, alphas = [], []
        for _ in range(max_len):
            ctx, alpha = self.attn(feats, h)
            h, c = self.lstm(torch.cat([emb, ctx], dim=1), (h, c))
            logits = self.fc(h)
            x = logits.argmax(dim=-1)
            seqs.append(x)
            alphas.append(alpha.squeeze(-1))  # (B, T)
            emb = self.embed(x)
        out = []
        for b in range(B):
            toks = []
            for t in seqs:
                tok = int(t[b].item())
                if tok == eos_id: break
                toks.append(tok)
            out.append(toks)
        return out, alphas

    @torch.no_grad()
    def beam_search(self, feats, bos_id, eos_id, beam=3, max_len=20):
        assert feats.size(0) == 1, "Beam search supports B=1 only."
        device = feats.device
        h = torch.zeros(1, self.hidden_size, device=device)
        c = torch.zeros(1, self.hidden_size, device=device)
        beams = [([bos_id], 0.0, h, c)]
        for _ in range(max_len):
            new_beams = []
            for toks, score, h_prev, c_prev in beams:
                if toks[-1] == eos_id:
                    new_beams.append((toks, score, h_prev, c_prev))
                    continue
                x = torch.tensor([toks[-1]], device=device)
                emb = self.embed(x)
                ctx, _ = self.attn(feats, h_prev)
                h_new, c_new = self.lstm(torch.cat([emb, ctx], dim=1), (h_prev, c_prev))
                logits = self.fc(h_new)
                logprobs = F.log_softmax(logits, dim=-1)
                topk = torch.topk(logprobs, k=beam, dim=-1)
                for i in range(beam):
                    tok = int(topk.indices[0, i])
                    sc  = score + float(topk.values[0, i])
                    new_beams.append((toks + [tok], sc, h_new.clone(), c_new.clone()))
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam]
        best = beams[0][0]
        out = []
        for t in best[1:]:
            if t == eos_id: break
            out.append(t)
        return out
