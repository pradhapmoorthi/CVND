import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import Tuple, List
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1] # remove the last FC layer
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images) # Output might be [batch_size, 2048, 1, 1]
        features = features.view(features.size(0), -1) # Flatten the output to [batch_size, 2048]
        features = self.embed(features)
        return features

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
    Spatial-attention LSTM caption decoder (separate from the encoder).
    Call as: DecoderRNN(embed_size, hidden_size, vocab_size)
    Returns L logits to match captions.shape[1].
    """
    def __init__(
        self,
        embed_size: int,
        hidden_size: int,
        vocab_size: int,
        dropout: float = 0.3,
        teacher_forcing_ratio: float = 1.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.attn  = SpatialAttention(feat_dim=embed_size, hidden_dim=hidden_size)
        # input = word emb (embed_size) + context (embed_size)
        self.lstm  = nn.LSTMCell(embed_size + embed_size, hidden_size)
        self.drop  = nn.Dropout(dropout)
        self.fc    = nn.Linear(hidden_size, vocab_size)

    def set_teacher_forcing_ratio(self, tfr: float):
        self.teacher_forcing_ratio = float(tfr)

    def forward(self, feats: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """
        feats    : (B, T, embed_size) from EncoderCNN (B1)
        captions : (B, L) with BOS at index 0
        returns  : (B, L, vocab_size)  <-- matches your assert
        """
        B, L = captions.size()
        device = captions.device

        h = torch.zeros(B, self.hidden_size, device=device)
        c = torch.zeros(B, self.hidden_size, device=device)

        outputs = []

        # Step 0 input is BOS
        inp = self.embed(captions[:, 0])  # (B, embed_size)

        for t in range(L):  # produce L logits
            # Attend + step
            ctx, _ = self.attn(feats, h)                         # (B, embed_size)
            h, c  = self.lstm(torch.cat([inp, ctx], dim=1), (h, c))
            logits = self.fc(self.drop(h))                        # (B, vocab)
            outputs.append(logits.unsqueeze(1))                   # (B, 1, vocab)

            # Prepare input for next time step
            if t + 1 < L:  # only fetch next GT token if it exists
                if random.random() < self.teacher_forcing_ratio:
                    inp = self.embed(captions[:, t + 1])          # ground truth next token
                else:
                    inp = self.embed(logits.argmax(dim=-1))       # model's prediction

        return torch.cat(outputs, dim=1)  # (B, L, vocab_size)

    @torch.no_grad()
    def greedy_decode(self, feats: torch.Tensor, bos_id: int, eos_id: int, max_len: int = 20):
        B = feats.size(0)
        device = feats.device

        h = torch.zeros(B, self.hidden_size, device=device)
        c = torch.zeros(B, self.hidden_size, device=device)

        x = torch.full((B,), bos_id, dtype=torch.long, device=device)
        emb = self.embed(x)

        seqs = []
        for _ in range(max_len):
            ctx, alpha = self.attn(feats, h)
            h, c = self.lstm(torch.cat([emb, ctx], dim=1), (h, c))
            logits = self.fc(h)
            x = logits.argmax(dim=-1)
            seqs.append(x)
            emb = self.embed(x)

        out = []
        for b in range(B):
            toks = []
            for t in seqs:
                tok = int(t[b].item())
                if tok == eos_id: break
                toks.append(tok)
            out.append(toks)

        #print(f"DEBUG (greedy_decode): B={B}, len(out)={len(out)}") # ADDED DEBUG PRINT
        return out

    @torch.no_grad()
    def beam_search(self, feats: torch.Tensor, bos_id: int, eos_id: int, beam: int = 3, max_len: int = 20):
        assert feats.size(0) == 1, "Beam search supports batch size 1 only."
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
