import os, re, io, json, math, time, random
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

try:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    nltk_ok = True
except Exception:
    nltk_ok = False

# -------------------- Repro & Device --------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

SPECIAL_TOKENS = {'pad':'<pad>', 'bos':'<bos>', 'eos':'<eos>', 'unk':'<unk>'}

# ========================================================
# ================== Encoder: Spatial CNN =================
# ========================================================
class EncoderCNN(nn.Module):
    """
    Spatial feature encoder using ResNet-50.
    - Outputs spatial sequence of features (B, Hf*Wf, embed_size).
    - Call as: EncoderCNN(embed_size)
    """
    def __init__(self, embed_size: int):
        super().__init__()
        self.embed_size = embed_size

        # Backbone
        m = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        )
        self.cnn = nn.Sequential(*list(m.children())[:-2])  # B,2048,Hf,Wf
        self.adapt = nn.Conv2d(2048, embed_size, kernel_size=1)

        # By default, keep backbone frozen. Flip requires_grad to fine-tune if desired.
        for p in self.cnn.parameters():
            p.requires_grad = False

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Args:
            images: (B, 3, H, W)
        Returns:
            seq : (B, Hf*Wf, embed_size)
            (Hf, Wf): spatial dims of the feature map
        """
        fmap = self.cnn(images)          # (B, 2048, Hf, Wf) ~ (7,7) for 224x224
        fmap = self.adapt(fmap)          # (B, embed_size, Hf, Wf)
        B, C, Hf, Wf = fmap.shape
        seq = fmap.permute(0, 2, 3, 1).contiguous().view(B, Hf * Wf, C)  # (B, T, C)
        return seq, (Hf, Wf)

# ========================================================
# ===============  Spatial Attention Module  ==============
# ========================================================
class SpatialAttention(nn.Module):
    """
    Additive (Bahdanau-style) spatial attention over a set of features (B, T, C)
    conditioned on decoder hidden state h_t (B, H).
    """
    def __init__(self, feat_dim: int, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(feat_dim, hidden_dim, bias=True)
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, feats: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feats : (B, T, C)
            hidden: (B, H)
        Returns:
            ctx   : (B, C)   context vector
            alpha : (B, T, 1) attention weights
        """
        score = self.v(torch.tanh(self.W(feats) + self.U(hidden).unsqueeze(1)))  # (B, T, 1)
        alpha = torch.softmax(score, dim=1)                                      # (B, T, 1)
        ctx = (alpha * feats).sum(dim=1)                                         # (B, C)
        return ctx, alpha

# ========================================================
# ==============  Decoder: Spatial-only LSTM  =============
# ========================================================
class DecoderRNN(nn.Module):
    """
    Spatial-attention LSTM caption decoder (separate from the encoder).
    - Uses ONLY spatial attention
    - Supports teacher forcing ratio during training
    - Greedy decoding (returns attention maps)
    - Beam search (supports B=1)
    Call as: DecoderRNN(embed_size, hidden_size, vocab_size)
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
        self.lstm  = nn.LSTMCell(embed_size + embed_size, hidden_size)  # emb + ctx
        self.drop  = nn.Dropout(dropout)
        self.fc    = nn.Linear(hidden_size, vocab_size)

    def set_teacher_forcing_ratio(self, tfr: float):
        """Update teacher forcing ratio on the fly."""
        self.teacher_forcing_ratio = float(tfr)

    def forward(self, feats: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass with optional teacher forcing.
        Args:
            feats    : (B, T, embed_size) from EncoderCNN
            captions : (B, L) token ids; first token expected to be BOS
        Returns:
            logits   : (B, L-1, vocab_size) predictions for tokens 1..L-1
        """
        B, L = captions.size()
        device = captions.device

        h = torch.zeros(B, self.hidden_size, device=device)
        c = torch.zeros(B, self.hidden_size, device=device)

        outputs = []

        # First input is BOS
        inp = self.embed(captions[:, 0])  # (B, emb)

        for t in range(1, L):
            # Attention & LSTM step
            ctx, _ = self.attn(feats, h)                         # (B, emb)
            h, c = self.lstm(torch.cat([inp, ctx], dim=1), (h, c))
            logits = self.fc(self.drop(h))                        # (B, vocab)
            outputs.append(logits.unsqueeze(1))                   # (B, 1, vocab)

            # Teacher forcing / scheduled sampling
            use_tf = (random.random() < self.teacher_forcing_ratio)
            if use_tf:
                inp = self.embed(captions[:, t])                  # use ground-truth next token
            else:
                next_ids = logits.argmax(dim=-1)                  # use model prediction
                inp = self.embed(next_ids)

        return torch.cat(outputs, dim=1)  # (B, L-1, vocab_size)

    @torch.no_grad()
    def greedy_decode(
        self, feats: torch.Tensor, bos_id: int, eos_id: int, max_len: int = 20
    ) -> Tuple[List[List[int]], List[torch.Tensor]]:
        """
        Greedy decoding for a batch.
        Args:
            feats  : (B, T, embed_size)
            bos_id : int
            eos_id : int
            max_len: maximum generated length (excluding BOS)
        Returns:
            seqs   : List[List[int]] token ids per sample (without BOS/EOS)
            alphas : List[Tensor] list of attention weights (B, T) per step
        """
        B = feats.size(0)
        device = feats.device

        h = torch.zeros(B, self.hidden_size, device=device)
        c = torch.zeros(B, self.hidden_size, device=device)

        x = torch.full((B,), bos_id, dtype=torch.long, device=device)
        emb = self.embed(x)

        seqs = []
        alphas = []

        for _ in range(max_len):
            ctx, alpha = self.attn(feats, h)               # alpha: (B, T, 1)
            h, c = self.lstm(torch.cat([emb, ctx], dim=1), (h, c))
            logits = self.fc(h)                             # (B, vocab)
            x = logits.argmax(dim=-1)                      # (B,)

            seqs.append(x)                                  # store step-wise predictions
            alphas.append(alpha.squeeze(-1))               # (B, T)

            emb = self.embed(x)

        # Stop at EOS per sample
        out = []
        for b in range(B):
            toks = []
            for t in seqs:
                tok = int(t[b].item())
                if tok == eos_id:
                    break
                toks.append(tok)
            out.append(toks)

        return out, alphas

    @torch.no_grad()
    def beam_search(
        self, feats: torch.Tensor, bos_id: int, eos_id: int,
        beam: int = 3, max_len: int = 20
    ) -> List[int]:
        """
        Beam search decoding (supports batch size = 1).
        Args:
            feats  : (1, T, embed_size)
            bos_id : int
            eos_id : int
            beam   : beam width
            max_len: maximum generated length
        Returns:
            best sequence of token ids (without BOS/EOS)
        """
        assert feats.size(0) == 1, "Beam search supports batch size 1 only."
        device = feats.device

        h = torch.zeros(1, self.hidden_size, device=device)
        c = torch.zeros(1, self.hidden_size, device=device)

        # beams: list of tuples (tokens, logprob, h, c)
        beams = [([bos_id], 0.0, h, c)]

        for _ in range(max_len):
            new_beams = []
            for toks, score, h_prev, c_prev in beams:
                if toks[-1] == eos_id:
                    # Already ended; keep as is
                    new_beams.append((toks, score, h_prev, c_prev))
                    continue

                x = torch.tensor([toks[-1]], device=device)
                emb = self.embed(x)

                ctx, _ = self.attn(feats, h_prev)
                h_new, c_new = self.lstm(torch.cat([emb, ctx], dim=1), (h_prev, c_prev))

                logits = self.fc(h_new)                    # (1, vocab)
                logprobs = F.log_softmax(logits, dim=-1)   # (1, vocab)
                topk = torch.topk(logprobs, k=beam, dim=-1)

                for i in range(beam):
                    tok = int(topk.indices[0, i])
                    sc  = score + float(topk.values[0, i])
                    new_beams.append((toks + [tok], sc, h_new.clone(), c_new.clone()))

            # Keep top-K beams
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam]

        # Best sequence (highest logprob)
        best_tokens = beams[0][0]

        # Strip BOS and cut at EOS
        out = []
        for t in best_tokens[1:]:
            if t == eos_id:
                break
            out.append(t)
        return out
