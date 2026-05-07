"""
model.py — Image Captioning with Bahdanau (Soft) Attention
===========================================================
Architecture: Show, Attend and Tell (Xu et al., 2015)
  • EncoderCNN  — ResNet-50 spatial feature extractor (14×14 grid = 196 locations)
  • BahdanauAttention — soft-attention over the spatial grid
  • DecoderRNN  — LSTMCell conditioned on attended context at every step

Interface preserved from Udacity CVND project:
  encoder = EncoderCNN(embed_size)
  decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
  features = encoder(images)                  # training forward
  outputs  = decoder(features, captions)      # teacher-forcing forward
  output   = decoder.sample(features)         # greedy inference → list[int]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ─────────────────────────────────────────────────────────────────────────────
#  Attention module
# ─────────────────────────────────────────────────────────────────────────────

class BahdanauAttention(nn.Module):
    """
    Additive (Bahdanau) soft-attention over encoder spatial features.

    At each decoding step t the attention energy for pixel i is:
        e_ti  = v^T  tanh( W_enc * h_enc_i  +  W_dec * h_dec_t )
    Attention weights are α = softmax(e).
    Context vector is c = Σ_i  α_i * h_enc_i.
    """

    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        super().__init__()
        self.encoder_att  = nn.Linear(encoder_dim,  attention_dim, bias=False)
        self.decoder_att  = nn.Linear(decoder_dim,  attention_dim, bias=False)
        self.full_att     = nn.Linear(attention_dim, 1,            bias=False)
        self.bias         = nn.Parameter(torch.zeros(attention_dim))
        self.relu         = nn.ReLU(inplace=True)
        self.softmax      = nn.Softmax(dim=1)

    def forward(
        self,
        encoder_out:    torch.Tensor,   # (B, num_pixels, encoder_dim)
        decoder_hidden: torch.Tensor,   # (B, decoder_dim)
    ):
        """
        Returns:
            context : (B, encoder_dim)   weighted sum of encoder features
            alpha   : (B, num_pixels)    attention weights (for visualisation / regularisation)
        """
        att_enc = self.encoder_att(encoder_out)                       # (B, P, att_dim)
        att_dec = self.decoder_att(decoder_hidden).unsqueeze(1)       # (B, 1, att_dim)
        energy  = self.full_att(
            self.relu(att_enc + att_dec + self.bias)
        ).squeeze(2)                                                   # (B, P)
        alpha   = self.softmax(energy)                                 # (B, P)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)        # (B, encoder_dim)
        return context, alpha


# ─────────────────────────────────────────────────────────────────────────────
#  Encoder
# ─────────────────────────────────────────────────────────────────────────────

class EncoderCNN(nn.Module):
    """
    ResNet-50 spatial encoder.

    Replaces global average-pool + FC with:
      • AdaptiveAvgPool2d(14, 14) → (B, 2048, 14, 14) = 196 grid locations
      • Linear projection 2048 → embed_size
      • BatchNorm over the 196 locations

    The top ResNet blocks can be fine-tuned after an initial warm-up epoch
    by calling encoder.fine_tune(True).
    """

    def __init__(self, embed_size: int):
        super().__init__()
        self.embed_size = embed_size

        # Support both old (pretrained=True) and new (weights=...) torchvision API
        try:
            from torchvision.models import ResNet50_Weights
            resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        except ImportError:
            resnet = models.resnet50(pretrained=True)   # torchvision < 0.13

        # Freeze the entire backbone initially
        for p in resnet.parameters():
            p.requires_grad_(False)

        # Drop avgpool + fc; keep everything up to layer4
        # For 224×224 input ResNet outputs (B, 2048, 7, 7) after layer4
        self.backbone     = nn.Sequential(*list(resnet.children())[:-2])

        # Upsample to a 14×14 grid → 196 attention locations (richer attention)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))

        # Project to embed_size
        self.proj = nn.Sequential(
            nn.Linear(2048, embed_size),
            nn.BatchNorm1d(196, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        # Learnable spatial positional bias (optional but helpful)
        self.pos_embed = nn.Parameter(torch.zeros(1, 196, embed_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    # ------------------------------------------------------------------
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images : (B, 3, 224, 224)
        Returns:
            features : (B, 196, embed_size)   spatial encoder features
        """
        feat = self.backbone(images)              # (B, 2048,  7,  7)
        feat = self.adaptive_pool(feat)           # (B, 2048, 14, 14)
        B, C, H, W = feat.shape
        feat = feat.permute(0, 2, 3, 1)           # (B, 14, 14, 2048)
        feat = feat.reshape(B, H * W, C)          # (B, 196,  2048)
        feat = self.proj(feat)                    # (B, 196,  embed_size)
        feat = feat + self.pos_embed              # add positional bias
        return feat

    # ------------------------------------------------------------------
    def fine_tune(self, fine_tune: bool = True):
        """
        Allow gradient updates through the upper ResNet blocks (conv3–conv5).
        Call encoder.fine_tune(True) after the first warm-up epoch.
        """
        for p in self.backbone.parameters():
            p.requires_grad = False
        # Unfreeze layer2, layer3, layer4  (children index 5, 6, 7)
        for child in list(self.backbone.children())[5:]:
            for p in child.parameters():
                p.requires_grad = fine_tune


# ─────────────────────────────────────────────────────────────────────────────
#  Decoder
# ─────────────────────────────────────────────────────────────────────────────

class DecoderRNN(nn.Module):
    """
    LSTM decoder with Bahdanau soft-attention.

    At each decoding step t:
      1. Compute attention context c_t from encoder features and h_{t-1}
      2. Gating scalar β_t modulates the context (doubly-stochastic attention)
      3. LSTMCell input = [word_embed_t || β_t * c_t]
      4. Output logits = FC( dropout( h_t ) )

    Hyperparameters embedded here (can be overridden via class attributes):
        ATTENTION_DIM = 512   attention hidden size
        DROPOUT_P     = 0.5   dropout probability
    """

    ATTENTION_DIM: int   = 512
    DROPOUT_P:     float = 0.5

    def __init__(
        self,
        embed_size:  int,
        hidden_size: int,
        vocab_size:  int,
        num_layers:  int = 1,       # kept for API compatibility; always uses 1 LSTMCell
    ):
        super().__init__()
        self.embed_size  = embed_size
        self.hidden_size = hidden_size
        self.vocab_size  = vocab_size

        # ── Attention ──────────────────────────────────────────────────────
        self.attention = BahdanauAttention(
            encoder_dim  = embed_size,
            decoder_dim  = hidden_size,
            attention_dim= self.ATTENTION_DIM,
        )

        # ── Word embedding ────────────────────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

        # ── LSTM cell ─────────────────────────────────────────────────────
        # Input: concatenated [word_embed (E) || context (E)] → 2E
        self.lstm_cell = nn.LSTMCell(embed_size + embed_size, hidden_size)

        # ── LSTM state init from mean encoder features ────────────────────
        self.init_h = nn.Linear(embed_size, hidden_size)
        self.init_c = nn.Linear(embed_size, hidden_size)

        # ── Attention gating (doubly-stochastic regularisation) ───────────
        self.f_beta = nn.Linear(hidden_size, embed_size)

        # ── Output projection ─────────────────────────────────────────────
        self.dropout = nn.Dropout(self.DROPOUT_P)
        self.fc      = nn.Linear(hidden_size, vocab_size)
        nn.init.uniform_(self.fc.weight, -0.1, 0.1)
        nn.init.zeros_(self.fc.bias)

    # ------------------------------------------------------------------
    def _init_hidden(self, encoder_out: torch.Tensor):
        """
        Initialise LSTM hidden state and cell from mean encoder features.
        Args:
            encoder_out : (B, P, embed_size)
        Returns:
            h, c : each (B, hidden_size)
        """
        mean = encoder_out.mean(dim=1)           # (B, embed_size)
        h    = torch.tanh(self.init_h(mean))
        c    = torch.tanh(self.init_c(mean))
        return h, c

    # ------------------------------------------------------------------
    def forward(
        self,
        encoder_out: torch.Tensor,   # (B, P, embed_size)  from EncoderCNN
        captions:    torch.Tensor,   # (B, T)  token ids incl. <start> and <end>
    ) -> torch.Tensor:
        """
        Teacher-forcing forward pass.

        Returns:
            predictions : (B, T-1, vocab_size)
                          At step t the model predicts caption token t+1
                          (so predictions[:, 0, :] targets captions[:, 1]).
        """
        B         = encoder_out.size(0)
        T         = captions.size(1)
        decode_len = T - 1           # predict T-1 tokens (exclude <start> from targets)

        embeds = self.dropout(self.embedding(captions))   # (B, T, E)
        h, c   = self._init_hidden(encoder_out)           # (B, H)

        predictions = encoder_out.new_zeros(B, decode_len, self.vocab_size)

        for t in range(decode_len):
            # ── Attention step ─────────────────────────────────────────
            context, alpha = self.attention(encoder_out, h)   # (B, E)

            # ── Gating ────────────────────────────────────────────────
            gate    = torch.sigmoid(self.f_beta(h))           # (B, E)
            context = gate * context                           # (B, E)

            # ── LSTMCell step ─────────────────────────────────────────
            word_embed  = embeds[:, t, :]                      # (B, E)  ← token t (starts at <start>)
            lstm_input  = torch.cat([word_embed, context], 1)  # (B, 2E)
            h, c        = self.lstm_cell(lstm_input, (h, c))

            # ── Predict token at position t+1 ─────────────────────────
            predictions[:, t, :] = self.fc(self.dropout(h))   # (B, vocab_size)

        return predictions   # (B, T-1, vocab_size)

    # ------------------------------------------------------------------
    def sample(
        self,
        encoder_out: torch.Tensor,   # (1, P, E)  or  (1, 1, P, E) — both handled
        states:      object = None,  # unused; kept for API compatibility
        max_len:     int    = 20,
    ):
        """
        Greedy decoding (argmax at every step).

        Args:
            encoder_out : output of EncoderCNN; shape (1, P, embed_size).
                          If inference code calls encoder(img).unsqueeze(1),
                          the resulting (1, 1, P, E) shape is squeezed back.
            max_len     : maximum caption length.
        Returns:
            output : list[int] — predicted token ids (stops at <end> = 1 or max_len)
        """
        # Backward-compat: handle accidental unsqueeze(1)
        if encoder_out.dim() == 4:
            encoder_out = encoder_out.squeeze(1)    # (1, P, E)

        h, c   = self._init_hidden(encoder_out)      # (1, H)
        device = encoder_out.device

        # <start> token id is always 0  (see vocabulary.py build order)
        word_id = torch.zeros(1, dtype=torch.long, device=device)   # tensor([0])
        output  = []

        for _ in range(max_len):
            embed   = self.embedding(word_id)                      # (1, E)
            context, _ = self.attention(encoder_out, h)            # (1, E)
            gate    = torch.sigmoid(self.f_beta(h))
            context = gate * context
            h, c    = self.lstm_cell(
                torch.cat([embed, context], dim=1), (h, c)
            )
            scores   = self.fc(h)                                   # (1, vocab_size)
            word_id  = scores.argmax(dim=1)                        # (1,)
            token    = word_id.item()
            output.append(token)
            if token == 1:   # <end>
                break

        return output
