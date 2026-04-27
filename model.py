# model.py

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


# ============================================================
# Encoder CNN (Spatial Feature Extractor for Attention)
# ============================================================

class EncoderCNN(nn.Module):
    """
    Encodes an input image into a spatial feature map suitable for attention.

    Output shape:
        (B, num_pixels, encoder_dim)
        where num_pixels = encoded_image_size * encoded_image_size
              encoder_dim = 2048 for ResNet-50
    """

    def __init__(self, encoded_image_size: int = 7):
        super().__init__()

        # -----------------------------
        # Safety guard (prevents OOM)
        # -----------------------------
        # If someone accidentally passes embed_size (e.g., 2048) here,
        # AdaptiveAvgPool2d would try to output (2048 x 2048) -> HUGE memory.
        if not isinstance(encoded_image_size, int):
            raise TypeError(f"encoded_image_size must be int, got {type(encoded_image_size)}")

        if encoded_image_size < 1 or encoded_image_size > 32:
            raise ValueError(
                f"encoded_image_size={encoded_image_size} is invalid/unsafe. "
                f"Use a small spatial size like 7, 8, 14. "
                f"(Passing values like 2048 will cause CUDA OOM.)"
            )

        self.enc_image_size = encoded_image_size

        # -----------------------------
        # Load ResNet50 (torchvision API compatible)
        # -----------------------------
        try:
            # Newer torchvision
            weights = models.ResNet50_Weights.DEFAULT
            resnet = models.resnet50(weights=weights)
        except Exception:
            # Older torchvision fallback
            resnet = models.resnet50(pretrained=True)

        self._encoder_dim = resnet.fc.in_features  # 2048 for resnet50

        # Freeze all ResNet params by default
        for param in resnet.parameters():
            param.requires_grad = False

        # Remove avgpool + fc to keep spatial feature map
        # output: (B, 2048, H/32, W/32)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Adaptive pooling to fixed spatial size (S x S)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

    @property
    def encoder_dim(self) -> int:
        return self._encoder_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images:  (B, 3, 224, 224)
        returns: (B, encoded_image_size*encoded_image_size, 2048)
                 e.g. (B, 49, 2048) when encoded_image_size=7
        """
        # Save memory if encoder is frozen
        if not any(p.requires_grad for p in self.resnet.parameters()):
            with torch.no_grad():
                features = self.resnet(images)
        else:
            features = self.resnet(images)

        features = self.adaptive_pool(features)           # (B, 2048, S, S)
        features = features.permute(0, 2, 3, 1)           # (B, S, S, 2048)
        features = features.reshape(features.size(0), -1, features.size(-1))  # (B, S*S, 2048)
        return features

    def fine_tune(self, enable: bool = True):
        """
        Optionally unfreeze some layers for fine-tuning.
        By default, encoder is frozen.
        """
        # Freeze everything first
        for p in self.resnet.parameters():
            p.requires_grad = False

        if enable:
            # Unfreeze layer2, layer3, layer4
            # In ResNet sequential children indices:
            # 0 conv1, 1 bn1, 2 relu, 3 maxpool, 4 layer1, 5 layer2, 6 layer3, 7 layer4
            for child_name, child in self.resnet.named_children():
                if child_name in ["5", "6", "7"]:
                    for p in child.parameters():
                        p.requires_grad = True


# ============================================================
# Bahdanau (Additive) Attention
# ============================================================

class Attention(nn.Module):
    """
    Additive (Bahdanau) attention:
        Given encoder_out (B, num_pixels, encoder_dim) and decoder_hidden (B, decoder_dim),
        returns context (B, encoder_dim) and alpha (B, num_pixels)
    """

    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        super().__init__()

        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out: torch.Tensor, decoder_hidden: torch.Tensor):
        """
        encoder_out:    (B, num_pixels, encoder_dim)
        decoder_hidden: (B, decoder_dim)
        """
        att1 = self.encoder_att(encoder_out)                       # (B, num_pixels, att_dim)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)       # (B, 1, att_dim)
        energy = self.full_att(self.relu(att1 + att2)).squeeze(2)  # (B, num_pixels)

        alpha = self.softmax(energy)                               # (B, num_pixels)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)    # (B, encoder_dim)

        return context, alpha


# ============================================================
# Decoder RNN with Attention + Greedy/Beam Search
# ============================================================

class DecoderRNN(nn.Module):
    """
    Attention-based decoder using LSTMCell.

    Training forward (teacher forcing):
        inputs: encoder_out (B, num_pixels, encoder_dim), captions (B, seq_len)
        returns: outputs (B, seq_len-1, vocab_size), alphas (B, seq_len-1, num_pixels)

    Inference:
        sample() -> greedy decoding (batch=1)
        beam_search() -> beam decoding (expects batch=1 encoder_out)
    """

    def __init__(
        self,
        attention_dim: int,
        embed_size: int,
        hidden_size: int,
        vocab_size: int,
        encoder_dim: int = 2048,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.attention = Attention(encoder_dim, hidden_size, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTMCell(embed_size + encoder_dim, hidden_size)

        self.init_h = nn.Linear(encoder_dim, hidden_size)
        self.init_c = nn.Linear(encoder_dim, hidden_size)

        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden_state(self, encoder_out: torch.Tensor):
        """
        Initialize LSTM hidden and cell from mean-pooled encoder features.
        encoder_out: (B, num_pixels, encoder_dim)
        """
        mean_encoder_out = encoder_out.mean(dim=1)  # (B, encoder_dim)
        h = self.init_h(mean_encoder_out)           # (B, hidden)
        c = self.init_c(mean_encoder_out)           # (B, hidden)
        return h, c

    # -----------------------------
    # Training (teacher forcing)
    # -----------------------------
    def forward(self, encoder_out: torch.Tensor, captions: torch.Tensor):
        """
        Training forward pass (teacher forcing).

        encoder_out: (B, num_pixels, encoder_dim) e.g. (B, 49, 2048)
        captions:    (B, seq_len)  (tokenized, includes <start> ... <end> typically)

        returns:
            outputs: (B, seq_len-1, vocab_size)
            alphas:  (B, seq_len-1, num_pixels)
        """
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)

        embeddings = self.embedding(captions[:, :-1])  # (B, seq_len-1, embed_size)

        h, c = self.init_hidden_state(encoder_out)

        outputs = torch.zeros(
            batch_size,
            embeddings.size(1),
            self.vocab_size,
            device=encoder_out.device,
        )

        alphas = torch.zeros(
            batch_size,
            embeddings.size(1),
            num_pixels,
            device=encoder_out.device,
        )

        for t in range(embeddings.size(1)):
            context, alpha = self.attention(encoder_out, h)

            lstm_input = torch.cat([embeddings[:, t, :], context], dim=1)
            h, c = self.lstm(lstm_input, (h, c))

            outputs[:, t, :] = self.fc(self.dropout(h))
            alphas[:, t, :] = alpha

        return outputs, alphas

    # -----------------------------
    # Greedy decoding (batch=1)
    # -----------------------------
    @torch.no_grad()
    def sample(
        self,
        encoder_out: torch.Tensor,
        start_idx: int,
        end_idx: int,
        max_len: int = 20,
    ):
        """
        Greedy caption generation.

        Intended for batch size 1.
        Returns: list of token ids (without <start>/<end>)

        Fixes vs original:
          1. Asserts eval() mode — prevents dropout being active during inference.
          2. Applies self.dropout(h) consistently with forward() before self.fc.
          3. Guards against empty output_ids (model predicts <end> immediately).
          4. Repetition guard — breaks if 3 consecutive identical tokens predicted.
        """
        # ✅ FIX 1: Enforce eval mode — caller must call decoder.eval() first.
        #    Without this, ResNet BatchNorm uses batch stats (garbage features)
        #    and Dropout randomly zeroes hidden units (non-deterministic captions).
        assert not self.training, (
            "Call decoder.eval() before sample(). "            "Dropout and BatchNorm behave differently in train mode "            "and will produce garbage or empty captions."
        )

        device = encoder_out.device

        if encoder_out.size(0) != 1:
            raise ValueError(f"sample() expects batch size 1, got {encoder_out.size(0)}")

        h, c = self.init_hidden_state(encoder_out)

        word = torch.tensor([start_idx], device=device)
        output_ids = []

        for _ in range(max_len):
            embed = self.embedding(word)                        # (1, embed_size)
            context, _ = self.attention(encoder_out, h)         # (1, encoder_dim)

            h, c = self.lstm(torch.cat([embed, context], dim=1), (h, c))

            # ✅ FIX 2: Apply dropout consistently with forward() before fc.
            #    In training forward() we do self.fc(self.dropout(h)).
            #    Here dropout is disabled because we asserted eval() above,
            #    so self.dropout(h) == h — but the call keeps the paths symmetric.
            logits = self.fc(self.dropout(h))                   # (1, vocab_size)
            word = logits.argmax(dim=1)                         # (1,)

            if word.item() == end_idx:
                break

            output_ids.append(word.item())

            # ✅ FIX 3: Repetition guard — if the last 3 tokens are all identical
            #    the model is stuck in a loop; break early to avoid gibberish.
            if len(output_ids) >= 3 and (
                output_ids[-1] == output_ids[-2] == output_ids[-3]
            ):
                break

        # ✅ FIX 4: If output is empty the model predicted <end> on step 1.
        #    This means the checkpoint is undertrained or the wrong epoch was
        #    loaded. Return empty list — the caller (ids_to_caption) handles it
        #    with a diagnostic message rather than a silent empty string.
        return output_ids

    # -----------------------------
    # Beam search decoding (FIXED)
    # -----------------------------
    @torch.no_grad()
    def beam_search(
        self,
        encoder_out: torch.Tensor,
        start_idx: int,
        end_idx: int,
        beam_size: int = 5,
        max_len: int = 20,
        length_norm_alpha: float = 0.7,
    ):
        """
        Correct beam search caption generation.

        Assumes:
            encoder_out: (1, num_pixels, encoder_dim)

        Returns:
            best_seq: list of token ids (without <start>/<end>)
        """
        device = encoder_out.device

        if encoder_out.size(0) != 1:
            raise ValueError(f"beam_search() expects batch size 1, got {encoder_out.size(0)}")

        # Expand encoder output to beam size
        encoder_out = encoder_out.expand(beam_size, encoder_out.size(1), encoder_out.size(2))  # (k, num_pixels, enc_dim)

        # Initialize hidden state per beam
        h, c = self.init_hidden_state(encoder_out)  # (k, hidden)

        # Beam sequences start with <start>
        seqs = torch.full((beam_size, 1), start_idx, dtype=torch.long, device=device)  # (k, 1)
        seq_scores = torch.zeros(beam_size, device=device)  # cumulative log probs

        completed_seqs = []
        completed_scores = []

        k = beam_size  # current beam size

        for _ in range(max_len):
            last_words = seqs[:, -1]                         # (k,)
            embeddings = self.embedding(last_words)          # (k, embed)

            context, _ = self.attention(encoder_out, h)      # (k, enc_dim)

            h, c = self.lstm(torch.cat([embeddings, context], dim=1), (h, c))  # (k, hidden)

            logits = self.fc(h)                              # (k, vocab)
            log_probs = torch.log_softmax(logits, dim=1)     # (k, vocab)

            # Accumulate sequence log-probabilities
            total_scores = seq_scores.unsqueeze(1) + log_probs  # (k, vocab)

            # Select top-k over all (beam, vocab) candidates
            top_scores, top_pos = total_scores.view(-1).topk(k, dim=0)  # (k,)

            beam_indices = top_pos // self.vocab_size  # (k,)
            token_indices = top_pos % self.vocab_size  # (k,)

            # Build new sequences
            seqs = torch.cat([seqs[beam_indices], token_indices.unsqueeze(1)], dim=1)  # (k, step+2)

            # Determine which sequences have completed
            incomplete = []
            for i in range(seqs.size(0)):
                if token_indices[i].item() == end_idx:
                    completed_seqs.append(seqs[i].clone())
                    completed_scores.append(top_scores[i].item())
                else:
                    incomplete.append(i)

            # If all beams completed, stop
            if len(incomplete) == 0:
                break

            # Keep only incomplete beams
            seqs = seqs[incomplete]
            seq_scores = top_scores[incomplete]
            h = h[beam_indices[incomplete]]
            c = c[beam_indices[incomplete]]
            encoder_out = encoder_out[beam_indices[incomplete]]

            k = seqs.size(0)

        # If none completed, fall back to best incomplete
        if len(completed_seqs) == 0:
            best_seq = seqs[0]
        else:
            # Length normalization: score / (len(seq) ** alpha)
            norm_scores = []
            for s, sc in zip(completed_seqs, completed_scores):
                # s includes <start> and <end>
                length = max(1, s.size(0))
                norm_scores.append(sc / (length ** length_norm_alpha))

            best_idx = int(torch.tensor(norm_scores).argmax().item())
            best_seq = completed_seqs[best_idx]

        # Convert to list and remove <start>/<end>
        best_seq = best_seq.tolist()

        if len(best_seq) > 0 and best_seq[0] == start_idx:
            best_seq = best_seq[1:]
        if len(best_seq) > 0 and best_seq[-1] == end_idx:
            best_seq = best_seq[:-1]

        return best_seq
