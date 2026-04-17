# model.py

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
        # AdaptiveAvgPool2d would try to output (2048 x 2048) spatial map -> HUGE memory.
        if not isinstance(encoded_image_size, int):
            raise TypeError(f"encoded_image_size must be int, got {type(encoded_image_size)}")

        if encoded_image_size < 1 or encoded_image_size > 32:
            raise ValueError(
                f"encoded_image_size={encoded_image_size} is invalid/unsafe. "
                f"Use a small spatial size like 7, 14, or 8. "
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
        # If resnet is frozen, save memory by not storing intermediate gradients.
        # (Even if caller forgot torch.no_grad(), this keeps inference safe.)
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
        sample() -> greedy decoding
        beam_search() -> beam decoding (assumes batch=1 encoder_out)
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
    # Greedy decoding
    # -----------------------------
    @torch.no_grad()
    def sample(self, encoder_out: torch.Tensor, start_idx: int, end_idx: int, max_len: int = 20):
        """
        Greedy caption generation.

        NOTE:
            This implementation is intended for batch size 1 in typical CVND inference.
            If you pass B>1, you should vectorize word selection per batch.
        """
        device = encoder_out.device

        if encoder_out.size(0) != 1:
            raise ValueError(f"sample() expects batch size 1, got {encoder_out.size(0)}")

        h, c = self.init_hidden_state(encoder_out)

        word = torch.tensor([start_idx], device=device)
        output_ids = []

        for _ in range(max_len):
            embed = self.embedding(word)          # (1, embed_size)
            context, _ = self.attention(encoder_out, h)

            h, c = self.lstm(torch.cat([embed, context], dim=1), (h, c))
            scores = self.fc(h)                   # (1, vocab_size)
            word = scores.argmax(dim=1)           # (1,)

            if word.item() == end_idx:
                break

            output_ids.append(word.item())

        return output_ids

    # -----------------------------
    # Beam search decoding
    # -----------------------------
    @torch.no_grad()
    def beam_search(
        self,
        encoder_out: torch.Tensor,
        start_idx: int,
        end_idx: int,
        beam_size: int = 3,
        max_len: int = 20,
        alpha: float = 0.7,
    ):
        """
        Beam search caption generation.

        Assumes:
            encoder_out: (1, num_pixels, encoder_dim)
        Returns:
            best_seq: list of token ids (without <start>/<end>)
        """
        device = encoder_out.device

        if encoder_out.size(0) != 1:
            raise ValueError(f"beam_search() expects batch size 1, got {encoder_out.size(0)}")

        h, c = self.init_hidden_state(encoder_out)

        beams = [(0.0, [start_idx], h, c)]
        completed = []

        for _ in range(max_len):
            new_beams = []

            for log_prob, seq, h_b, c_b in beams:
                if seq[-1] == end_idx:
                    completed.append((log_prob, seq))
                    continue

                word = torch.tensor([seq[-1]], device=device)
                embed = self.embedding(word)
                context, _ = self.attention(encoder_out, h_b)

                h_new, c_new = self.lstm(torch.cat([embed, context], dim=1), (h_b, c_b))
                scores = self.fc(h_new)
                log_probs = torch.log_softmax(scores, dim=1)

                top_log_probs, top_words = log_probs.topk(beam_size, dim=1)

                for i in range(beam_size):
                    next_word = top_words[0, i].item()
                    next_log_prob = top_log_probs[0, i].item()

                    new_seq = seq + [next_word]
                    new_log_prob = log_prob + next_log_prob

                    new_beams.append((new_log_prob, new_seq, h_new, c_new))

            if not new_beams:
                break

            beams = sorted(
                new_beams,
                key=lambda x: x[0] / (len(x[1]) ** alpha),
                reverse=True,
            )[:beam_size]

            if all(seq[-1] == end_idx for _, seq, _, _ in beams):
                break

        for log_prob, seq, _, _ in beams:
            completed.append((log_prob, seq))

        completed.sort(
            key=lambda x: x[0] / (len(x[1]) ** alpha),
            reverse=True,
        )

        best_seq = completed[0][1]

        if best_seq and best_seq[0] == start_idx:
            best_seq = best_seq[1:]
        if best_seq and best_seq[-1] == end_idx:
            best_seq = best_seq[:-1]

        return best_seq
