# model.py
#
# Attention-based image captioning model for the Udacity CVND project.
# Architecture: ResNet-50 encoder → Bahdanau attention → LSTMCell decoder.
#
# Three classes live here:
#   EncoderCNN  — extracts a spatial feature grid from the input image
#   Attention   — computes a soft attention distribution over that grid
#   DecoderRNN  — generates the caption word by word using the attended context

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


# ---------------------------------------------------------------------------
# EncoderCNN
# ---------------------------------------------------------------------------

class EncoderCNN(nn.Module):
    """
    Wraps a pretrained ResNet-50 and turns an image into a set of spatial
    feature vectors that the attention mechanism can selectively focus on.

    Instead of using ResNet's global average-pool output (a single 2048-d
    vector), we stop before the pooling layer and keep the full spatial map.
    An AdaptiveAvgPool then resizes it to a fixed S×S grid regardless of the
    input resolution.  Each of the S*S grid cells is a 2048-d descriptor
    representing one region of the image.

    Args:
        encoded_image_size: Side length S of the output grid.  7 gives a
            7×7 = 49-region grid, which is the standard choice for a 224×224
            input through ResNet-50.  Keep this between 1 and 32 — larger
            values cause quadratic memory growth and are never needed here.

    Output shape: (batch, S*S, 2048)
    """

    def __init__(self, encoded_image_size: int = 7):
        super().__init__()

        # Catch the common mistake of passing embed_size (e.g. 256 or 512)
        # instead of the spatial grid size.  AdaptiveAvgPool2d with a size of
        # 256 would allocate a 256×256 feature map — 1,300× more memory than
        # intended and an instant OOM on any GPU.
        if not isinstance(encoded_image_size, int):
            raise TypeError(
                f"encoded_image_size must be a plain int, got {type(encoded_image_size)}. "
                f"Did you accidentally pass embed_size here?"
            )
        if encoded_image_size < 1 or encoded_image_size > 32:
            raise ValueError(
                f"encoded_image_size={encoded_image_size} is outside the safe range [1, 32]. "
                f"Use a small spatial size like 7 (default) or 14. "
                f"Values like 256 or 512 will cause CUDA OOM."
            )

        self.enc_image_size = encoded_image_size

        # Load ResNet-50 with ImageNet weights.  The newer torchvision API uses
        # an explicit Weights enum; fall back to the deprecated flag for older
        # installations so the code runs on both.
        try:
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except Exception:
            resnet = models.resnet50(pretrained=True)

        # Record the channel depth before we strip the head — always 2048 for
        # ResNet-50 but reading it programmatically avoids hardcoding.
        self._encoder_dim = resnet.fc.in_features

        # The encoder is kept frozen by default.  The pretrained features are
        # already very good for natural images, and fine-tuning only pays off
        # after the decoder has learned something reasonable (typically epoch 10+).
        # Call fine_tune(enable=True) explicitly if you want to unfreeze later.
        for param in resnet.parameters():
            param.requires_grad = False

        # Drop the average-pool and the classification head.  Everything up to
        # (but not including) those two layers produces the spatial feature map
        # at (B, 2048, H/32, W/32) for a 224×224 input → (B, 2048, 7, 7).
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize whatever spatial resolution comes out of ResNet to the fixed
        # S×S grid the decoder expects.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

    @property
    def encoder_dim(self) -> int:
        """Channel depth of each spatial feature vector (2048 for ResNet-50)."""
        return self._encoder_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, 224, 224)  — normalised with ImageNet mean/std

        Returns:
            (B, S*S, 2048)  — e.g. (B, 49, 2048) for the default S=7
        """
        # Skip gradient tracking entirely when the encoder is frozen.
        # This saves roughly 40% of GPU memory during training because
        # PyTorch doesn't need to store intermediate activations for backprop.
        if not any(p.requires_grad for p in self.resnet.parameters()):
            with torch.no_grad():
                features = self.resnet(images)
        else:
            features = self.resnet(images)

        features = self.adaptive_pool(features)                               # (B, 2048, S, S)
        features = features.permute(0, 2, 3, 1)                              # (B, S, S, 2048)
        features = features.reshape(features.size(0), -1, features.size(-1)) # (B, S*S, 2048)
        return features

    def fine_tune(self, enable: bool = True):
        """
        Selectively unfreeze the deeper ResNet layers for fine-tuning.

        Only layer2, layer3, and layer4 are ever unfrozen — the earlier
        layers (conv1, bn1, layer1) learn very generic low-level features
        that transfer well to any image domain and don't need updating.
        Fine-tuning them usually hurts more than it helps.

        Don't call this until the decoder has trained for at least 10 epochs.
        Starting fine-tuning too early destabilises training because the
        decoder gradients are still noisy and will corrupt the pretrained
        ResNet weights.
        """
        # Always start by freezing everything, then selectively unfreeze.
        # This way the method is idempotent — calling fine_tune(False) is
        # a safe way to re-freeze without worrying about the current state.
        for p in self.resnet.parameters():
            p.requires_grad = False

        if enable:
            # In nn.Sequential the ResNet children are indexed by insertion
            # order: 0=conv1, 1=bn1, 2=relu, 3=maxpool, 4=layer1,
            # 5=layer2, 6=layer3, 7=layer4.  We unfreeze indices 5, 6, 7.
            for name, child in self.resnet.named_children():
                if name in ("5", "6", "7"):
                    for p in child.parameters():
                        p.requires_grad = True


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """
    Bahdanau (additive) attention.

    At each decoder timestep the attention module receives the full set of
    encoder feature vectors and the decoder's current hidden state, and
    produces a single context vector that summarises which parts of the image
    the decoder should focus on right now.

    Mathematically:
        energy_i = W_full · tanh(W_enc · encoder_out_i + W_dec · h)
        alpha    = softmax(energy)               — weights summing to 1
        context  = sum_i(alpha_i * encoder_out_i) — weighted average

    Args:
        encoder_dim:   Depth of each encoder feature vector (2048).
        decoder_dim:   Size of the decoder LSTM hidden state.
        attention_dim: Internal projection dimension.  Both the encoder
                       features and the decoder hidden state are projected
                       into this space before being added together.
    """

    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        super().__init__()

        # Project encoder features into attention space.
        # Applied once per forward call across all S*S positions simultaneously.
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)

        # Project the decoder hidden state into the same attention space.
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)

        # Collapse the attention_dim representation down to a scalar energy
        # score for each spatial position.
        self.full_att = nn.Linear(attention_dim, 1)

        self.relu    = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)   # normalise across spatial positions

    def forward(self, encoder_out: torch.Tensor, decoder_hidden: torch.Tensor):
        """
        Args:
            encoder_out:    (B, num_pixels, encoder_dim)
            decoder_hidden: (B, decoder_dim)  — h_t from the LSTMCell

        Returns:
            context: (B, encoder_dim)   — attended image representation
            alpha:   (B, num_pixels)    — attention weights; useful for visualising
                                          where the model is looking at each step
        """
        # Project and combine.  The unsqueeze on att2 broadcasts the single
        # decoder vector across all num_pixels encoder positions before adding.
        att1   = self.encoder_att(encoder_out)                       # (B, num_pixels, att_dim)
        att2   = self.decoder_att(decoder_hidden).unsqueeze(1)       # (B, 1,          att_dim)
        energy = self.full_att(self.relu(att1 + att2)).squeeze(2)    # (B, num_pixels)

        alpha   = self.softmax(energy)                               # (B, num_pixels)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)      # (B, encoder_dim)

        return context, alpha


# ---------------------------------------------------------------------------
# DecoderRNN
# ---------------------------------------------------------------------------

class DecoderRNN(nn.Module):
    """
    Generates a caption one word at a time using an attention-gated LSTMCell.

    At each step t:
      1. The attention module reads the encoder features and the current hidden
         state to produce a context vector — a weighted summary of the image.
      2. The context is concatenated with the embedding of the previous word.
      3. The LSTMCell updates its hidden state using that concatenated input.
      4. A linear layer projects the hidden state to a vocabulary distribution.

    During training we use teacher forcing: the ground-truth word at position t
    is always fed as input at position t+1, which stabilises early training.
    During inference we autoregressively feed the model's own predictions.

    Args:
        attention_dim: Projection size inside the Attention module.
        embed_size:    Dimensionality of the word embedding vectors.
        hidden_size:   Number of units in the LSTMCell hidden state.
        vocab_size:    Total number of tokens in the vocabulary.
        encoder_dim:   Depth of encoder feature vectors (2048 for ResNet-50).
        dropout:       Dropout rate applied to the hidden state before the
                       output projection.  Only active in training mode.
    """

    def __init__(
        self,
        attention_dim: int,
        embed_size:    int,
        hidden_size:   int,
        vocab_size:    int,
        encoder_dim:   int   = 2048,
        dropout:       float = 0.5,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.hidden_size = hidden_size
        self.vocab_size  = vocab_size

        self.attention = Attention(encoder_dim, hidden_size, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout   = nn.Dropout(dropout)

        # The LSTM input at each step is the word embedding concatenated with
        # the context vector, so the input size is embed_size + encoder_dim.
        self.lstm = nn.LSTMCell(embed_size + encoder_dim, hidden_size)

        # The initial hidden and cell states are computed from the mean of the
        # encoder features rather than being zeros — this gives the decoder a
        # warm start that already reflects the overall image content.
        self.init_h = nn.Linear(encoder_dim, hidden_size)
        self.init_c = nn.Linear(encoder_dim, hidden_size)

        # Final projection from hidden state to vocabulary logits.
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden_state(self, encoder_out: torch.Tensor):
        """
        Derive the initial LSTM (h, c) from the mean-pooled encoder output.

        Averaging over all spatial positions gives a global image summary that
        is a much better starting point than zeros, particularly in the first
        few timesteps before the attention mechanism has settled.

        Args:
            encoder_out: (B, num_pixels, encoder_dim)

        Returns:
            h: (B, hidden_size)
            c: (B, hidden_size)
        """
        mean_enc = encoder_out.mean(dim=1)        # collapse spatial dim → (B, encoder_dim)
        h = torch.tanh(self.init_h(mean_enc))
        c = torch.tanh(self.init_c(mean_enc))
        return h, c

    # ------------------------------------------------------------------
    # Training forward pass
    # ------------------------------------------------------------------

    def forward(self, encoder_out: torch.Tensor, captions: torch.Tensor):
        """
        Teacher-forcing forward pass used during training.

        We feed the ground-truth token at position t as input at step t+1.
        That means we process positions 0 … seq_len-2 and predict positions
        1 … seq_len-1.  For example, if captions = [<start>, w1, w2, <end>],
        the inputs are [<start>, w1, w2] and the targets are [w1, w2, <end>].

        Args:
            encoder_out: (B, num_pixels, encoder_dim)
            captions:    (B, seq_len)  — token ids including <start> and <end>

        Returns:
            outputs: (B, seq_len-1, vocab_size)  — unnormalised logits
            alphas:  (B, seq_len-1, num_pixels)  — attention weights per step
        """
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)

        # Drop the final <end> token from inputs — we never need to feed it
        # because there is no next word to predict after <end>.
        embeddings = self.embedding(captions[:, :-1])   # (B, seq_len-1, embed_size)

        h, c = self.init_hidden_state(encoder_out)

        outputs = torch.zeros(batch_size, embeddings.size(1), self.vocab_size, device=encoder_out.device)
        alphas  = torch.zeros(batch_size, embeddings.size(1), num_pixels,      device=encoder_out.device)

        for t in range(embeddings.size(1)):
            context, alpha = self.attention(encoder_out, h)

            # Concatenate word embedding and attended context, then step the LSTM.
            lstm_input = torch.cat([embeddings[:, t, :], context], dim=1)
            h, c = self.lstm(lstm_input, (h, c))

            # Apply dropout before the output projection.  In eval mode
            # dropout is a no-op, so training and inference use identical paths.
            outputs[:, t, :] = self.fc(self.dropout(h))
            alphas[:, t, :]  = alpha

        return outputs, alphas

    # ------------------------------------------------------------------
    # Greedy decoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        encoder_out: torch.Tensor,
        start_idx:   int,
        end_idx:     int,
        max_len:     int = 20,
    ):
        """
        Generate a caption greedily — always pick the highest-probability word.

        Greedy decoding is fast and good enough for quick checks during training,
        but it tends to produce safe, sometimes repetitive captions.  For final
        evaluation and the inference notebook, prefer beam_search().

        Must be called after decoder.eval().  If called in training mode,
        Dropout randomly zeros hidden units every step, making every caption
        different and usually wrong.  The assert here surfaces this mistake
        immediately instead of letting it produce mysteriously bad output.

        Args:
            encoder_out: (1, num_pixels, encoder_dim)  — single image only
            start_idx:   Token id of <start>
            end_idx:     Token id of <end>
            max_len:     Cap on output length (not counting <start> / <end>)

        Returns:
            List of token ids, not including <start> or <end>.
            An empty list means <end> was predicted on the very first step,
            which usually means the checkpoint is from a failed training run.
        """
        assert not self.training, (
            "Call decoder.eval() before sample(). "
            "In training mode, Dropout randomly zeros hidden units and "
            "BatchNorm uses noisy single-sample statistics — both corrupt captions."
        )

        if encoder_out.size(0) != 1:
            raise ValueError(
                f"sample() is designed for a single image (batch size 1), "
                f"but received batch size {encoder_out.size(0)}."
            )

        device = encoder_out.device
        h, c   = self.init_hidden_state(encoder_out)
        word   = torch.tensor([start_idx], device=device)
        output_ids = []

        for _ in range(max_len):
            embed      = self.embedding(word)
            context, _ = self.attention(encoder_out, h)
            h, c       = self.lstm(torch.cat([embed, context], dim=1), (h, c))

            # dropout(h) is an identity in eval mode — the call is here only
            # to keep this path structurally identical to the training forward.
            logits = self.fc(self.dropout(h))     # (1, vocab_size)
            word   = logits.argmax(dim=1)         # (1,)

            if word.item() == end_idx:
                break

            output_ids.append(word.item())

            # If the model predicts the same word three times in a row it has
            # entered a degenerate repetition loop.  Stop early to avoid filling
            # the caption with noise.
            if len(output_ids) >= 3 and (
                output_ids[-1] == output_ids[-2] == output_ids[-3]
            ):
                break

        return output_ids

    # ------------------------------------------------------------------
    # Beam search decoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def beam_search(
        self,
        encoder_out:       torch.Tensor,
        start_idx:         int,
        end_idx:           int,
        beam_size:         int   = 5,
        max_len:           int   = 20,
        length_norm_alpha: float = 0.7,
    ):
        """
        Generate a caption using beam search.

        Beam search maintains the top-k partial sequences at every step instead
        of committing to a single greedy choice.  This consistently produces
        better captions than greedy decoding at the cost of running k forward
        passes per image instead of one.

        length_norm_alpha controls how much to penalise short captions.
        Alpha=0.0 means no normalisation (longer beams win simply by accumulating
        more log-probability terms). Alpha=1.0 divides the score by raw length.
        0.7 is the value from the Google NMT paper and works well here.

        Args:
            encoder_out:       (1, num_pixels, encoder_dim)
            start_idx:         Token id of <start>
            end_idx:           Token id of <end>
            beam_size:         Number of concurrent hypotheses to keep (k).
            max_len:           Maximum tokens to generate per beam.
            length_norm_alpha: Length penalty exponent.

        Returns:
            List of token ids, not including <start> or <end>.
        """
        if encoder_out.size(0) != 1:
            raise ValueError(
                f"beam_search() expects a single image (batch size 1), "
                f"got {encoder_out.size(0)}."
            )

        device = encoder_out.device

        # Duplicate the encoder output once for each beam so that attention
        # can process all k hypotheses in a single batched call.
        enc = encoder_out.expand(beam_size, -1, -1).contiguous()   # (k, num_pixels, enc_dim)

        h, c = self.init_hidden_state(enc)   # (k, hidden)

        # Every beam starts with the single token <start>.
        seqs       = torch.full((beam_size, 1), start_idx, dtype=torch.long, device=device)
        seq_scores = torch.zeros(beam_size, device=device)   # cumulative log-prob per beam

        completed_seqs   = []
        completed_scores = []
        k = beam_size

        for _ in range(max_len):
            last_words = seqs[:, -1]                                      # (k,)
            embeds     = self.embedding(last_words)                        # (k, embed)
            context, _ = self.attention(enc, h)                            # (k, enc_dim)

            h, c = self.lstm(torch.cat([embeds, context], dim=1), (h, c))

            log_probs    = torch.log_softmax(self.fc(h), dim=1)            # (k, vocab)
            total_scores = seq_scores.unsqueeze(1) + log_probs             # (k, vocab)

            # Flatten across (beam, vocab) and pick the global top-k.
            # This is the heart of beam search — we choose the k best
            # continuations across all beams and all vocabulary tokens at once.
            topk_scores, topk_flat = total_scores.view(-1).topk(k)
            beam_idx  = topk_flat // self.vocab_size   # which beam each winner came from
            token_idx = topk_flat  % self.vocab_size   # which token was chosen

            # Extend every surviving beam with its selected token.
            seqs = torch.cat([seqs[beam_idx], token_idx.unsqueeze(1)], dim=1)

            # Separate completed beams from those still generating.
            # Doing this in a single pass is important: we need the list of
            # incomplete indices to re-index h, c, and enc in one atomic step
            # below.  Two separate passes (collect, then filter) would corrupt
            # hidden states because intermediate indexing would be wrong once
            # beam counts change.
            incomplete = []
            for i in range(k):
                if token_idx[i].item() == end_idx:
                    completed_seqs.append(seqs[i].clone())
                    completed_scores.append(topk_scores[i].item())
                else:
                    incomplete.append(i)

            if not incomplete:
                break

            inc = torch.tensor(incomplete, device=device)

            # Re-index all beam-dependent tensors using beam_idx[inc] in one
            # combined operation.  This is the fix for a subtle bug in the
            # original code, which applied beam_idx and inc as two separate
            # index operations — that double-indexing caused h/c to diverge
            # from seqs after the first beam completed.
            seqs        = seqs[inc]
            seq_scores  = topk_scores[inc]
            h           = h[beam_idx[inc]]
            c           = c[beam_idx[inc]]
            enc         = enc[beam_idx[inc]]
            k           = seqs.size(0)

        # If no beam ever emitted <end>, use the best scoring incomplete beam.
        if not completed_seqs:
            best = seqs[seq_scores.argmax().item()]
        else:
            # Length-normalise before picking the winner so that short sequences
            # are not unfairly preferred over longer, more descriptive ones.
            norm_scores = [
                sc / max(1, seq.size(0)) ** length_norm_alpha
                for seq, sc in zip(completed_seqs, completed_scores)
            ]
            best = completed_seqs[int(torch.tensor(norm_scores).argmax().item())]

        # Convert to a plain list and strip the bookend special tokens.
        best = best.tolist()
        if best and best[0] == start_idx:
            best = best[1:]
        if end_idx in best:
            best = best[:best.index(end_idx)]

        return best
