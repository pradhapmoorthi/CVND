import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EncoderCNN(nn.Module):
    """
    Encoder that outputs spatial feature map:
      output shape: (B, num_pixels, encoder_dim)
      where num_pixels = encoded_image_size * encoded_image_size
    """
    def __init__(self, encoded_image_size=14, fine_tune=False):
        super().__init__()
        self.enc_image_size = encoded_image_size

        # Pretrained ResNet-50 backbone (remove avgpool and fc)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # until conv5_x output

        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune(fine_tune)

    def forward(self, images):
        """
        images: (B, 3, H, W)
        returns: (B, num_pixels, encoder_dim)
        """
        features = self.resnet(images)              # (B, 2048, H/32, W/32)
        features = self.adaptive_pool(features)     # (B, 2048, enc, enc)

        B, C, H, W = features.size()
        features = features.permute(0, 2, 3, 1).contiguous()  # (B, enc, enc, 2048)
        features = features.view(B, -1, C)                    # (B, num_pixels, 2048)
        return features

    def fine_tune(self, fine_tune=False):
        """
        If fine_tune=True, unfreeze last conv blocks to fine-tune.
        """
        for p in self.resnet.parameters():
            p.requires_grad = False

        if fine_tune:
            # Unfreeze last 2 blocks for mild fine-tuning
            for c in list(self.resnet.children())[-2:]:
                for p in c.parameters():
                    p.requires_grad = True


class Attention(nn.Module):
    """
    Additive (Bahdanau) attention over spatial encoder features.
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

    def forward(self, encoder_out, decoder_hidden):
        """
        encoder_out: (B, num_pixels, encoder_dim)
        decoder_hidden: (B, decoder_dim)
        returns:
          context: (B, encoder_dim)
          alpha: (B, num_pixels)
        """
        att1 = self.encoder_att(encoder_out)                    # (B, num_pixels, att_dim)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)    # (B, 1, att_dim)
        e = self.full_att(torch.tanh(att1 + att2)).squeeze(2)   # (B, num_pixels)
        alpha = F.softmax(e, dim=1)                             # (B, num_pixels)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1) # (B, encoder_dim)
        return context, alpha


class DecoderRNN(nn.Module):
    """
    Attention-based LSTM decoder.
    """
    def __init__(self, embed_size, hidden_size, vocab_size,
                 encoder_dim=2048, attention_dim=512, dropout=0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.hidden_size = hidden_size

        self.attention = Attention(encoder_dim, hidden_size, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)

        # Context gating (often helps)
        self.f_beta = nn.Linear(hidden_size, encoder_dim)
        self.sigmoid = nn.Sigmoid()

        # LSTMCell input is concatenated [word_embedding, context_vector]
        self.decode_step = nn.LSTMCell(embed_size + encoder_dim, hidden_size, bias=True)

        # init hidden from mean encoder features
        self.init_h = nn.Linear(encoder_dim, hidden_size)
        self.init_c = nn.Linear(encoder_dim, hidden_size)

        self.fc = nn.Linear(hidden_size, vocab_size)
        self._init_weights()

    def _init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """
        encoder_out: (B, num_pixels, encoder_dim)
        """
        mean_enc = encoder_out.mean(dim=1)    # (B, encoder_dim)
        h = self.init_h(mean_enc)             # (B, hidden)
        c = self.init_c(mean_enc)
        return h, c

    def forward(self, encoder_out, captions, lengths):
        """
        Training forward.
        encoder_out: (B, num_pixels, encoder_dim)
        captions: (B, max_len) with <start> at pos 0
        lengths: list[int] lengths including <start>/<end>

        returns:
          predictions: (B, max_len-1, vocab_size)
          alphas: (B, max_len-1, num_pixels)
        """
        B = encoder_out.size(0)
        num_pixels = encoder_out.size(1)

        embeddings = self.embedding(captions)  # (B, max_len, embed)

        h, c = self.init_hidden_state(encoder_out)

        max_t = max(lengths) - 1  # predict next token for each time step
        predictions = torch.zeros(B, max_t, self.vocab_size, device=encoder_out.device)
        alphas = torch.zeros(B, max_t, num_pixels, device=encoder_out.device)

        for t in range(max_t):
            context, alpha = self.attention(encoder_out, h)

            gate = self.sigmoid(self.f_beta(h))
            context = gate * context

            lstm_input = torch.cat([embeddings[:, t, :], context], dim=1)
            h, c = self.decode_step(lstm_input, (h, c))

            preds = self.fc(self.dropout(h))
            predictions[:, t, :] = preds
            alphas[:, t, :] = alpha

        return predictions, alphas

    @torch.no_grad()
    def beam_search(self, encoder_out, start_idx, end_idx, beam_size=5,
                    max_len=25, length_norm_alpha=0.7):
        """
        Beam search decoding for a single image.
        encoder_out: (1, num_pixels, encoder_dim)
        returns: list[int] token ids
        """
        device = encoder_out.device
        assert encoder_out.size(0) == 1, "beam_search expects batch size 1"

        num_pixels = encoder_out.size(1)

        # Expand to (beam, num_pixels, enc_dim)
        encoder_out = encoder_out.expand(beam_size, num_pixels, self.encoder_dim)

        h, c = self.init_hidden_state(encoder_out)

        seqs = torch.full((beam_size, 1), start_idx, dtype=torch.long, device=device)
        scores = torch.zeros(beam_size, device=device)

        complete_seqs = []
        complete_scores = []

        cur_beam = beam_size

        for t in range(max_len):
            prev_words = seqs[:, -1]                 # (beam,)
            embeddings = self.embedding(prev_words)  # (beam, embed)

            context, alpha = self.attention(encoder_out, h)
            gate = self.sigmoid(self.f_beta(h))
            context = gate * context

            inp = torch.cat([embeddings, context], dim=1)
            h, c = self.decode_step(inp, (h, c))

            logits = self.fc(h)
            log_probs = F.log_softmax(logits, dim=1)

            # Add log probs to existing beam scores
            total_scores = scores.unsqueeze(1) + log_probs     # (beam, vocab)
            flat_scores = total_scores.view(-1)                # (beam*vocab,)

            top_scores, top_pos = flat_scores.topk(cur_beam, dim=0)

            beam_indices = top_pos // self.vocab_size
            word_indices = top_pos % self.vocab_size

            seqs = torch.cat([seqs[beam_indices], word_indices.unsqueeze(1)], dim=1)
            h = h[beam_indices]
            c = c[beam_indices]
            encoder_out = encoder_out[beam_indices]
            scores = top_scores

            # Completed sequences
            done = (word_indices == end_idx)
            if done.any():
                done_idx = torch.where(done)[0].tolist()
                for i in done_idx:
                    seq = seqs[i].clone()
                    L = seq.size(0)
                    norm = ((5 + L) / 6) ** length_norm_alpha
                    complete_seqs.append(seq)
                    complete_scores.append((scores[i] / norm).item())

                keep = torch.where(~done)[0]
                if keep.numel() == 0:
                    break

                seqs = seqs[keep]
                h = h[keep]
                c = c[keep]
                encoder_out = encoder_out[keep]
                scores = scores[keep]
                cur_beam = keep.numel()

        if len(complete_seqs) > 0:
            best = int(torch.tensor(complete_scores).argmax().item())
            best_seq = complete_seqs[best]
        else:
            best_seq = seqs[scores.argmax().item()]

        return best_seq.tolist()
    def sample(self, features, max_len=25):
    """
    Udacity-required sampler.
    Args:
        features: embedded image features for a single image.
                  Common shapes:
                    - (1, num_pixels, encoder_dim)  e.g., (1, 65536, 2048)
                    - (1, 1, num_pixels, encoder_dim) if notebook did unsqueeze(1)
                    - (1, 1, embed_dim) in non-attention baselines
        max_len: maximum caption length to generate

    Returns:
        output: Python list of ints (token ids)
    """

    # --- Normalize feature shape to what beam_search expects: (B, num_pixels, encoder_dim) ---
    encoder_out = features

    # If notebook passed (1, 1, num_pixels, encoder_dim), squeeze the extra dim
    if encoder_out.dim() == 4 and encoder_out.size(1) == 1:
        encoder_out = encoder_out.squeeze(1)  # -> (1, num_pixels, encoder_dim)

    # If baseline passed (1, embed_dim), convert to (1, 1, embed_dim)
    if encoder_out.dim() == 2:
        encoder_out = encoder_out.unsqueeze(1)

    # If baseline passed (1, 1, embed_dim) keep as-is; beam_search should handle num_pixels=1

    # --- Determine start/end token ids ---
    # Preferred: stored on the model (you can set these after loading vocab)
    start_idx = getattr(self, "start_idx", None)
    end_idx = getattr(self, "end_idx", None)

    # Next best: if vocab is attached to decoder (optional)
    if (start_idx is None or end_idx is None) and hasattr(self, "vocab"):
        if start_idx is None:
            start_idx = self.vocab.word2idx.get("<start>", None)
        if end_idx is None:
            end_idx = self.vocab.word2idx.get("<end>", None)

    # Fallback: common Udacity convention (<pad>=0, <start>=1, <end>=2, <unk>=3)
    # If your vocab differs and you don't set start_idx/end_idx, output may be wrong.
    if start_idx is None:
        start_idx = 1
    if end_idx is None:
        end_idx = 2

    # --- Use existing beam_search with beam_size=1 (greedy) to satisfy Udacity sampler contract ---
    output = self.beam_search(
        encoder_out=encoder_out,
        start_idx=start_idx,
        end_idx=end_idx,
        beam_size=1,
        max_len=max_len,
        length_norm_alpha=0.0
    )

    # Ensure Python list[int]
    if torch.is_tensor(output):
        output = output.detach().cpu().tolist()

    # Some implementations return nested lists for beam outputs; flatten if needed
    if isinstance(output, list) and len(output) == 1 and isinstance(output[0], list):
        output = output[0]

    # Ensure all ints
    output = [int(x) for x in output]

    return output
