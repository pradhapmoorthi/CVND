import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EncoderCNN(nn.Module):
    """
    CNN Encoder that outputs spatial features for attention.
    Output: encoder_out of shape (B, num_pixels, encoder_dim)
    where num_pixels = H*W (e.g., 7*7=49) and encoder_dim=2048 for ResNet-50.
    """
    def __init__(self, encoded_image_size=7, backbone="resnet50", train_cnn=False):
        super().__init__()

        self.encoded_image_size = encoded_image_size

        if backbone == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            encoder_dim = 2048
        elif backbone == "resnet101":
            resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            encoder_dim = 2048
        else:
            raise ValueError("Supported backbones: resnet50, resnet101")

        # Remove avgpool and fc; keep conv layers up to layer4
        modules = list(resnet.children())[:-2]  # till last conv block
        self.cnn = nn.Sequential(*modules)

        # Adaptive pool to fixed spatial size (e.g., 7x7)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.encoder_dim = encoder_dim

        # Freeze CNN if desired
        self.set_trainable(train_cnn)

    def set_trainable(self, train_cnn: bool):
        for p in self.cnn.parameters():
            p.requires_grad = train_cnn

    def forward(self, images):
        """
        images: (B, 3, H, W)
        returns: (B, num_pixels, encoder_dim)
        """
        x = self.cnn(images)  # (B, encoder_dim, H', W')
        x = self.adaptive_pool(x)  # (B, encoder_dim, encoded_image_size, encoded_image_size)
        B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        x = x.view(B, H * W, C)  # (B, num_pixels, encoder_dim)
        return x


class Attention(nn.Module):
    """
    Additive (Bahdanau-style) attention.
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
        att1 = self.encoder_att(encoder_out)                 # (B, num_pixels, att_dim)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1) # (B, 1, att_dim)
        e = self.full_att(torch.tanh(att1 + att2)).squeeze(2) # (B, num_pixels)
        alpha = F.softmax(e, dim=1)                          # (B, num_pixels)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1) # (B, encoder_dim)
        return context, alpha


class DecoderRNN(nn.Module):
    """
    Attention-based decoder with LSTMCell.
    Training: teacher forcing with packed sequences.
    Inference: greedy sample() or beam_search().
    """
    def __init__(
        self,
        embed_size,
        vocab_size,
        encoder_dim=2048,
        attention_dim=512,
        decoder_dim=512,
        dropout=0.5
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)

        # LSTMCell input: [word_embed + context]
        self.decode_step = nn.LSTMCell(embed_size + encoder_dim, decoder_dim, bias=True)

        # Initialize hidden state from mean of encoder features
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # gating scalar
        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Linear(decoder_dim, vocab_size)

        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """
        encoder_out: (B, num_pixels, encoder_dim)
        """
        mean_encoder = encoder_out.mean(dim=1)  # (B, encoder_dim)
        h = self.init_h(mean_encoder)           # (B, decoder_dim)
        c = self.init_c(mean_encoder)           # (B, decoder_dim)
        return h, c

    def forward(self, encoder_out, captions, lengths):
        """
        encoder_out: (B, num_pixels, encoder_dim)
        captions: (B, max_len)  with <start> token at captions[:,0]
        lengths: list or tensor of lengths (includes <start> and <end> typically)

        returns:
          predictions: (sum(lengths-1), vocab_size) after packing
          alphas: attention weights per time step (B, max_len-1, num_pixels)
        """
        device = encoder_out.device
        B = encoder_out.size(0)
        num_pixels = encoder_out.size(1)

        # Sort by lengths (required by pack_padded_sequence if enforce_sorted=True)
        lengths = torch.as_tensor(lengths, device=device)
        lengths_sorted, sort_ind = lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        captions = captions[sort_ind]

        # Embed words
        embeddings = self.embedding(captions)  # (B, max_len, embed_size)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)

        # We won't predict at t=0 (<start>), so decode lengths-1 steps
        decode_lengths = (lengths_sorted - 1).tolist()
        max_decode = max(decode_lengths)

        predictions = torch.zeros(B, max_decode, self.vocab_size, device=device)
        alphas = torch.zeros(B, max_decode, num_pixels, device=device)

        for t in range(max_decode):
            batch_size_t = sum([l > t for l in decode_lengths])

            context, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # (batch_size_t, encoder_dim)
            context = gate * context

            lstm_input = torch.cat([embeddings[:batch_size_t, t, :], context], dim=1)
            h_t, c_t = self.decode_step(lstm_input, (h[:batch_size_t], c[:batch_size_t]))

            preds = self.fc(self.dropout(h_t))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

            h[:batch_size_t] = h_t
            c[:batch_size_t] = c_t

        # Pack predictions to ignore pads
        packed_preds = nn.utils.rnn.pack_padded_sequence(
            predictions, decode_lengths, batch_first=True, enforce_sorted=True
        ).data

        # Also need packed targets outside (in training loop)
        return packed_preds, alphas, sort_ind, decode_lengths

    @torch.no_grad()
    def sample(self, encoder_out, start_idx, end_idx, max_len=20):
        """
        Greedy decoding (baseline).
        encoder_out: (1, num_pixels, encoder_dim)
        returns: list of token ids
        """
        device = encoder_out.device
        h, c = self.init_hidden_state(encoder_out)

        word = torch.tensor([start_idx], device=device, dtype=torch.long)
        seq = [start_idx]

        for _ in range(max_len):
            emb = self.embedding(word)  # (1, embed_size)
            context, alpha = self.attention(encoder_out, h)
            gate = self.sigmoid(self.f_beta(h))
            context = gate * context

            h, c = self.decode_step(torch.cat([emb.squeeze(1), context], dim=1), (h, c))
            logits = self.fc(h)
            word = torch.argmax(logits, dim=1)  # (1,)
            w = word.item()
            seq.append(w)
            if w == end_idx:
                break
        return seq

    @torch.no_grad()
    def beam_search(self, encoder_out, start_idx, end_idx, beam_size=5, max_len=20, length_norm_alpha=0.7):
        """
        Beam search decoding for better captions.

        encoder_out: (1, num_pixels, encoder_dim)
        returns: best sequence (list of token ids)
        """
        device = encoder_out.device
        k = beam_size

        # Expand encoder_out to k
        encoder_out = encoder_out.expand(k, encoder_out.size(1), encoder_out.size(2))  # (k, num_pixels, enc_dim)

        # Initialize hidden/cell
        h, c = self.init_hidden_state(encoder_out)  # (k, dec_dim)

        # Sequences and scores
        seqs = torch.full((k, 1), start_idx, dtype=torch.long, device=device)  # (k, 1)
        topk_scores = torch.zeros(k, 1, device=device)  # (k, 1), log-prob

        complete_seqs = []
        complete_scores = []

        for step in range(max_len):
            last_words = seqs[:, -1]  # (k,)
            emb = self.embedding(last_words)  # (k, embed)

            context, alpha = self.attention(encoder_out, h)
            gate = self.sigmoid(self.f_beta(h))
            context = gate * context

            h, c = self.decode_step(torch.cat([emb, context], dim=1), (h, c))
            logits = self.fc(h)  # (k, vocab)
            log_probs = F.log_softmax(logits, dim=1)  # (k, vocab)

            # Add previous scores
            scores = topk_scores + log_probs  # (k, vocab)

            # Flatten and pick top k
            if step == 0:
                topk_scores_, topk_words = scores[0].topk(k, dim=0)  # from first beam only
                prev_beam_inds = torch.zeros(k, dtype=torch.long, device=device)
            else:
                topk_scores_, topk_words = scores.view(-1).topk(k, dim=0)
                prev_beam_inds = topk_words // self.vocab_size
            next_word_inds = topk_words % self.vocab_size

            # Update sequences
            seqs = torch.cat([seqs[prev_beam_inds], next_word_inds.unsqueeze(1)], dim=1)
            h = h[prev_beam_inds]
            c = c[prev_beam_inds]
            encoder_out = encoder_out[prev_beam_inds]
            topk_scores = topk_scores_.unsqueeze(1)

            # Check completed
            incomplete_inds = []
            for i in range(seqs.size(0)):
                if seqs[i, -1].item() == end_idx:
                    complete_seqs.append(seqs[i].tolist())
                    # length normalization (optional)
                    length = seqs[i].size(0)
                    norm = ((5 + length) / 6) ** length_norm_alpha
                    complete_scores.append((topk_scores[i].item()) / norm)
                else:
                    incomplete_inds.append(i)

            if len(incomplete_inds) == 0:
                break

            # Keep only incomplete beams
            seqs = seqs[incomplete_inds]
            h = h[incomplete_inds]
            c = c[incomplete_inds]
            encoder_out = encoder_out[incomplete_inds]
            topk_scores = topk_scores[incomplete_inds]
            k = seqs.size(0)

        if len(complete_seqs) == 0:
            # If nothing ended, take best ongoing
            best = seqs[0].tolist()
            return best

        best_idx = int(torch.tensor(complete_scores).argmax().item())
        return complete_seqs[best_idx]
