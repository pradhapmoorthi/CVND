import torch
import torch.nn as nn
import torchvision.models as models


# ============================================================
# Encoder CNN (Spatial Feature Extractor for Attention)
# ============================================================

class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size=7):
        super(EncoderCNN, self).__init__()

        self.enc_image_size = encoded_image_size

        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False

        # Keep spatial feature map
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size)
        )

    def forward(self, images):
        """
        images: (B, 3, 224, 224)
        returns: (B, 49, 2048)
        """
        features = self.resnet(images)
        features = self.adaptive_pool(features)
        features = features.permute(0, 2, 3, 1)
        features = features.view(
            features.size(0), -1, features.size(-1)
        )
        return features


# ============================================================
# Bahdanau (Additive) Attention
# ============================================================

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()

        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)
        energy = self.full_att(
            self.relu(att1 + att2)
        ).squeeze(2)

        alpha = self.softmax(energy)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)

        return context, alpha


# ============================================================
# Decoder RNN with Attention + Beam Search
# ============================================================

class DecoderRNN(nn.Module):
    def __init__(
        self,
        attention_dim,
        embed_size,
        hidden_size,
        vocab_size,
        encoder_dim=2048,
        dropout=0.5,
    ):
        super(DecoderRNN, self).__init__()

        self.encoder_dim = encoder_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.attention = Attention(
            encoder_dim, hidden_size, attention_dim
        )

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTMCell(
            embed_size + encoder_dim, hidden_size
        )

        self.init_h = nn.Linear(encoder_dim, hidden_size)
        self.init_c = nn.Linear(encoder_dim, hidden_size)

        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    # -----------------------------
    # Training (teacher forcing)
    # -----------------------------
    def forward(self, encoder_out, captions):
    """
    Training forward pass (teacher forcing).

    encoder_out: (B, 49, 2048)
    captions:    (B, seq_len)

    returns:
        outputs: (B, seq_len-1, vocab_size)
        alphas:  (B, seq_len-1, num_pixels)
    """
    batch_size = encoder_out.size(0)
    num_pixels = encoder_out.size(1)

    embeddings = self.embedding(captions[:, :-1])

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

        lstm_input = torch.cat(
            [embeddings[:, t, :], context], dim=1
        )
        h, c = self.lstm(lstm_input, (h, c))

        outputs[:, t, :] = self.fc(self.dropout(h))
        alphas[:, t, :] = alpha

    return outputs, alphas
    # -----------------------------
    # Greedy decoding
    # -----------------------------
    def sample(self, encoder_out, start_idx, end_idx, max_len=20):
        device = encoder_out.device
        h, c = self.init_hidden_state(encoder_out)

        word = torch.tensor([start_idx]).to(device)
        output_ids = []

        for _ in range(max_len):
            embed = self.embedding(word)
            context, _ = self.attention(encoder_out, h)
            h, c = self.lstm(
                torch.cat([embed, context], dim=1), (h, c)
            )
            scores = self.fc(h)
            word = scores.argmax(dim=1)

            if word.item() == end_idx:
                break

            output_ids.append(word.item())

        return output_ids

    # -----------------------------
    # Beam search decoding
    # -----------------------------
    def beam_search(
        self,
        encoder_out,
        start_idx,
        end_idx,
        beam_size=3,
        max_len=20,
        alpha=0.7,
    ):
        """
        Beam search caption generation.
        encoder_out: (1, 49, 2048)
        """
        device = encoder_out.device

        h, c = self.init_hidden_state(encoder_out)

        # Each beam: (log_prob, sequence, h, c)
        beams = [(0.0, [start_idx], h, c)]
        completed = []

        for _ in range(max_len):
            new_beams = []

            for log_prob, seq, h, c in beams:
                if seq[-1] == end_idx:
                    completed.append((log_prob, seq))
                    continue

                word = torch.tensor([seq[-1]]).to(device)
                embed = self.embedding(word)
                context, _ = self.attention(encoder_out, h)

                h_new, c_new = self.lstm(
                    torch.cat([embed, context], dim=1), (h, c)
                )

                scores = self.fc(h_new)
                log_probs = torch.log_softmax(scores, dim=1)
                top_log_probs, top_words = log_probs.topk(beam_size, dim=1)

                for i in range(beam_size):
                    new_seq = seq + [top_words[0, i].item()]
                    new_log_prob = log_prob + top_log_probs[0, i].item()

                    new_beams.append(
                        (new_log_prob, new_seq, h_new, c_new)
                    )

            beams = sorted(
                new_beams,
                key=lambda x: x[0] / (len(x[1]) ** alpha),
                reverse=True,
            )[:beam_size]

            if all(seq[-1] == end_idx for _, seq, _, _ in beams):
                break

        completed.extend((log_prob, seq) for log_prob, seq, _, _ in beams)

        completed.sort(
            key=lambda x: x[0] / (len(x[1]) ** alpha),
            reverse=True,
        )

        best_seq = completed[0][1]

        # Remove <start> and <end>
        if best_seq[0] == start_idx:
            best_seq = best_seq[1:]
        if best_seq and best_seq[-1] == end_idx:
            best_seq = best_seq[:-1]

        return best_seq
