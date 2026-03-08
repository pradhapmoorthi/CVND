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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

SPECIAL_TOKENS = {'pad':'<pad>', 'bos':'<bos>', 'eos':'<eos>', 'unk':'<unk>'}

# -------- Global feature encoder & attention --------
class EncoderCNN_Global(nn.Module):
    """
    CNN-based encoder that extracts global features from images using a pre-trained ResNet-50.
    """
    def __init__(self, cnn_name='resnet50', embed_dim=256, train_backbone=False):
        """
        Initializes the EncoderCNN_Global.
        Args:
            cnn_name (str): The name of the CNN backbone to use (currently only 'resnet50').
            embed_dim (int): The dimension to which the extracted image features will be projected.
            train_backbone (bool): If True, the CNN backbone parameters will be fine-tuned.
        """
        super().__init__()
        if cnn_name == 'resnet50':
            # Load a pre-trained ResNet-50 model
            backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
            # Remove the last fully connected layer to get feature maps
            modules = list(backbone.children())[:-1]
            self.cnn = nn.Sequential(*modules)
            feat_dim = backbone.fc.in_features # Get the dimension of features from the backbone
        else:
            raise ValueError('Unsupported CNN: ' + cnn_name)
        # Freeze backbone parameters if train_backbone is False
        for p in self.cnn.parameters():
            p.requires_grad = train_backbone
        self.fc = nn.Linear(feat_dim, embed_dim) # Fully connected layer to project features to embed_dim
        self.bn = nn.BatchNorm1d(embed_dim) # Batch normalization layer

    def forward(self, images):
        """
        Processes input images through the global CNN encoder.
        Args:
            images (torch.Tensor): A batch of input images.
        Returns:
            torch.Tensor: A tensor of extracted global image features of shape (batch_size, embed_dim).
        """
        feats = self.cnn(images).flatten(1) # Pass images through CNN and flatten features
        feats = F.relu(self.bn(self.fc(feats)))  # (B,E) # Apply FC, BN, and ReLU activation
        return feats

class BahdanauAttention_Global(nn.Module):
    """
    Bahdanau-style attention mechanism for global features.
    """
    def __init__(self, enc_dim, dec_hidden, attn_dim):
        """
        Initializes the BahdanauAttention_Global.
        Args:
            enc_dim (int): The dimension of the encoder output features.
            dec_hidden (int): The dimension of the decoder's hidden state.
            attn_dim (int): The dimension of the attention intermediate layer.
        """
        super().__init__()
        self.W = nn.Linear(enc_dim, attn_dim) # Linear layer for encoder output
        self.U = nn.Linear(dec_hidden, attn_dim) # Linear layer for decoder hidden state
        self.v = nn.Linear(attn_dim, 1) # Linear layer to compute attention scores

    def forward(self, enc_out, hidden):
        """
        Computes attention weights and a context vector.
        Args:
            enc_out (torch.Tensor): The encoder's output feature vector (batch_size, enc_dim).
            hidden (torch.Tensor): The decoder's current hidden state (batch_size, dec_hidden).
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                context (torch.Tensor): The context vector (batch_size, enc_dim).
                alpha (torch.Tensor): The attention weights (batch_size, 1).
        """
        # Compute attention scores
        score = self.v(torch.tanh(self.W(enc_out) + self.U(hidden)))  # (B,1)
        alpha = torch.softmax(score, dim=1)                            # (B,1) # Apply softmax to get attention weights
        context = alpha * enc_out                                      # (B,E) # Compute context vector as weighted sum of encoder output
        return context, alpha

class Decoder_Global(nn.Module):
    """
    Decoder for global attention, generating captions word by word.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, attn_dim, dropout=0.3):
        """
        Initializes the Decoder_Global.
        Args:
            vocab_size (int): The size of the vocabulary.
            embed_dim (int): The dimension of word embeddings.
            hidden_dim (int): The dimension of the LSTM hidden state.
            attn_dim (int): The dimension of the attention intermediate layer.
            dropout (float): Dropout probability for regularization.
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0) # Embedding layer for tokens
        self.attn  = BahdanauAttention_Global(embed_dim, hidden_dim, attn_dim) # Bahdanau attention mechanism
        self.lstm  = nn.LSTMCell(embed_dim + embed_dim, hidden_dim) # LSTM cell for sequential processing
        self.fc    = nn.Linear(hidden_dim, vocab_size) # Fully connected layer to predict next token
        self.drop  = nn.Dropout(dropout) # Dropout layer for regularization

    def forward(self, enc_feat, captions):
        """
        Performs a forward pass during training, decoding a sequence of captions.
        Args:
            enc_feat (torch.Tensor): Global features from the encoder (B, E).
            captions (torch.Tensor): Ground truth captions (B, T).
        Returns:
            torch.Tensor: Logits for each token prediction at each time step (B, T-1, vocab_size).
        """
        B,T = captions.size() # Batch size and sequence length
        # Initialize hidden and cell states of LSTM
        h = captions.new_zeros((B, self.lstm.hidden_size), dtype=torch.float).to(captions.device)
        c = captions.new_zeros((B, self.lstm.hidden_size), dtype=torch.float).to(captions.device)
        inp = self.embed(captions[:,0]) # Get embedding for the first token (BOS)
        outs=[] # List to store decoder outputs
        for t in range(1,T):
            ctx,_ = self.attn(enc_feat, h) # Compute context vector using attention
            h,c = self.lstm(torch.cat([inp, ctx], dim=1), (h,c)) # Pass concatenated input and context to LSTM
            logits = self.fc(self.drop(h)) # Predict logits for the next token
            outs.append(logits.unsqueeze(1)) # Store logits
            inp = self.embed(captions[:,t]) # Get embedding for the next token from input captions
        return torch.cat(outs, dim=1) # Concatenate all outputs

    def greedy_decode(self, enc_feat, bos_id, eos_id, max_len=20):
        """
        Generates captions using a greedy search strategy.
        Args:
            enc_feat (torch.Tensor): Global features from the encoder (B, E).
            bos_id (int): ID of the Begin-Of-Sentence token.
            eos_id (int): ID of the End-Of-Sentence token.
            max_len (int): Maximum length of the generated caption.
        Returns:
            List[List[int]]: A list of decoded token ID sequences, one for each image in the batch.
        """
        B = enc_feat.size(0) # Batch size
        # Initialize hidden and cell states
        h = enc_feat.new_zeros((B, self.lstm.hidden_size))
        c = enc_feat.new_zeros((B, self.lstm.hidden_size))
        # Start with BOS token
        x = torch.full((B,), bos_id, dtype=torch.long, device=enc_feat.device)
        emb = self.embed(x) # Get embedding for current token
        seqs=[] # List to store decoded token IDs
        for _ in range(max_len):
            ctx,_ = self.attn(enc_feat, h) # Compute context vector
            h,c = self.lstm(torch.cat([emb, ctx], dim=1), (h,c)) # Update LSTM states
            logit = self.fc(h) # Predict logits
            x = logit.argmax(-1) # Get token with highest probability (greedy choice)
            seqs.append(x) # Store predicted token
            emb = self.embed(x) # Get embedding for the next predicted token
        out=[] # List to store decoded captions (list of tokens)
        # Process each sample in the batch
        for b in range(B):
            toks=[]
            for t in seqs:
                tok = t[b].item()
                if tok==eos_id: break # Stop if EOS token is encountered
                toks.append(tok) # Add token to caption
            out.append(toks)
        return out

# -------- Spatial feature encoder & attention --------
class EncoderCNN_Spatial(nn.Module):
    """
    CNN-based encoder that extracts spatial feature maps from images for spatial attention.
    """
    def __init__(self):
        """
        Initializes the EncoderCNN_Spatial.
        """
        super().__init__()
        m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2) # Load pre-trained ResNet-50
        self.cnn = nn.Sequential(*list(m.children())[:-2])  # Bx2048x7x7 # Remove last two layers to get convolutional feature maps
        self.adapt = nn.Conv2d(2048, 512, kernel_size=1) # 1x1 convolution to reduce feature map depth

    def forward(self, images):
        """
        Processes input images through the spatial CNN encoder to extract spatial feature maps.
        Args:
            images (torch.Tensor): A batch of input images.
        Returns:
            Tuple[torch.Tensor, Tuple[int, int]]:
                seq (torch.Tensor): A sequence of spatial features, reshaped to (batch_size, H*W, channels).
                (H,W) (Tuple[int, int]): The height and width of the feature maps.
        """
        fmap = self.cnn(images)     # B,2048,7,7 # Get feature maps from CNN
        fmap = self.adapt(fmap)     # B,512,7,7 # Apply 1x1 convolution
        B,C,H,W = fmap.shape # Get batch size, channels, height, width
        seq = fmap.permute(0,2,3,1).contiguous().view(B, H*W, C)  # B,T(=49),512 # Reshape feature map for attention (B, H*W, C)
        return seq, (H,W) # Return sequence of features and original H,W dimensions

class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism that focuses on different regions of an image.
    """
    def __init__(self, feat_dim, hidden_dim):
        """
        Initializes the SpatialAttention.
        Args:
            feat_dim (int): The dimension of the spatial feature vectors.
            hidden_dim (int): The dimension of the decoder's hidden state.
        """
        super().__init__()
        self.W = nn.Linear(feat_dim, hidden_dim) # Linear layer for input features
        self.U = nn.Linear(hidden_dim, hidden_dim) # Linear layer for hidden state
        self.v = nn.Linear(hidden_dim, 1) # Linear layer to compute attention scores

    def forward(self, feats, hidden):
        """
        Computes spatial attention weights and a context vector from spatial features and decoder hidden state.
        Args:
            feats (torch.Tensor): Spatial features (B, T, C).
            hidden (torch.Tensor): The decoder's current hidden state (B, H).
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                ctx (torch.Tensor): The context vector (B, C).
                alpha (torch.Tensor): The attention weights (B, T, 1).
        """
        # feats: B,T,C; hidden: B,H
        # Compute attention scores
        score = self.v(torch.tanh(self.W(feats) + self.U(hidden).unsqueeze(1)))  # B,T,1
        alpha = torch.softmax(score, dim=1)                                       # B,T,1 # Apply softmax for attention weights
        ctx = (alpha * feats).sum(1)                                              # B,C # Compute context vector
        return ctx, alpha

class Decoder_Spatial(nn.Module):
    """
    Decoder for spatial attention, generating captions word by word.
    """
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, feat_dim=512, dropout=0.3):
        """
        Initializes the Decoder_Spatial.
        Args:
            vocab_size (int): The size of the vocabulary.
            embed_dim (int): The dimension of word embeddings.
            hidden_dim (int): The dimension of the LSTM hidden state.
            feat_dim (int): The dimension of the spatial feature vectors.
            dropout (float): Dropout probability for regularization.
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0) # Embedding layer for tokens
        self.attn  = SpatialAttention(feat_dim, hidden_dim) # Spatial attention mechanism
        self.lstm  = nn.LSTMCell(embed_dim + feat_dim, hidden_dim) # LSTM cell
        self.fc    = nn.Linear(hidden_dim, vocab_size) # Fully connected layer for token prediction
        self.drop  = nn.Dropout(dropout) # Dropout layer
        self.hidden_dim = hidden_dim

    def forward(self, feats, captions):
        """
        Performs a forward pass during training for the spatial decoder.
        Args:
            feats (torch.Tensor): Spatial features from the encoder (B, H*W, C).
            captions (torch.Tensor): Ground truth captions (B, T).
        Returns:
            torch.Tensor: Logits for each token prediction at each time step (B, T-1, vocab_size).
        """
        B,T = captions.size()
        # Initialize hidden and cell states
        h = feats.new_zeros((B, self.hidden_dim))
        c = feats.new_zeros((B, self.hidden_dim))
        inp = self.embed(captions[:,0]) # Embedding for the first token
        outs=[]
        for t in range(1,T):
            ctx,_ = self.attn(feats, h) # Compute context vector with spatial attention
            h,c = self.lstm(torch.cat([inp, ctx], dim=1), (h,c)) # Update LSTM states
            logits = self.fc(self.drop(h)) # Predict logits
            outs.append(logits.unsqueeze(1))
            inp = self.embed(captions[:,t]) # Embedding for the next token
        return torch.cat(outs, dim=1)

    def greedy_decode(self, feats, bos_id, eos_id, max_len=20):
        """
        Generates captions using a greedy search strategy with spatial attention.
        Args:
            feats (torch.Tensor): Spatial features from the encoder (B, H*W, C).
            bos_id (int): ID of the Begin-Of-Sentence token.
            eos_id (int): ID of the End-Of-Sentence token.
            max_len (int): Maximum length of the generated caption.
        Returns:
            Tuple[List[List[int]], List[torch.Tensor]]:
                List[List[int]]: A list of decoded token ID sequences.
                List[torch.Tensor]: A list of attention weights (B, H*W) for each generated token.
        """
        B = feats.size(0)
        # Initialize hidden and cell states
        h = feats.new_zeros((B, self.hidden_dim))
        c = feats.new_zeros((B, self.hidden_dim))
        # Start with BOS token
        x = torch.full((B,), bos_id, dtype=torch.long, device=feats.device)
        emb = self.embed(x)
        seqs=[]; alphas=[] # Lists to store sequences and attention weights
        for _ in range(max_len):
            ctx,alpha = self.attn(feats, h) # Compute context vector and attention weights
            h,c = self.lstm(torch.cat([emb, ctx], dim=1), (h,c)) # Update LSTM states
            logit = self.fc(h)
            x = logit.argmax(-1) # Greedy token prediction
            seqs.append(x)
            alphas.append(alpha.squeeze(-1))  # B,T # Store attention weights
            emb = self.embed(x)
        # stop at eos per sample
        out=[]
        for b in range(B):
            toks=[]
            for t in seqs:
                tok=t[b].item()
                if tok==eos_id: break
                toks.append(tok)
            out.append(toks)
        return out, alphas  # list of tokens per sample, and list of alpha tensors per step

    def beam_search(self, feats, bos_id, eos_id, beam=3, max_len=20):
        """
        Generates a single caption using beam search, which explores multiple high-probability sequences.
        Args:
            feats (torch.Tensor): Spatial features from the encoder (batch_size=1, H*W, C).
            bos_id (int): ID of the Begin-Of-Sentence token.
            eos_id (int): ID of the End-Of-Sentence token.
            beam (int): The beam width (number of sequences to keep at each step).
            max_len (int): Maximum length of the generated caption.
        Returns:
            List[int]: The best decoded token ID sequence.
        Note: This implementation currently supports a batch size of 1.
        """
        # NOTE: supports B==1 for simplicity
        assert feats.size(0) == 1, 'Beam search currently supports batch size 1.'
        # Initialize hidden and cell states
        h = feats.new_zeros((1, self.hidden_dim))
        c = feats.new_zeros((1, self.hidden_dim))
        # Initialize sequences for beam search: (tokens, logprob, h, c)
        sequences = [([bos_id], 0.0, h, c)]
        for _ in range(max_len):
            new_list = []
            for toks,score,hx,cx in sequences:
                if toks[-1] == eos_id:
                    new_list.append((toks, score, hx, cx))
                    continue
                x = torch.tensor([toks[-1]], device=feats.device) # Current token
                emb = self.embed(x) # Embedding for the current token
                ctx,_ = self.attn(feats, hx) # Compute context vector
                hx, cx = self.lstm(torch.cat([emb, ctx], dim=1), (hx, cx)) # Update LSTM states
                logits = self.fc(hx) # Predict logits
                logprobs = F.log_softmax(logits, dim=-1) # Convert logits to log probabilities
                topk = torch.topk(logprobs, beam) # Get top 'beam' probable next tokens
                for i in range(beam):
                    tok = int(topk.indices[0, i].item())
                    sc  = float(score + topk.values[0, i].item())
                    new_list.append((toks + [tok], sc, hx.clone(), cx.clone())) # Add new sequences to the list
            # Prune sequences to keep only the top 'beam' sequences based on log probability
            new_list.sort(key=lambda x: x[1], reverse=True)
            sequences = new_list[:beam]
        best = sequences[0][0] # Get the best sequence from beam search
        # strip BOS and cut at EOS
        out=[]
        for t in best[1:]:
            if t==eos_id: break # Stop at EOS token
            out.append(t)
        return out
