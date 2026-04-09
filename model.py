import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-50 and replace top classifier layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # Freeze parameters so that we don't backpropagate through them
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1] # Delete the last FC layer (AvgPool and FC)
        self.resnet = nn.Sequential(*modules)
        # The output of resnet.children()[:-1] will be (batch_size, 2048, 1, 1)
        # We need to flatten it and then pass through a linear layer to embed_size
        self.linear = nn.Linear(resnet.fc.in_features, embed_size) # resnet.fc.in_features is 2048 for resnet50
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        features = self.resnet(images)
        features = features.view(features.size(0), -1) # Flatten to (batch_size, 2048)
        features = self.bn(self.linear(features)) # Apply linear layer and batch norm
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers # Store num_layers for LSTM state initialization

    def forward(self, features, captions):
        # Embed the captions
        embeddings = self.embed(captions)

        # Concatenate image features with embedded captions
        # features: (batch_size, embed_size) -> (batch_size, 1, embed_size)
        # embeddings: (batch_size, caption_length, embed_size)
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)

        # Pass through LSTM
        hiddens, _ = self.lstm(inputs)

        # Linear layer to get vocabulary scores
        outputs = self.linear(hiddens)
        return outputs


    def sample(self, features, states=None, max_len=20):
        """
        Accepts pre-computed CNN features (batch_size=1, embed_size) and generates captions.
        Uses greedy search to generate a caption of maximum length `max_len`,
        stopping when the <end> token is generated.
        """
        # Initialize hidden and cell states for the LSTM
        hidden = (torch.zeros(self.num_layers, 1, self.hidden_size).to(features.device),
                  torch.zeros(self.num_layers, 1, self.hidden_size).to(features.device))

        outputs = []
        # The first input to the LSTM will be the image features
        # features: (1, embed_size) -> unsqueeze for sequence dim (1, 1, embed_size)
        inputs = features.unsqueeze(1)

        for i in range(max_len):
            hiddens, hidden = self.lstm(inputs, hidden) # hiddens: (1, 1, hidden_size)
            scores = self.linear(hiddens.squeeze(1)) # scores: (1, vocab_size)
            predicted_id = scores.argmax(1) # predicted_id: (1)

            outputs.append(predicted_id.item())

            # Stop if the <end> token is predicted. Assuming <end> token is 1 in vocab.idx2word.
            if predicted_id.item() == 1:
                break

            # Prepare input for next step: embed the predicted word
            inputs = self.embed(predicted_id).unsqueeze(1) # inputs: (1, 1, embed_size)

        return outputs
