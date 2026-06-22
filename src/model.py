# src/model.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()

        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Remove final classification layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        # Freeze ResNet parameters
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(512, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, images):
        """
        images shape:
            [batch_size, 3, 224, 224]

        output shape:
            [batch_size, embed_size]
        """
        with torch.no_grad():
            features = self.resnet(images)

        features = features.view(features.size(0), -1)
        features = self.fc(features)
        features = self.relu(features)
        features = self.dropout(features)

        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.init_h = nn.Linear(embed_size, hidden_size)
        self.init_c = nn.Linear(embed_size, hidden_size)

        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def init_hidden_state(self, image_features):
        """
        Uses image feature vector to initialize LSTM hidden state.
        """
        h = self.init_h(image_features)
        c = self.init_c(image_features)

        h = h.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c = c.unsqueeze(0).repeat(self.num_layers, 1, 1)

        return h, c

    def forward(self, image_features, captions):
        """
        image_features:
            [batch_size, embed_size]

        captions:
            [batch_size, caption_length]

        returns:
            [batch_size, caption_length, vocab_size]
        """
        embeddings = self.dropout(self.embedding(captions))

        hidden = self.init_hidden_state(image_features)

        outputs, _ = self.lstm(embeddings, hidden)

        predictions = self.fc(outputs)

        return predictions

    def generate_caption(self, image_features, vocab, max_length=30):
        """
        Greedy decoding.

        At each step:
            choose the word with the highest probability.
        """
        result = []

        hidden = self.init_hidden_state(image_features)

        current_token = torch.tensor(
            [[vocab.stoi[vocab.start_token]]],
            device=image_features.device
        )

        for _ in range(max_length):
            embedding = self.embedding(current_token)

            output, hidden = self.lstm(embedding, hidden)

            prediction = self.fc(output.squeeze(1))

            predicted_idx = prediction.argmax(dim=1).item()

            predicted_word = vocab.itos[predicted_idx]

            if predicted_word == vocab.end_token:
                break

            result.append(predicted_word)

            current_token = torch.tensor(
                [[predicted_idx]],
                device=image_features.device
            )

        return " ".join(result)


class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout):
        super().__init__()

        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(
            embed_size=embed_size,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_layers=num_layers,
            dropout=dropout
        )

    def forward(self, images, captions):
        image_features = self.encoder(images)
        predictions = self.decoder(image_features, captions)
        return predictions