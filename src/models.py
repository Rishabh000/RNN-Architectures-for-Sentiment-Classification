from typing import Literal

import torch
import torch.nn as nn


ActivationLiteral = Literal["relu", "tanh", "sigmoid"]
ArchitectureLiteral = Literal["rnn", "lstm", "bilstm"]

ACTIVATION_FNS = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
}


class RecurrentSentimentClassifier(nn.Module):
    """Configurable recurrent neural network for binary sentiment classification."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.4,
        architecture: ArchitectureLiteral = "lstm",
        activation: ActivationLiteral = "relu",
    ):
        super().__init__()
        if activation not in ACTIVATION_FNS:
            raise ValueError(f"Unsupported activation '{activation}'.")
        if architecture not in {"rnn", "lstm", "bilstm"}:
            raise ValueError(f"Unsupported architecture '{architecture}'.")

        self.architecture = architecture
        self.activation = ACTIVATION_FNS[activation]
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = architecture == "bilstm"
        self.num_directions = 2 if self.bidirectional else 1

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )
        self.embedding_dropout = nn.Dropout(dropout)

        rnn_kwargs = dict(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        if architecture == "rnn":
            self.recurrent = nn.RNN(nonlinearity="tanh", **rnn_kwargs)
        else:
            self.recurrent = nn.LSTM(**rnn_kwargs)

        self.hidden_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * self.num_directions, 1)
        self.output_activation = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.embedding.weight)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)
        for name, param in self.recurrent.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Tensor of shape (batch, seq_len)
        Returns:
            probs: Tensor of shape (batch,) with probabilities in [0, 1].
        """
        embedded = self.embedding(inputs)
        embedded = self.embedding_dropout(embedded)
        outputs, hidden = self.recurrent(embedded)

        if isinstance(hidden, tuple):
            hidden_states = hidden[0]
        else:
            hidden_states = hidden

        hidden_states = hidden_states.view(self.num_layers, self.num_directions, inputs.size(0), self.hidden_size)
        last_layer_hidden = hidden_states[-1]
        last_hidden = last_layer_hidden.transpose(0, 1).reshape(inputs.size(0), -1)

        activated = self.activation(last_hidden)
        activated = self.hidden_dropout(activated)
        logits = self.classifier(activated).squeeze(-1)
        probs = self.output_activation(logits)
        return probs


def build_model(
    architecture: ArchitectureLiteral,
    activation: ActivationLiteral,
    vocab_size: int,
    embedding_dim: int = 100,
    hidden_size: int = 64,
    dropout: float = 0.4,
    num_layers: int = 2,
) -> RecurrentSentimentClassifier:
    """Factory function to create a configured sentiment classifier."""
    return RecurrentSentimentClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        architecture=architecture,
        activation=activation,
    )

