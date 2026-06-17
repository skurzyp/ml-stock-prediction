import torch.nn as nn
from torch import Tensor


class LSTMClassifier(nn.Module):
    """Stacked LSTM → FC binary classifier for sequential stock data."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, seq_len, n_features)
        _, (h_n, _) = self.lstm(x)  # h_n: (num_layers, batch, hidden_size)
        x = h_n[-1]  # last layer hidden state: (batch, hidden_size)
        return self.head(x).squeeze(-1)  # (batch,) raw logits


def build_model(
    n_features: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
) -> LSTMClassifier:
    model = LSTMClassifier(n_features, hidden_size, num_layers, dropout)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LSTMClassifier — trainable params: {n_params:,}")
    return model


if __name__ == "__main__":
    import torch
    from . import config

    model = build_model(
        n_features=len(config.FEATURE_COLS),
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
    )
    dummy = torch.randn(config.BATCH_SIZE, config.SEQ_LEN, len(config.FEATURE_COLS))
    out = model(dummy)
    print(f"Input:  {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}")
