import torch.nn as nn
from torch import Tensor


class CNNClassifier(nn.Module):
    """1-D CNN → GlobalAvgPool → FC binary classifier for sequential stock data.

    num_layers conv blocks with increasing channels capture patterns at different scales,
    global average pooling collapses the time dimension before the FC head.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        # channel sizes: hidden_size//2 → hidden_size//2 → ... → hidden_size
        channel_in = hidden_size // 2
        channel_out = hidden_size

        layers: list[nn.Module] = []
        in_ch = n_features
        for i in range(num_layers):
            out_ch = channel_out if i == num_layers - 1 else channel_in
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch

        self.conv_blocks = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.Linear(channel_out, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, seq_len, n_features)
        x = x.permute(0, 2, 1)  # → (batch, n_features, seq_len)
        x = self.conv_blocks(x)  # → (batch, channels[1], seq_len)
        x = x.mean(dim=-1)  # global average pool → (batch, channels[1])
        return self.head(x).squeeze(-1)  # (batch,) raw logits


def build_model(
    n_features: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
) -> CNNClassifier:
    model = CNNClassifier(n_features, hidden_size, num_layers, dropout)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CNNClassifier — trainable params: {n_params:,}")
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
