from torch import nn

class CIFARBaseline(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            # Layer 1: 3*32*32 -> 32*32*32
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),

            # Layer 2: 32*32*32 -> 64*16*16
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 3: 64*16*16 -> 128*8*8
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.network = nn.Sequential(
            nn.Flatten(),

            # Layer 1: 8192 -> 256
            nn.Linear(
                in_features=128*8*8,
                out_features=256
            ),
            nn.ReLU(),

            # Layer 2: 256 -> 128
            nn.Linear(
                in_features=256,
                out_features=128
            ),
            nn.ReLU(),

            # Layer 3: 128 -> 10
            nn.Linear(
                in_features=128,
                out_features=10
            )
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.network(x)

        return x