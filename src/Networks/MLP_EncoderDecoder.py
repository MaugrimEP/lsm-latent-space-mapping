from typing import List, Tuple
import torch
import torch.nn as nn

from src.Networks.EncoderDecoder import get_activation


class BaseEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        layers: List[int],
        latent_space_function: str,
    ):
        """Encoder to extract the representations
        """
        super().__init__()

        layers = [in_dim] + layers

        operations = [
            nn.Flatten(),
        ]
        for (in_dim, out_dim) in zip(layers, layers[1:]):
            operations.append(nn.Linear(in_features=in_dim, out_features=out_dim, bias=False))
            operations.append(nn.BatchNorm1d(out_dim))
            operations.append(nn.LeakyReLU())
        operations[-1] = get_activation(latent_space_function)
        self.operations = nn.Sequential(*operations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.operations(x)
        return x


class BaseDecoder(nn.Module):
    def __init__(
        self,
        layers: List[int],
        last_dim: int,
    ):
        super().__init__()

        operations = []
        for (in_dim, out_dim) in zip(layers, layers[1:]):
            operations.append(nn.Linear(in_features=in_dim, out_features=out_dim, bias=False))
            operations.append(nn.BatchNorm1d(out_dim))
            operations.append(nn.LeakyReLU())

        prediction_head = [
            nn.Linear(in_features=out_dim, out_features=last_dim, bias=True),
            nn.Sigmoid(),
        ]

        operations += prediction_head
        self.operations = nn.Sequential(*operations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.operations(x)
        return x


class BaseEncoderDecoder(nn.Module):
    def __init__(
            self,
            in_dim: int,
            layers: List[int],
            out_dim: int,
            latent_space_function: str,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.encoder = BaseEncoder(in_dim, layers, latent_space_function)
        self.decoder = BaseDecoder(layers[::-1], out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    AE = BaseEncoderDecoder(
        in_dim=68*2,
        layers=[64, 126, 256, 512, 1024],
        out_dim=68*2,
        latent_space_function='Sigmoid',
    )

    print(AE)
    from torchsummary import summary
    AE.cuda()
    summary(
        AE.encoder,
        input_size=(68*2,),
        device='cuda',
    )

    summary(
        AE.decoder,
        input_size=(1024,),
        device='cuda',
    )

    summary(
        AE,
        input_size=(68*2,),
        device='cuda',
    )
