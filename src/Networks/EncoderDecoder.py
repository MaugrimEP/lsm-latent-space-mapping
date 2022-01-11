from typing import List, Tuple
import torch
import torch.nn as nn

def get_activation(name: str):
    return {
        'ReLU': nn.ReLU(),
        'LeakyReLU': nn.LeakyReLU(),
        'Sigmoid': nn.Sigmoid(),
    }[name]


class BaseEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        layers: List[Tuple[int, int]],
        latent_space_function: str,
    ):
        """
        {'ReLU': nn.ReLU(), 'LeakyReLU': nn.LeakyReLU(), 'Sigmoid': nn.Sigmoid()}
        :param in_channels:
        :param layers:
        :param latent_space_function: {'ReLU': nn.ReLU(), 'LeakyReLU': nn.LeakyReLU(), 'Sigmoid': nn.Sigmoid()}
        """
        super().__init__()

        layers = [(in_channels, 1)] + layers

        operations = []
        for (in_channel, _), (curr_channel, nb_conv) in zip(layers, layers[1:]):
            down_block = []
            for i in range(nb_conv):
                in_channel = in_channel if i == 0 else curr_channel

                down_block.append(nn.Conv2d(in_channel, curr_channel, kernel_size=(3, 3), padding=1, bias=False))
                down_block.append(nn.BatchNorm2d(curr_channel))
                down_block.append(nn.LeakyReLU())
            down_block.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
            operations.append(down_block)

        operations[-1][-2] = get_activation(latent_space_function)
        operations = [nn.Sequential(*down_block_i) for down_block_i in operations]
        self.operations = nn.Sequential(*operations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.operations(x)
        return x


class BaseDecoder(nn.Module):
    def __init__(
        self,
        layers: List[Tuple[int, int]],
        out_dim: int,
    ):
        super().__init__()

        operations = []
        for (in_channel, _), (curr_channel, nb_conv) in zip(layers, layers[1:]):
            up_block = []
            up_block.append(nn.ConvTranspose2d(in_channel, curr_channel, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),dilation=1, output_padding=(1, 1)))
            for i in range(nb_conv):
                up_block.append(nn.Conv2d(curr_channel, curr_channel, kernel_size=(3, 3), padding='same', bias=False))
                up_block.append(nn.BatchNorm2d(curr_channel))
                up_block.append(nn.LeakyReLU())
            operations.append(nn.Sequential(*up_block))

        channel_out_size = layers[-1][0]
        prediction_head = [
            nn.ConvTranspose2d(channel_out_size, channel_out_size, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=1,output_padding=(1, 1)),
            nn.Conv2d(channel_out_size, out_dim, kernel_size=(1, 1), padding='same'),
        ]
        operations += prediction_head
        self.operations = nn.Sequential(*operations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.operations(x)
        return x


class BaseEncoderDecoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            layers: List[Tuple[int, int]],
            out_dim: int,
            latent_space_function: str,
    ):
        super().__init__()
        self.in_dim = in_channels
        self.out_dim = out_dim
        self.encoder = BaseEncoder(in_channels, layers, latent_space_function)
        self.decoder = BaseDecoder(layers[::-1], out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    AE = BaseEncoderDecoder(
        in_channels=5,
        layers=[(64, 2), (126, 2), (256, 3), (512, 3), (1024, 1)],
        out_dim=5,
        latent_space_function='LeakyReLU',
    )

    print(AE)
    from torchsummary import summary
    AE.cuda()
    summary(
        AE.encoder,
        input_size=(5, 128, 128),
        device='cuda',
    )

    summary(
        AE.decoder,
        input_size=(1024, 4, 4),
        device='cuda',
    )

    summary(
        AE,
        input_size=(5, 128, 128),
        device='cuda',
    )
