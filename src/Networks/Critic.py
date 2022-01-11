# https://arxiv.org/pdf/1409.7495.pdf
import functools
from typing import List

import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

from src.Networks.CycleGAN.networks import _get_activation
from src.Networks.RGL import RevGradLayer


class Critic(nn.Module):
	def __init__(
			self,
			need_GRL: bool,
			layers: List[int],
			output_function: str,
	):
		super(Critic, self).__init__()

		output_function = _get_activation(output_function)

		net = [
			RevGradLayer() if need_GRL else nn.Identity(),
			nn.Flatten(),
		]
		for layer_dim_in, layer_dim_out in zip(layers, layers[1:]):
			net += [
				nn.Linear(layer_dim_in, layer_dim_out, bias=True),
				nn.LeakyReLU(),
			]
		# remove last activation function
		net = net[:-1]
		net.append(output_function)

		self.layers = nn.Sequential(*net)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.layers(x)
