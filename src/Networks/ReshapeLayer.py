from typing import List

import torch
import torch.nn as nn

class ReshapeLayer(nn.Module):
	def __init__(self, s: List[int], keep_bs: bool = False):
		super(ReshapeLayer, self).__init__()
		self.s = s
		self.keep_bs = keep_bs

	def forward(self, x):
		if self.keep_bs:
			shapes = x.shape
			bs = shapes[0]
			return x.reshape([bs] + self.s)
		else:
			return x.reshape(self.s)
