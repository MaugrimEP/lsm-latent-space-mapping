from __future__ import annotations

import torch


def inter_ocular_dist(data: torch.Tensor):
	"""
	Calculate the Euclidean  distance between the outer corners of the
		eyes.
	"""
	# batchsize, 68, 2
	indice_a = 37
	indice_b = 46

	data = data.reshape([-1, 68, 2])

	# batchsize, 1, 2
	pts_a = data[:, indice_a, :]
	pts_b = data[:, indice_b, :]

	distance = (pts_a - pts_b).square().sum(axis=-1).sqrt()

	return distance
