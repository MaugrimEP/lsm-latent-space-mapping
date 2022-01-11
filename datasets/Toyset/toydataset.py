from __future__ import annotations

import csv
import random
from typing import Tuple, Callable, Set, Dict
from PIL import Image
import numpy as np
import cv2
from typing import List
from numpy import genfromtxt
from torch.utils.data import Dataset

from Utils.utils import mask2mode

mode2int = {
	'NONE': -1,
	'FULL': 0,
	'XY'  : 1,
	'X'   : 2,
	'Y'   : 3,
}

# region utils dataset function
def randomize_tiles_shuffle_within(a, M, N):
	# Elements in each block are randomized and that same randomized order is maintained in all blocks.
	# M,n are the height and width of the blocks
	m, n = a.shape
	b = a.reshape(m // M, M, n // N, N).swapaxes(1, 2).reshape(-1, M * N)
	np.random.shuffle(b.T)
	return b.reshape(m // M, n // N, M, N).swapaxes(1, 2).reshape(a.shape)


def randomize_tiles_shuffle_blocks(a, M, N):
	# Blocks are randomized w.r.t each other, while keeping the order within each block same as in the original array.
	# M,n are the height and width of the blocks
	m, n = a.shape
	b = a.reshape(m // M, M, n // N, N).swapaxes(1, 2).reshape(-1, M * N)
	np.random.shuffle(b)
	return b.reshape(m // M, n // N, M, N).swapaxes(1, 2).reshape(a.shape)


def normalized_domain(resolution: int) -> List[float]:
	return [i / resolution for i in range(0, resolution + 1)]


def get_from_normalized(frac: float, value_range: List) -> float:
	return value_range[0] + frac * (value_range[1] - value_range[0])


def add_gaussian_noise(image, mean, var):
	row, col = image.shape
	sigma = var ** 0.5
	gauss = np.random.normal(mean, sigma, (row, col))
	gauss = gauss.reshape([row, col])
	noisy = image + gauss
	return noisy


def get_image(path: str) -> np.ndarray:
	return np.array(Image.open(path))
# endregion

class ToyDataset_old(Dataset):
	FOREGROUND_circle = "D17.gif"
	BACKGROUND_circle = "D77.gif"
	FOREGROUND_square = "D23.gif"
	BACKGROUND_square = "D49.gif"
	FIXED_BASE = "datasets/Toyset/fixed_toyset"
	DATAPATH = "data/"

	# domain 1: normalized x,y  position of the circle
	# domain 2: normalized x,y position of the square
	# domain 3: _

	# domain 1: normalized inner diameter for the circle
	# domain 2: normalized half side length for the square
	# domain 3: _

	# domain 1: normalized circle line thickness
	# domain 2: normalized square line thickness
	# domain 3: _

	# Image generation setting
	SPAWN_RANGE = [40, 128 - 40]
	INNER_SIZE_RANGE = [10, 30]
	THICKNESS_RANGE = [5, 30]

	def __init__(
			self,
			image_size: List[int],
			swap_segmentation: bool,
	        from_fixed: bool,

			modes: Set[str],
			return_mode: bool,
			proportion_xy: float = None,
			proportion_x : float = None,
			proportion_y : float = None,
			mode2mask: Callable = None,
	):
		self.image_size    = image_size
		self.proportion_xy = proportion_xy
		self.proportion_x  = proportion_x
		self.proportion_y  = proportion_y
		self.return_mode   = return_mode
		self.mode2mask     = mode2mask if mode2mask is not None else lambda x: x
		int_target_modes = set(mode2int[mode] for mode in modes)
		self.modes = modes

		self.foreground_circle = get_image(ToyDataset_old.DATAPATH + ToyDataset_old.FOREGROUND_circle)
		self.background_circle = get_image(ToyDataset_old.DATAPATH + ToyDataset_old.BACKGROUND_circle)

		self.foreground_square = get_image(ToyDataset_old.DATAPATH + ToyDataset_old.FOREGROUND_square)
		self.background_square = get_image(ToyDataset_old.DATAPATH + ToyDataset_old.BACKGROUND_square)

		self.swap_segmentation = swap_segmentation

		self.from_fixed = from_fixed
		if self.from_fixed:
			data = genfromtxt(f"{ToyDataset_old.FIXED_BASE}/params.csv", delimiter=',')

			dataset_length = len(data)
			nb_x  = int(dataset_length * proportion_x)
			nb_y  = int(dataset_length * proportion_y)
			nb_xy = dataset_length - nb_x - nb_y

			# NONE = we do not care about the mode, just treat them as XY
			if 'NONE' in modes:
				ds_mode = np.array([1]*dataset_length).reshape([-1, 1])
			else:
				ds_mode = np.array([1] * nb_xy + [2] * nb_x + [3] * nb_y).reshape([-1, 1])

			data = np.concatenate([data, ds_mode], axis=-1)

			# if we are not using everything, we need to perform a filtering
			if 'NONE' not in modes and 'FULL' not in modes:
				data = self._filter_mode(data, target_modes=int_target_modes)
			self.data = data

	def __len__(self) -> int:
		return self.data.shape[0]

	def _filter_mode(self, data: np.ndarray, target_modes: Set[int]):
		data_filter_mask = []
		for *other_data, curr_mode in data:
			data_filter_mask.append(curr_mode in target_modes)
		return data[data_filter_mask]

	def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray]:
		if self.from_fixed:
			sample_param = self.data[index, :-1]
			mode         = self.data[index,  -1]
		else:
			sample_param = self._generate_random_params()
			mode         = random.choices(
				population=[1, 2, 3],
				weights=[self.proportion_xy, self.proportion_x, self.proportion_y],
				k=1
			)[0]

		circle, square, circle_segmentation = self._get_samples(*sample_param)

		circle = np.expand_dims(circle, axis=0).astype(np.float32)
		square = np.expand_dims(square, axis=0).astype(np.float32)
		circle_segmentation = np.expand_dims(circle_segmentation, axis=0).astype(np.float32)

		if self.return_mode:
			return circle, circle_segmentation, self.mode2mask(int(mode))
		else:
			return circle, circle_segmentation

	def _get_background(self, source_image: np.ndarray, target_size: List[int], x_shift: float,
	                    y_shift: float) -> np.ndarray:
		"""
		Cut a part of the background source_image and return it, random
		:param source_image: the background image
		:param target_size: size to sample
		:param x_shift: \in [0,1]
		:param y_shift: \in [0,1]
		:return:
		"""
		width, height = source_image.shape
		x_marge = width - target_size[0]
		y_marge = height - target_size[1]
		if x_marge < 0 or y_marge < 0:
			raise Exception("Target cropping bigger than source source_image")

		random_start_x = int(x_shift * x_marge)
		random_start_y = int(y_shift * y_marge)

		cutted_image = source_image[random_start_x:random_start_x + target_size[0],
		               random_start_y:random_start_y + target_size[1]]

		return cutted_image

	def _image_prepross(self, source_image: np.ndarray, x_shift: float, y_shift: float) -> np.ndarray:
		r = source_image
		r = r / 255
		r = self._get_background(r, self.image_size, x_shift, y_shift)
		return r

	def _get_circle_segmentation(self, x: float, y: float, z: float, w: float) -> np.ndarray:
		circle_center = (int(get_from_normalized(x, ToyDataset_old.SPAWN_RANGE)),
		                 int(get_from_normalized(y, ToyDataset_old.SPAWN_RANGE)))
		radii = int(get_from_normalized(z, ToyDataset_old.INNER_SIZE_RANGE))
		thickness = int(get_from_normalized(w, ToyDataset_old.THICKNESS_RANGE))

		image = np.zeros(self.image_size)
		image = cv2.circle(image, circle_center, radii + thickness // 2, 1, thickness)
		return image

	def _get_square_segmentation(self, x: float, y: float, z: float, w: float) -> np.ndarray:
		square_center = (int(get_from_normalized(x, ToyDataset_old.SPAWN_RANGE)),
		                 int(get_from_normalized(y, ToyDataset_old.SPAWN_RANGE)))
		half_side_size = int(get_from_normalized(z, ToyDataset_old.INNER_SIZE_RANGE))
		thickness = int(get_from_normalized(w, ToyDataset_old.THICKNESS_RANGE))

		lower_lower = (
			square_center[0] - half_side_size - thickness // 2,
			square_center[1] - half_side_size - thickness // 2,
		)

		upper_upper = (
			square_center[0] + half_side_size + thickness // 2,
			square_center[1] + half_side_size + thickness // 2,
		)

		mask = np.zeros(self.image_size)
		mask = cv2.rectangle(mask, lower_lower, upper_upper, 1, thickness)

		return mask

	def _get_samples(self, x: float, y: float, z: float, w: float, x_shift: float, y_shift: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

		circle_segmentation = self._get_circle_segmentation(x, y, z, w)
		square_segmentation = self._get_square_segmentation(x, y, z, w)

		circle = circle_segmentation * self._image_prepross(self.foreground_circle, x_shift, y_shift) \
		         + (1 - circle_segmentation) * self._image_prepross(self.background_circle, x_shift, y_shift)
		square = square_segmentation * self._image_prepross(self.foreground_square, x_shift, y_shift) \
		         + (1 - square_segmentation) * self._image_prepross(self.background_square, x_shift, y_shift)

		if self.swap_segmentation:
			circle_segmentation = self._get_circle_segmentation(w, z, y,
			                                                    x)  # for the d3 domain, we will spam y and the demi-width z

		return circle, square, circle_segmentation

	def _generated_fixed_samples(self, n: int):
		"""
		Generate a fixed dataset
		:param n: dataset_size
		:return:
		"""
		def _save_image(img, path: str):
			img = img * 255
			img = Image.fromarray(img)
			img = img.convert('RGB')
			img.save(path)

		csv_f = open(f"{ToyDataset_old.FIXED_BASE}/params.csv", "w", newline='')
		writer = csv.writer(csv_f)
		params, circles, squares, circle_segmentations = self.get_batch_with_params(n)
		for i in range(n):
			writer.writerow(params[i])
			#_save_image(circles[i], f"{ToyDataset.FIXED_BASE}/{i}_circle.png")
			#_save_image(squares[i], f"{ToyDataset.FIXED_BASE}/{i}_square.png")
			#_save_image(circle_segmentations[i], f"{ToyDataset.FIXED_BASE}/{i}_circle_segmentation.png")
		csv_f.close()

	def _generate_random_params(self) -> List[float]:
		x = random.random()
		y = random.random()
		z = random.random()
		w = random.random()
		x_shift, y_shift = random.random(), random.random()

		sample_param = [x, y, z, w, x_shift, y_shift]
		return sample_param

	def get_batch_with_params(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		params = []
		circles = []
		squares = []
		segmentations = []
		for i in range(batch_size):
			sample_param = self._generate_random_params()
			d1, d2, d3 = self._get_samples(*sample_param)

			params.append(sample_param)
			circles.append(d1)
			squares.append(d2)
			segmentations.append(d3)

		circles = np.array(circles)
		params = np.array(params)
		squares = np.array(squares)
		segmentations = np.array(segmentations)
		return params, circles, squares, segmentations

	def get_stats(self) -> Tuple[Dict[int:int], List[int]]:
		"""
		return the number of elements which contains XY, just X, or just Y
		:return:
		"""
		modes_stats = {
			1: 0,  # cpt XY
			2: 0,  # cpt X only
			3: 0,  # cpt Y only
		}
		classes = []
		for *data, mode in self:
			mode = mask2mode(mode)
			modes_stats[mode] += 1
			classes.append(mode)

		return modes_stats, classes


if __name__ == '__main__':
	ToyDataset_old.FIXED_BASE = "fixed_toyset"
	ToyDataset_old.DATAPATH = "../../data/"

	dataset = ToyDataset_old(
		image_size=[128, 128],
		swap_segmentation=False,
		from_fixed=True,
		modes={'FULL'},
		return_mode=True,
		proportion_xy=0.2,
		proportion_x=0.3,
		proportion_y=0.5,
		mode2mask=None,
	)
	# {1: 300, 2: 450, 3: 750}
	print(dataset)
	print(len(dataset))
	stats = {1:0, 2:0, 3:0}
	for *data, mode in dataset:
		stats[mode]+= 1
	print(stats)
	pass
