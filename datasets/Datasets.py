from __future__ import annotations

import os
import random
from typing import Tuple, List

import wandb
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset

_modes = {
	(0, 'x', (1, 0, 0), torch.Tensor((1, 0, 0))),
	(1, 'y', (0, 1, 0), torch.Tensor((0, 1, 0))),
	(2, 'xy', (1, 1, 1), torch.Tensor((1, 1, 1))),
}
_g_modes = dict()
for values in _modes:
	for v in values[:-1]:
		_g_modes[v] = values


class Modes:
	modes = _g_modes

	@staticmethod
	def get(key):
		if isinstance(key, List):
			key = tuple(key)
		return Modes.modes[key]

	@staticmethod
	def get_int(key) -> int:
		return Modes.get(key)[0]

	@staticmethod
	def get_str(key) -> str:
		return Modes.get(key)[1]

	@staticmethod
	def get_tuple(key) -> Tuple:
		return Modes.get(key)[2]

	@staticmethod
	def get_tensor(key) -> torch.Tensor:
		return Modes.get(key)[3]

	@staticmethod
	def get_list(key) -> List:
		return list(Modes.get(key)[2])


class NumpyDataset(Dataset):
	def __init__(
			self,
			root_folder: str,
			transforms_source=None,
			transforms_target=None,
	):
		super(NumpyDataset, self).__init__()
		self.root_folder = root_folder
		self.transforms_source = transforms_source
		self.transforms_target = transforms_target

		files = [filename for filename in os.listdir(root_folder)
		         if os.path.isfile(os.path.join(root_folder, filename))
		         and 'numpy' in filename]

		getFileNumber_classical = lambda name: int(name.split('_')[0])
		getFileNumber_face      = lambda name: int(name.split('_')[1])

		try:
			x_files = sorted([f for f in files if '_x' in f], key=getFileNumber_classical)
			y_files = sorted([f for f in files if '_y' in f], key=getFileNumber_classical)
		except:
			x_files = sorted([f for f in files if '_x' in f], key=getFileNumber_face)
			y_files = sorted([f for f in files if '_y' in f], key=getFileNumber_face)

		self.files = list(zip(x_files, y_files))

	def __len__(self) -> int:
		return len(self.files)

	def __getitem__(self, idx):
		x_file, y_file = self.files[idx]

		x = np.load(os.path.join(self.root_folder, x_file)).astype(np.float32)
		y = np.load(os.path.join(self.root_folder, y_file)).astype(np.float32)

		if self.transforms_source is not None:
			x = self.transforms_source(x)
		if self.transforms_target is not None:
			y = self.transforms_target(y)

		return x, y


class ModeDataset(Dataset):
	def __init__(
			self,
			dataset: Dataset,
			proportion_xy: float,
			proportion_x: float,
			proportion_y: float,
			shuffle_modes: bool,
	):
		super(ModeDataset, self).__init__()
		"""Proportions should sum to one."""
		self.dataset = dataset

		dataset_length = len(dataset)

		length_xy = int(dataset_length * proportion_xy)
		length_x  = int(dataset_length * proportion_x)
		length_y  = dataset_length - length_xy - length_x

		modes = [Modes.get_list('xy')] * length_xy + [Modes.get_list('x')] * length_x + [Modes.get_list('y')] * length_y
		if shuffle_modes:
			random.shuffle(modes)

		self.indexes_xy = [i for i, m in enumerate(modes) if Modes.get_str(m) == 'xy']
		self.indexes_x  = [i for i, m in enumerate(modes) if Modes.get_str(m) == 'x']
		self.indexes_y  = [i for i, m in enumerate(modes) if Modes.get_str(m) == 'y']
		self.modes = modes

		self.supervised_length = len(self.indexes_xy)
		self.unsupervised_length = len(self.indexes_x) + len(self.indexes_y)

	def wandb_log_stats(self):
		wandb.run.summary['nb_xy'] = len(self.indexes_xy)
		wandb.run.summary['nb_x']  = len(self.indexes_x)
		wandb.run.summary['nb_y']  = len(self.indexes_y)
		print(f"""
			{len(self.indexes_xy)=}
			{len(self.indexes_x)=}
			{len(self.indexes_y)=}
		""")

	def __len__(self) -> int:
		return len(self.dataset)

	def __getitem__(self, index: int):
		x, y = self.dataset[index]
		mode = torch.Tensor(self.modes[index])
		return x, y, mode

	def get_filtered(self, filter_key: str, strict: bool):
		indices = []
		if not strict or filter_key == 'xy':
			indices += self.indexes_xy
		if filter_key == 'x':
			indices += self.indexes_x
		if filter_key == 'y':
			indices += self.indexes_y

		return Subset(self.dataset, indices)


class ModeNoLossDataset(Dataset):
	def __init__(
			self,
			dataset: Dataset,
			proportion_xy: float,
			shuffle_modes: bool,
			reserve: Dataset,
	):
		super(ModeNoLossDataset, self).__init__()

		dataset_length = len(dataset)
		length_xy      = int(dataset_length * proportion_xy)
		length_x_or_y  = dataset_length - length_xy

		if shuffle_modes:
			supervised_index   = random.sample(range(dataset_length), k=length_xy)
			unsupervised_index = list(set(range(dataset_length)) - set(supervised_index))
		else:
			supervised_index   = range(0, length_xy)
			unsupervised_index = range(length_xy, dataset_length)

		dataset_xy = Subset(dataset, supervised_index)
		dataset_x  = Subset(dataset, unsupervised_index)
		dataset_y  = Subset(reserve, unsupervised_index)

		self.dataset = ConcatDataset([dataset_xy, dataset_x, dataset_y])
		self.modes = ['xy'] * length_xy + ['x'] * length_x_or_y + ['y'] * length_x_or_y

		self.supervised_length = len(supervised_index)
		self.unsupervised_length = len(unsupervised_index)

	def wandb_log_stats(self):
		dict_cpt = {'xy': 0, 'x': 0, 'y': 0}
		for mode in self.modes:
			dict_cpt[mode] += 1

		wandb.run.summary['nb_xy'] = dict_cpt['xy']
		wandb.run.summary['nb_x']  = dict_cpt['x']
		wandb.run.summary['nb_y']  = dict_cpt['y']

	def __len__(self) -> int:
		return len(self.dataset)

	def __getitem__(self, index: int):
		x, y = self.dataset[index]
		mode = Modes.get_tensor(self.modes[index])
		return x, y, mode

	def get_filtered(self, filter_key: str, strict: bool) -> Dataset:
		keys = set() if strict else {'xy'}
		keys.add(filter_key)

		indices = [i for i, mode in enumerate(self.modes) if mode in keys]
		return Subset(self.dataset, indices)


class UnalignedDataset(Dataset):
	def __init__(self, dataset: Dataset, reserve: Dataset, proportion_xy: float, shuffle_modes: bool):
		super(UnalignedDataset, self).__init__()

		dataset_length = len(dataset)
		length_xy      = int(dataset_length * proportion_xy)
		length_x_or_y  = dataset_length - length_xy

		if shuffle_modes:
			supervised_index   = random.sample(range(dataset_length), k=length_xy)
			unsupervised_index = list(set(range(dataset_length)) - set(supervised_index))
		else:
			supervised_index   = range(0, length_xy)
			unsupervised_index = range(length_xy, dataset_length)

		dataset_xy = Subset(dataset, supervised_index)
		dataset_x  = Subset(dataset, unsupervised_index)
		dataset_y  = Subset(reserve, unsupervised_index)

		self.dataset_x = ConcatDataset([dataset_xy, dataset_x])
		self.dataset_y = ConcatDataset([dataset_xy, dataset_y])

		self.stats_nb_xy = len(dataset_xy)
		self.stats_nb_x  = len(dataset_x)
		self.stats_nb_y  = len(dataset_y)

	def wandb_log_stats(self):
		wandb.run.summary['nb_xy'] = self.stats_nb_xy
		wandb.run.summary['nb_x']  = self.stats_nb_x
		wandb.run.summary['nb_y']  = self.stats_nb_y

	def __len__(self):
		return max(len(self.dataset_x), len(self.dataset_y))

	def __getitem__(self, item_x: int):
		x1, y1 = self.dataset_x[item_x]

		item_y = random.randint(0, len(self.dataset_y) - 1)
		x2, y2 = self.dataset_y[item_y]

		return (x1, y1), (x2, y2)


class UnalignedDataset_OneLine(Dataset):
	def __init__(self, dataset: Dataset, reserve: Dataset, proportion_xy: float, shuffle_modes: bool):
		super(UnalignedDataset_OneLine, self).__init__()

		dataset_length = len(dataset)
		length_xy      = int(dataset_length * proportion_xy)
		length_x_or_y  = dataset_length - length_xy

		if shuffle_modes:
			supervised_index   = random.sample(range(dataset_length), k=length_xy)
			unsupervised_index = list(set(range(dataset_length)) - set(supervised_index))
		else:
			supervised_index   = range(0, length_xy)
			unsupervised_index = range(length_xy, dataset_length)

		dataset_xy = Subset(dataset, supervised_index)
		dataset_x  = Subset(dataset, unsupervised_index)
		dataset_y  = Subset(reserve, unsupervised_index)

		self.cat_dataset = ConcatDataset([dataset_xy, dataset_xy, dataset_x, dataset_y])
		self.modes = ['x'] * len(dataset_xy) + ['y'] * len(dataset_xy) + ['x'] * len(dataset_x) + ['y'] * len(dataset_y)

	def wandb_log_stats(self):
		dict_cpt = {'xy': 0, 'x': 0, 'y': 0}
		for mode in self.modes:
			dict_cpt[mode] += 1

		wandb.run.summary['nb_xy'] = dict_cpt['xy']
		wandb.run.summary['nb_x']  = dict_cpt['x']
		wandb.run.summary['nb_y']  = dict_cpt['y']

	def __len__(self):
		return len(self.cat_dataset)

	def __getitem__(self, item: int):
		x, y = self.cat_dataset[item]
		mode = Modes.get_tensor(self.modes[item])
		return x, y, mode


class UnalignedDataset_OneLine_Loss(Dataset):
	def __init__(self, dataset: Dataset, proportion_xy: float, shuffle_modes: bool):
		super(UnalignedDataset_OneLine_Loss, self).__init__()

		dataset_length = len(dataset)
		length_xy      = int(dataset_length * proportion_xy)
		length_x       = (dataset_length - length_xy)//2
		length_y       = dataset_length - length_xy - length_x

		if shuffle_modes:
			pool = set(range(dataset_length))
			xy_index = random.sample(pool, k=length_xy)
			pool    -= set(xy_index)
			x_index  = random.sample(pool, k=length_x)
			y_index  = list(pool - set(x_index))
		else:
			xy_index   = range(0, length_xy)
			x_index = range(length_xy, length_xy + length_x)
			y_index = range(length_xy + length_x, dataset_length)

		dataset_xy = Subset(dataset, xy_index)
		dataset_x  = Subset(dataset, x_index)
		dataset_y  = Subset(dataset, y_index)

		self.cat_dataset = ConcatDataset([dataset_xy, dataset_x, dataset_y])
		self.modes = ['xy'] * len(dataset_xy) + ['x'] * len(dataset_x) + ['y'] * len(dataset_y)

	def wandb_log_stats(self):
		dict_cpt = {'xy': 0, 'x': 0, 'y': 0}
		for mode in self.modes:
			dict_cpt[mode] += 1

		wandb.run.summary['nb_xy'] = dict_cpt['xy']
		wandb.run.summary['nb_x']  = dict_cpt['x']
		wandb.run.summary['nb_y']  = dict_cpt['y']
		print(f"""
		{dict_cpt['xy']=}
		{dict_cpt['x']=}
		{dict_cpt['y']=}
		""")

	def __len__(self):
		return len(self.cat_dataset)

	def __getitem__(self, item: int):
		x, y = self.cat_dataset[item]
		mode = Modes.get_tensor(self.modes[item])
		return x, y, mode
