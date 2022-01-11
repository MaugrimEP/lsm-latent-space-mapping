from dataclasses import dataclass

import torch
import wandb
from torch.utils.data import Subset, DataLoader, Dataset, WeightedRandomSampler

from datasets.Datasets import NumpyDataset, ModeNoLossDataset, Modes, ModeDataset
from datasets.dataset_utils import split_dataset, split_reserve


@dataclass
class SOP_data:
	dataset_train_nomode: Dataset
	dataset_train_mode  : Dataset
	dataset_valid: Dataset
	dataset_test : Dataset

	dataloader_train: DataLoader
	dataloader_valid: DataLoader
	dataloader_test : DataLoader


def get_dataloader_SOP(
		train_proportion: float,
		valid_proportion: float,
		test_proportion : float,

		batch_size_train: int,
		batch_size_valid: int,
		batch_size_test : int,

		shuffle_train: bool,

		reduce_train_size: bool,
		reduced_size     : int,

		proportion_xy: float,
		proportion_x : float,
		proportion_y : float,

		dataset_path: str,
		shuffle_mode: bool,
		transforms_source,
		transforms_target,
		pair_in_unsupervised: bool,
		dataset_reduction: bool
	) -> SOP_data:

	numpy_dataset = NumpyDataset(
		root_folder=dataset_path,
		transforms_source=transforms_source,
		transforms_target=transforms_target,
	)
	print(f'{len(numpy_dataset)=}')
	dataset_train, dataset_valid, dataset_test = split_dataset(numpy_dataset, train_proportion, valid_proportion, test_proportion)
	if reduce_train_size:
		dataset_train = Subset(dataset_train, range(reduced_size))
	print(f'{len(dataset_train)=}')
	print(f'{len(dataset_valid)=}')
	print(f'{len(dataset_test) =}')

	if dataset_reduction:
		dataset_mode_train = ModeDataset(
			dataset=dataset_train,
			proportion_xy=proportion_xy,
			proportion_x=proportion_x,
			proportion_y=proportion_y,
			shuffle_modes=shuffle_mode,
		)
	else:
		if pair_in_unsupervised:
			first_half, second_half = split_reserve(dataset_train)
		else:
			first_half, second_half = dataset_train, dataset_train

		dataset_mode_train = ModeNoLossDataset(
			dataset=first_half,
			proportion_xy=proportion_xy,
			shuffle_modes=shuffle_mode,
			reserve=second_half,
		)
	dataset_mode_train.wandb_log_stats()
	print(f'{len(dataset_mode_train)=}')

	# region compute weights for each sample
	dataset_size        = len(dataset_mode_train)
	weights_per_sample  = []
	for curr_mode in dataset_mode_train.modes:
		curr_mode = Modes.get_str(curr_mode)
		weights_per_sample.append(dataset_size / dataset_mode_train.supervised_length if curr_mode == 'xy'
		                          else dataset_size / dataset_mode_train.unsupervised_length)

	sampler = WeightedRandomSampler(
		weights=torch.DoubleTensor(weights_per_sample),
		num_samples=dataset_size,
		replacement=False,
	)
	# endregion

	dataloader_train = DataLoader(dataset_mode_train, batch_size=batch_size_train, sampler=sampler)
	dataloader_valid = DataLoader(dataset_valid     , batch_size=batch_size_valid)
	dataloader_test  = DataLoader(dataset_test      , batch_size=batch_size_test)

	print(f"[datasets] train size :{len(dataset_mode_train)}, valid size:{len(dataset_valid)}, test size: {len(dataset_test)}")

	wandb.run.summary['dataset/train/size'] = len(dataset_mode_train)
	wandb.run.summary['dataset/valid/size'] = len(dataset_valid)
	wandb.run.summary['dataset/test/size']  = len(dataset_test)

	return SOP_data(
		dataset_train_nomode=dataset_train,
		dataset_train_mode  =dataset_mode_train,

		dataset_valid=dataset_valid,
		dataset_test=dataset_test,

		dataloader_train=dataloader_train,
		dataloader_valid=dataloader_valid,
		dataloader_test=dataloader_test,
	)
