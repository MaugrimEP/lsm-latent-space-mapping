# for IODA model, generate each dataloader
from typing import Set, Callable, Dict, Tuple
import wandb
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset

from datasets.Datasets import NumpyDataset, ModeNoLossDataset, ModeDataset
from datasets.dataset_utils import split_dataset, split_reserve


def get_dataloader_IODA(
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
	):
	# each call return the dataset with only the targeted mode
	# just used to separate the datasets
	numpy_dataset = NumpyDataset(
		root_folder=dataset_path,
		transforms_source=transforms_source,
		transforms_target=transforms_target,
	)
	print(f'{len(numpy_dataset)=}')
	dataset_train, dataset_valid, dataset_test = split_dataset(numpy_dataset, train_proportion, valid_proportion, test_proportion)

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
			reserve=second_half,
			shuffle_modes=shuffle_mode,
		)
	dataset_mode_train.wandb_log_stats()
	print(f'{len(dataset_mode_train)=}')

	dataset_input_train  = dataset_mode_train.get_filtered('x', strict=False)
	dataset_output_train = dataset_mode_train.get_filtered('y', strict=False)
	dataset_final_train  = dataset_mode_train.get_filtered('xy', strict=False)
	print(f'{len(dataset_input_train)=}')
	print(f'{len(dataset_output_train)=}')
	print(f'{len(dataset_final_train)=}')

	if reduce_train_size:
		dataset_input_train  = Subset(dataset_input_train , range(reduced_size))
		dataset_output_train = Subset(dataset_output_train, range(reduced_size))
		dataset_final_train  = Subset(dataset_final_train , range(reduced_size))

	dl_i_train = DataLoader(dataset_input_train, batch_size=batch_size_train, shuffle=shuffle_train)
	dl_i_valid = DataLoader(dataset_valid, batch_size=batch_size_valid)
	dl_i_test  = DataLoader(dataset_test, batch_size=batch_size_test)

	dl_o_train = DataLoader(dataset_output_train, batch_size=batch_size_train, shuffle=shuffle_train)
	dl_o_valid = DataLoader(dataset_valid, batch_size=batch_size_valid)
	dl_o_test  = DataLoader(dataset_test, batch_size=batch_size_test)

	dl_f_train = DataLoader(dataset_final_train, batch_size=batch_size_train, shuffle=shuffle_train)
	dl_f_valid = DataLoader(dataset_valid, batch_size=batch_size_valid)
	dl_f_test  = DataLoader(dataset_test, batch_size=batch_size_test)

	data = {
		'dataloader': {
			'input': {
				'train': dl_i_train, 'valid': dl_i_valid, 'test': dl_i_test,
			},
			'output': {
				'train': dl_o_train, 'valid': dl_o_valid, 'test': dl_o_test,
			},
			'full': {
				'train': dl_f_train, 'valid': dl_f_valid, 'test': dl_f_test,
			},
		},
		'dataset': {
			'input': {
				'train': dataset_input_train, 'valid': dataset_valid, 'test': dataset_test,
			},
			'output': {
				'train': dataset_output_train, 'valid': dataset_valid, 'test': dataset_test,
			},
			'full': {
				'train': dataset_final_train, 'valid': dataset_valid, 'test': dataset_test,
			},
		},
	}

	print(f"[input] train size :{len(dataset_input_train)}, valid size:{len(dataset_valid)}, test size: {len(dataset_test)}")
	print(f"[output] train size:{len(dataset_output_train)}, valid size:{len(dataset_valid)}, test size: {len(dataset_test)}")
	print(f"[full] train size  :{len(dataset_final_train)}, valid size:{len(dataset_valid)}, test size: {len(dataset_test)}")

	wandb.run.summary['dataset/input/train/size'] = len(dataset_input_train)
	wandb.run.summary['dataset/input/valid/size'] = len(dataset_valid)
	wandb.run.summary['dataset/input/test/size']  = len(dataset_test)

	wandb.run.summary['dataset/output/train/size'] = len(dataset_output_train)
	wandb.run.summary['dataset/output/valid/size'] = len(dataset_valid)
	wandb.run.summary['dataset/output/test/size']  = len(dataset_test)

	wandb.run.summary['dataset/full/train/size'] = len(dataset_final_train)
	wandb.run.summary['dataset/full/valid/size'] = len(dataset_valid)
	wandb.run.summary['dataset/full/test/size']  = len(dataset_test)

	return data
