from dataclasses import dataclass

import wandb
from torch.utils.data import DataLoader, Subset, Dataset

from datasets.Datasets import NumpyDataset, UnalignedDataset_OneLine, UnalignedDataset_OneLine_Loss
from datasets.dataset_utils import split_dataset, split_reserve


@dataclass
class CycleGAN_Data:
	dataset_train_unaligned: Dataset
	dataset_train_aligned: Dataset
	dataset_valid: Dataset
	dataset_test: Dataset

	dataloader_train_unaligned: DataLoader
	dataloader_valid: DataLoader
	dataloader_test: DataLoader


def get_dataloader_CycleGAN(
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
	) -> CycleGAN_Data:
	# each call return the dataset with only the targeted mode
	# just used to separate the datasets
	numpy_dataset = NumpyDataset(
		root_folder=dataset_path,
		transforms_source=transforms_source,
		transforms_target=transforms_target,
	)
	# split dataset in train, valid, test
	train_dataset, valid_dataset, test_dataset = split_dataset(
		dataset=numpy_dataset,
		train_proportion=train_proportion,
		valid_proportion=valid_proportion,
		test_proportion=test_proportion,
	)

	print(f"""
	Data lengths for each sampled datasets:
	{len(train_dataset)=}, {len(valid_dataset)=}, {len(test_dataset)=},
	""")

	if reduce_train_size:
		train_dataset = Subset(train_dataset, range(reduced_size))

	if dataset_reduction:
		unaligned_train_dataset = UnalignedDataset_OneLine_Loss(
			dataset=train_dataset,
			proportion_xy=proportion_xy,
			shuffle_modes=shuffle_mode,
		)
	else:
		if pair_in_unsupervised:
			first_half, second_half = split_reserve(train_dataset)
		else:
			first_half, second_half = train_dataset, train_dataset

		unaligned_train_dataset = UnalignedDataset_OneLine(
			dataset=first_half,
			reserve=second_half,
			proportion_xy=proportion_xy,
			shuffle_modes=shuffle_mode,
		)
	unaligned_train_dataset.wandb_log_stats()
	train_dataloader = DataLoader(unaligned_train_dataset, batch_size=batch_size_train, shuffle=shuffle_train)
	valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size_valid)
	test_dataloader  = DataLoader(test_dataset, batch_size=batch_size_test)

	print(f"train size :{len(train_dataset)}, valid size:{len(valid_dataset)}, test size: {len(test_dataset)}")

	wandb.run.summary['dataset/train/size'] = len(train_dataset)
	wandb.run.summary['dataset/train/size'] = len(valid_dataset)
	wandb.run.summary['dataset/train/size'] = len(test_dataset)

	return CycleGAN_Data(
		dataset_train_unaligned=unaligned_train_dataset,
		dataset_train_aligned=train_dataset,
		dataset_valid=valid_dataset,
		dataset_test=test_dataset,

		dataloader_train_unaligned=train_dataloader,
		dataloader_valid=valid_dataloader,
		dataloader_test=test_dataloader,
	)
