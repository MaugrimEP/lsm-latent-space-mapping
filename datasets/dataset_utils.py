from typing import Tuple
from torch.utils.data import Dataset, Subset


def split_dataset(
		dataset,
		train_proportion,
		valid_proportion,
		test_proportion,
) -> Tuple[Dataset, Dataset, Dataset]:
	"""
	Divide the dataset into train, valid, test
	Add to the test set the complement not used
	"""
	# region compute train, valid, and test size
	dataset_size = len(dataset)
	train_proportion = int(train_proportion * dataset_size)
	valid_proportion = int(valid_proportion * dataset_size)
	test_proportion  = int(test_proportion  * dataset_size)

	# split the dataset
	train = Subset(dataset, range(0, train_proportion))
	valid = Subset(dataset, range(train_proportion, train_proportion + valid_proportion))
	test  = Subset(dataset, range(train_proportion + valid_proportion, train_proportion + valid_proportion + test_proportion))

	return train, valid, test


def split_reserve(dataset: Dataset) -> Tuple[Dataset, Dataset]:
	dataset_length = len(dataset)
	middle = dataset_length //2
	d1 = Subset(dataset, range(0, middle))
	d2 = Subset(dataset, range(middle, middle*2))
	assert len(d1) == len(d2)
	return d1, d2
