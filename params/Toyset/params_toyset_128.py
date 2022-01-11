from dataclasses import dataclass

root_w = 'datasets/Toyset'

root_l1 = '/home/2017018/tmayet01/datasets'
root_l2 = '/home/2022022/tmayet02/datasets'
root = root_w

dataset_name = "toyset_128_swap"

@dataclass
class cfg_dataset_class:
	dataset_root_folder: str = f'{root}/{dataset_name}/'
	dataset_name: str = dataset_name
	shuffle: bool = True
	shuffle_mode: bool = True
	pair_in_unsupervised: bool = False

	dataset_reduction: bool = False

	image_res: int = 128
	in_dim: int = 1
	out_dim: int = 2

	# to test
	reduce_train_size: bool = False
	reduced_size: int = 10

	# pairs information
	# set to None to use the file information, otherwise should sum to 1
	proportion_xy: float = 0.8
	proportion_x : float = 0.1
	proportion_y : float = 0.1
	dataset_scale: float = 1.

	train_size: float = 0.70
	valid_size: float = 0.10
	test_size : float = 0.20

cfg_dataset = cfg_dataset_class()
