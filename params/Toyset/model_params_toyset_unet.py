from dataclasses import dataclass
from typing import List, Tuple
import torch
from Utils.utils import return_factory

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
# device = torch.device('cpu')
print(f"{use_cuda=} {device=}")


@dataclass
class cfg_class:
	# does the model want to use the avail data
	use_dataset_mode: bool = True

	# region WANDB
	offline  : bool = True
	to_delete: bool = True
	project_name: str = "Toyset_UNET"
	tags: List[str] = return_factory([])
	# endregion

	# region TRAINING
	learning_rate: float = 1e-4
	epochs       : int   = 400

	batch_size_train: int = 25
	batch_size_valid: int = 25
	batch_size_test : int = 25
	accumulate_grad_batches: int = 1

	# early_stopping
	use_early_stop: bool  = True
	delta_min     : float = 0
	patience      : int   = 50
	# endregion

	# region TESTING
	epoch_frequency: int = 20
	sample_to_log  : int = 10
	# endregion

	# register device
	device         : str = device
	save_model     : bool = True
	save_model_name: str = "curr_model.ckpt"
	# endregion


cfg = cfg_class()
