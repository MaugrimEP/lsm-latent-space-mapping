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

	input_loss: str = 'mse'
	output_loss: str = 'ce'

	# region model
	use_native_archi: bool = False     # use the same architecture as IODA & SOP
	# region native architecture parameters
	layers: List[Tuple[int, int]] = return_factory([[32, 2], [64, 2], [126, 2], [256, 3], [512, 3], [1024, 1]])
	link: List[int] = return_factory([1024, 1024])
	cnn_link: bool = True
	latent_space_function: str = 'LeakyReLU'  # [ LeakyReLU | ReLU | Sigmoid ]
	# endregion

	o_activation_x: str = 'sigmoid'     # last layer activation function [ sigmoid | tanh | softmax | none]
	o_activation_y: str = 'softmax'     # last layer activation function [ sigmoid | tanh | softmax | none]

	ngf: int = 64                     # # of generator filters in the last conv layer
	netG: str = 'resnet_6blocks'      # [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]
	latent_resolution: int = 256      # resolution of the feature map in the latent space
	norm_layer: str = 'instance'      # batch | instance | none
	init_type: str = 'normal'         # network initialization [normal | xavier | kaiming | orthogonal | none]
	init_gain: float = 0.02           # scaling factor for normal, xavier and orthogonal
	dropout: bool = False             # dropout for the generator
	# endregion

	# does the model want to use the avail data
	use_dataset_mode: bool = True

	# region WANDB
	offline: bool = True
	to_delete: bool = True
	project_name: str = "Sarcopenia_IODA"
	tags: List[str] = return_factory([])
	wdb_name: str = None
	# endregion

	# region TRAINING
	pretrain_input : bool = True
	pretrain_output: bool = True
	train_full     : bool = True

	learning_rate  : float = 1e-4

	epochs_input : int = 400
	epochs_output: int = 400
	epochs_full  : int = 400

	batch_size_train: int = 32
	batch_size_valid: int = 32
	batch_size_test : int = 32
	accumulate_grad_batches: int = 1

	# early_stopping
	pretrain_best      : bool = True
	pretrain_early_stop: bool = False

	use_early_stop: bool  = True
	delta_min     : float = 0
	patience      : int   = 50
	# endregion

	# region TESTING
	epoch_frequency: int = 20
	sample_to_log  : int = 10
	# endregion

	# register device
	device         : str  = device
	save_model     : bool = True
	save_model_name: str  = "curr_model.ckpt"
	# endregion


cfg = cfg_class()
