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
	# region model
	use_link        : bool = True      # use the IOPA/SOP/LSM link
	latent_resolution: int = 256      # resolution of the feature map in the latent space

	use_native_archi: bool = False     # use the same architecture as IODA & SOP
	# region native architecture parameters
	layers: List[Tuple[int, int]] = return_factory([[32, 2], [64, 2], [64, 2], [128, 3], [128, 3], [10, 1]])
	link: List[int] = return_factory([10, 10])
	cnn_link: bool = True
	latent_space_function: str = 'LeakyReLU'  # [ LeakyReLU | ReLU | Sigmoid ]
	# endregion

	o_activation: str = 'softmax'     # last layer activation function [ sigmoid | tanh | softmax | none]
	o_activation_x: str = 'sigmoid'     # should not be used
	o_activation_y: str = 'softmax'     # last layer activation function [ sigmoid | tanh | softmax | none]

	ngf: int = 64                     # # of generator filters in the last conv layer
	ndf: int = 64                     # # of discriminator filters in the fist conv layer
	netG: str = 'resnet_6blocks'      # [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]
	netD: str = 'basic'               # [basic [ n_layers | pixel]
	n_layers_D: int = 3               # only used if netD==n_layers
	norm_layer: str = 'instance'      # batch | instance | none
	init_type: str = 'normal'         # network initialization [normal | xavier | kaiming | orthogonal | none]
	init_gain: float = 0.02           # scaling factor for normal, xavier and orthogonal
	dropout: bool = False             # dropout for the generator

	use_scheduler: bool = False       # if we want to use a scheduler or not
	n_epochs: int = 100               # number of epochs with the initial learning rate
	n_epochs_decay: int = 100         # number of epochs to linearly decay learning rate to zero
	lr_policy: str = 'linear'         # learning rate policy. [linear | step | plateau | cosine]
	lr_decay_iters: int = 50          # multiply by a gamma every lr_decay_iters iterations

	beta1: float = 0.5                # momentum term of adam (default from adam is 0.9)

	discriminator_input_channels: int = 1 + 2
	# endregion

	# does the model want to use the avail data
	use_dataset_mode: bool = True

	# region WANDB
	offline  : bool = True
	to_delete: bool = True
	project_name: str = "Toyset_Pix2Pix"
	tags: List[str] = return_factory([])
	wdb_name: str = None
	# endregion

	# region TRAINING
	adversarial_loss: str = 'BCE'  # [BCE | BCE_NS | SOFTPLUS | MSE]
	output_loss: str = 'ce'
	lambda_supervised: float = 1.

	epochs          : int = 1000
	batch_size_train: int = 32
	learning_rate: float = 2e-3
	batch_size_valid: int = 32
	batch_size_test : int = 32
	accumulate_grad_batches: int = 1

	# early_stopping
	use_early_stop: bool = True
	delta_min     : int = 0
	patience      : int = 100
	# endregion

	# region logging_image
	epoch_frequency: int = 50
	sample_to_log_train: int = 5
	sample_to_log_valid: int = 5
	sample_to_log_test : int = 5
	# endregion

	# register device
	device: str = device
	save_model     : bool = True
	save_model_name: str = "curr_model.ckpt"
	# endregion


cfg = cfg_class()
