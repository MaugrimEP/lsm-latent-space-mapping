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
	landmark_d_layers: List[int] = return_factory([68*2, 256, 128, 64, 1])
	landmark_d_act   : str = 'none'  # last layer activation function [ sigmoid | tanh | softmax | none]

	use_link        : bool = True      # use the IOPA/SOP/LSM link
	latent_resolution: int = 256      # resolution of the feature map in the latent space

	use_native_archi: bool = False     # use the same architecture as IODA & SOP
	# region native architecture parameters
	layers_cnn: List[Tuple[int, int]] = return_factory([[64, 2], [126, 2], [256, 2], [512, 2], [1024, 1]])
	layers_mlp: List[int]             = return_factory([64, 256, 1024])
	link: List[int] = return_factory([1024*1*2*2, 1024])
	latent_space_function: str = 'LeakyReLU'
	# endregion

	o_activation_x: str = 'sigmoid'   # last layer activation function [ sigmoid | tanh | softmax | none]
	o_activation_y: str = 'sigmoid'   # last layer activation function [ sigmoid | tanh | softmax | none]

	ngf: int = 64                     # # of generator filters in the last conv layer
	ndf: int = 64                     # # of discriminator filters in the fist conv layer
	netG: str = 'resnet_6blocks'      # [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]
	netD: str = 'basic'               # [basic [ n_layers | pixel]
	n_layers_D: int = 3               # only used if netD==n_layers
	norm_layer: str = 'instance'      # batch | instance | none
	init_type: str = 'normal'         # network initialization [normal | xavier | kaiming | orthogonal]
	init_gain: float = 0.02           # scaling factor for normal, xavier and orthogonal
	dropout: bool = True              # dropout for the generator

	use_scheduler: bool = False       # if we want to use a scheduler or not
	n_epochs: int = 100               # number of epochs with the initial learning rate
	n_epochs_decay: int = 100         # number of epochs to linearly decay learning rate to zero
	lr_policy: str = 'linear'         # learning rate policy. [linear | step | plateau | cosine]
	lr_decay_iters: int = 50          # multiply by a gamma every lr_decay_iters iterations

	beta1: float = 0.5                # momentum term of adam
	# endregion

	# does the model want to use the avail data
	use_dataset_mode: bool = True

	# region WANDB
	offline  : bool = True
	to_delete: bool = True
	project_name: str = "Face_CycleGAN"
	tags: List[str] = return_factory([])
	wdb_name: str = None
	# endregion

	# region TRAINING
	lambda_cycle: float = 10

	adversarial_loss: str = 'BCE'  # [BCE | BCE_NS | SOFTPLUS | MSE]
	input_loss : str = 'l1'
	output_loss: str = 'mse'

	epochs          : int = 1000
	batch_size_train: int = 32
	learning_rate_g : float = 2e-4
	learning_rate_d : float = 2e-4
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
