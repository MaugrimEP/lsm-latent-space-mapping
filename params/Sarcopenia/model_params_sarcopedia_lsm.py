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
	# LSM PARAMS
	# region lambda
	lambda_xx_loss            : float = 1.
	lambda_yy_loss            : float = 1.

	lambda_latent_supervised  : float = 1.
	lambda_supervised_domain_x: float = 10.
	lambda_supervised_domain_y: float = 10.

	lambda_adv_g_domain       : float = 0
	lambda_adv_g_latent       : float = 1.

	lambda_adv_d_domain       : float = 0
	lambda_adv_d_latent       : float = 1.

	lambda_cycle_xx           : float = 0.  # currently not used
	lambda_cycle_yy           : float = 0.  # currently not used

	lambda_cycle_xLxL         : float = 0.  # currently not used
	lambda_cycle_yLyL         : float = 0.  # currently not used
	# endregion

	input_loss : str = 'l1'
	output_loss: str = 'ce'
	latent_loss: str = 'l1'

	domain_adversarial_loss: str = 'BCE'  # [BCE | BCE_NS | SOFTPLUS | MSE]
	latent_adversarial_loss: str = 'BCE'  # [BCE | BCE_NS | SOFTPLUS | MSE]
	latent_is_confusing    : bool = False

	# region model
	use_native_archi: bool = False     # use the same architecture as IODA & SOP
	# region native architecture parameters
	layers: List[Tuple[int, int]] = return_factory([[32, 2], [64, 2], [126, 2], [256, 3], [512, 3], [1024, 1]])
	link: List[int] = return_factory([1024, 1024])
	cnn_link: bool = True
	latent_space_function: str = 'LeakyReLU'  # [ LeakyReLU | ReLU | Sigmoid ]

	native_discriminator_latent_layer_in: List[int] = return_factory([1024*1*2*2, 256, 128, 64, 1])
	native_discriminator_latent_act_in: str = 'none'  # last layer activation function [ sigmoid | tanh | softmax | none]

	native_discriminator_latent_layer_out: List[int] = return_factory([1024*1*2*2, 256, 128, 64, 1])
	native_discriminator_latent_act_out: str = 'none'  # last layer activation function [ sigmoid | tanh | softmax | none]
	# endregion

	o_activation_x: str = 'sigmoid'     # last layer activation function [ sigmoid | tanh | softmax | none]
	o_activation_y: str = 'softmax'     # last layer activation function [ sigmoid | tanh | softmax | none]

	ngf: int = 64                     # # of generator filters in the last conv layer
	latent_resolution: int = 256
	ndf: int = 64                     # # of discriminator filters in the fist conv layer
	netG: str = 'resnet_6blocks'      # [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]
	netD_domain: str = 'basic'               # [basic [ n_layers | pixel]
	netD_latent: str = 'basic'               # [basic [ n_layers | pixel]
	n_layers_D_domain: int = 3               # only used if netD==n_layers
	n_layers_D_latent: int = 3               # only used if netD==n_layers
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
	project_name: str = "Sarcopenia_LSM"
	tags: List[str] = return_factory([])
	wdb_name: str = None
	# endregion

	# region TRAINING
	learning_rate  : float = 1e-4

	epochs: int = 400

	batch_size_train: int = 32
	batch_size_valid: int = 32
	batch_size_test : int = 32
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
	device         : str  = device
	save_model     : bool = True
	save_model_name: str  = "curr_model.ckpt"
	# endregion


cfg = cfg_class()
