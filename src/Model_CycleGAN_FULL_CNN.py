from Utils.loss_metrics import iou_torch
from Utils.utils import seg2onehot
from src.Model_CycleGAN_General import *
from src.Networks.CycleGAN.networks import ResnetGenerator, get_norm_layer, define_G, define_D, _get_activation
from src.Networks.IODA_SOP_utils import full_cnn_network, _get_linking_part, \
	get_cnn_link, get_full_cnn_native

type_x = TensorType['bs', 'c', 'h', 'w']
type_y = TensorType['bs', 'c_seg', 'h', 'w']

class CycleGAN_Full_CNN(PL_CycleGAN_Base):
	def get_x_to_y(self, params) -> nn.Module:
		if params['use_native_archi']:
			x_to_x, y_to_y, link_x_to_y, link_y_to_x = get_full_cnn_native(params)
			x_to_y = nn.Sequential(
				x_to_x.encoder,
				link_x_to_y,
				y_to_y.decoder,
			)
			return x_to_y
		else:
			generator = define_G(input_nc=self.in_dim, output_nc=self.out_dim, ngf=params['ngf'], netG=params['netG'],
							norm=params['norm_layer'], use_dropout=params['dropout'], init_type=params['init_type'],
							init_gain=params['init_gain'], o_activation=params['o_activation_y'])
			link = get_cnn_link(params) if params['use_link'] else nn.Identity()

			x_to_y = nn.Sequential()
			x_to_y.add_module('encoder', generator.encoder)
			x_to_y.add_module('link'   , link)
			x_to_y.add_module('decoder', generator.decoder)

			return x_to_y

	def get_y_to_x(self, params) -> nn.Module:
		if params['use_native_archi']:
			x_to_x, y_to_y, link_x_to_y, link_y_to_x = get_full_cnn_native(params)
			y_to_x = nn.Sequential(
				y_to_y.encoder,
				link_y_to_x,
				x_to_x.decoder,
			)
			return y_to_x
		else:
			generator = define_G(input_nc=self.out_dim, output_nc=self.in_dim, ngf=params['ngf'], netG=params['netG'],
							norm=params['norm_layer'], use_dropout=params['dropout'], init_type=params['init_type'],
							init_gain=params['init_gain'], o_activation=params['o_activation_x'])
			link = get_cnn_link(params) if params['use_link'] else nn.Identity()

			y_to_x = nn.Sequential()
			y_to_x.add_module('encoder', generator.encoder)
			y_to_x.add_module('link'   , link)
			y_to_x.add_module('decoder', generator.decoder)

			return y_to_x

	def get_discriminator_x(self, params) -> nn.Module:
		return define_D(input_nc=self.in_dim, ndf=params['ndf'], netD=params['netD'], n_layers_D=params['n_layers_D'],
		                norm=params['norm_layer'], init_type=params['init_type'], init_gain=params['init_gain'])

	def get_discriminator_y(self, params) -> nn.Module:
		return define_D(input_nc=self.out_dim, ndf=params['ndf'], netD=params['netD'], n_layers_D=params['n_layers_D'],
		                norm=params['norm_layer'], init_type=params['init_type'], init_gain=params['init_gain'])

	def _specific_step(self, pred: CycleGAN_Output, losses: CycleGAN_Losses, log_dict: dict, prefix: str) -> dict:
		miou_xy = iou_torch((pred.xy > 0.5).int(), pred.y.int())
		miou_yy = iou_torch((pred.yy > 0.5).int(), pred.y.int())

		bs, c, h, w = pred.x.shape

		weights = (h * w) / torch.sum(pred.y, dim=[0, 2, 3])
		ce_xy = F.cross_entropy(pred.xy, pred.y, weight=weights)
		ce_yy = F.cross_entropy(pred.yy, pred.y, weight=weights)

		log_dict |= {'miou_xy': miou_xy, 'miou_yy': miou_yy,
					 'ce_xy': ce_xy, 'ce_yy': ce_yy, }

		return log_dict

	#####################################################################################################

	def training_step(self, batch, batch_idx):
		x, y, mode = batch
		y = seg2onehot(y, self.out_dim)
		return super().training_step(batch=(x, y, mode), batch_idx=batch_idx)

	def validation_step(self, batch, batch_idx):
		x, y = batch
		y = seg2onehot(y, self.out_dim)
		return super().validation_step(batch=(x, y), batch_idx=batch_idx)

	def test_step(self, batch, batch_idx):
		x, y = batch
		y = seg2onehot(y, self.out_dim)
		return super().test_step(batch=(x, y), batch_idx=batch_idx)
