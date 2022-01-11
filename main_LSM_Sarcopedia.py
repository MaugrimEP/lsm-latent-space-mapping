from dataclasses import asdict
import pytorch_lightning as pl
from pprint import pprint
import wandb
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from Utils.HyperParams import HP_parsed
from Utils.loss_metrics import get_loss
from Utils.utils import timestamp, get_wandb_init
from datasets.SOP_dataloaders import SOP_data, get_dataloader_SOP
from datasets.transforms import sarco_transforms_source, sarco_transforms_target
from src.Callbacks.BestModel import BestModel
from src.Callbacks.SampleLogger.SampleLogger_IODA_SOP_SegFCNN import SampleLogger_Sarcopedia_IODA
from src.Model_LSM_General import PL_LSM_FULL_CNN, LSM_Lambdas

from params.Sarcopenia.model_params_sarcopedia_lsm import cfg
from params.slurm_params import cfg_slurm
from params.Sarcopenia.params_sarcopedia_128 import cfg_dataset


def get_params():
	# region params
	parser = HP_parsed([cfg, cfg_dataset, cfg_slurm])
	params = asdict(cfg) | asdict(cfg_dataset) | asdict(cfg_slurm) | parser.parse_from_cmd()
	pprint(params)
	# endregion
	return params


def main(params: dict) -> float:
	# region wandb
	run = get_wandb_init(params)
	# endregion

	# region data
	data: SOP_data = get_dataloader_SOP(
		train_proportion   =params['train_size'],
		valid_proportion   =params['valid_size'],
		test_proportion    =params['test_size'],

		batch_size_train   =params['batch_size_train'],
		batch_size_valid   =params['batch_size_valid'],
		batch_size_test    =params['batch_size_test'],

		shuffle_train      =params['shuffle'],

		reduced_size       =params['reduced_size'],
		reduce_train_size  =params['reduce_train_size'],

		proportion_xy      =params['proportion_xy'],
		proportion_x       =params['proportion_x'],
		proportion_y       =params['proportion_y'],

		dataset_path=params['dataset_root_folder'],
		shuffle_mode=True,
		transforms_source=sarco_transforms_source,
		transforms_target=sarco_transforms_target,
		pair_in_unsupervised=params['pair_in_unsupervised'],
		dataset_reduction=params['dataset_reduction'],
	)
	# endregion

	# region model
	model = PL_LSM_FULL_CNN(
		image_res=params['image_res'],
		input_dimension=params['in_dim'],
		output_dimension=params['out_dim'],

		input_loss =get_loss(params['input_loss'])(reduction=False),
		output_loss=get_loss(params['output_loss'])(reduction=False),
		latent_loss=get_loss(params['latent_loss'])(reduction=False),
		domain_adversarial_loss=get_loss(params['domain_adversarial_loss'])(),
		latent_adversarial_loss=get_loss(params['latent_adversarial_loss'])(params['latent_is_confusing']),
		lambdas=LSM_Lambdas(
			lambda_xx_loss=params['lambda_xx_loss'],
			lambda_yy_loss=params['lambda_yy_loss'],
			lambda_latent_supervised=params['lambda_latent_supervised'],
			lambda_supervised_domain_y=params['lambda_supervised_domain_y'],
			lambda_supervised_domain_x=params['lambda_supervised_domain_x'],
			lambda_adv_g_domain=params['lambda_adv_g_domain'],
			lambda_adv_g_latent=params['lambda_adv_g_latent'],
			lambda_adv_d_domain=params['lambda_adv_d_domain'],
			lambda_adv_d_latent=params['lambda_adv_d_latent'],
		),
		lr=params['learning_rate'],
		params=params,
	)
	print(model)
	# endregion

	print(f'[{timestamp()}] <START GLOBAL TRAINING>')
	# region training

	wandb_logger = WandbLogger()
	callbacks = [
		BestModel(
			monitor="valid/metric/miou_xy",
			mode='max',
		),
		SampleLogger_Sarcopedia_IODA(
			data.dataset_train_nomode, data.dataset_valid, data.dataset_test,
			params['sample_to_log'], params['sample_to_log'], params['sample_to_log'],
			epoch_frequency=params['epoch_frequency'],
		),
	]
	if params['use_early_stop']:
		early_stop = EarlyStopping(monitor=f"valid/metric/miou_xy", min_delta=params['delta_min'],
		                           patience=params['patience'], verbose=True, mode="max", strict=True)
		callbacks.append(early_stop)

	trainer = pl.Trainer(
		max_epochs=params['epochs'],
		accelerator='gpu' if str(params['device']) == 'cuda' else 'cpu',
		devices=1,
		check_val_every_n_epoch=1,
		callbacks=callbacks,
		logger=wandb_logger,
		enable_checkpointing=False,
		log_every_n_steps=1,
		accumulate_grad_batches=params['accumulate_grad_batches'],
	)
	trainer.fit(model, data.dataloader_train, data.dataloader_valid)
	print("end fitting")
	# endregion

	if params['save_model']:
		print("[SAVING MODEL]")
		trainer.save_checkpoint(f"Saves/{params['save_model_name']}")
		print("[MODE SAVED]")

	# region testing

	print("start testing")
	trainer.test(model=model, dataloaders=data.dataloader_test)
	# endregion
	print(f'[{timestamp()}] <END GLOBAL TRAINING + TESTING>')
	return wandb.summary['test/metric/miou_xy']

	# print(f'[{timestamp()}] <TERMINATE WANDB>')
	# wandb.finish()
	# print(f'[{timestamp()}] <WANDB TERMINATED>')
	# import sys
	# sys.exit()


if __name__ == '__main__':
	main(get_params())
