from copy import deepcopy
from pytorch_lightning import Callback
import pytorch_lightning as pl

import wandb

class BestModel(Callback):
	"""
	At the end of the training, load the best parameters for the model
	chosen using validation loss
	"""

	def __init__(self, monitor: str, mode: str):
		super().__init__()
		self.monitor = monitor
		self.mode = mode

		self.best_epoch   = None
		self.best_monitor = None
		self.best_model_params = dict()

	def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
		self.epoch = 0

	def _compare(self, curr, best):
		if self.mode == 'min':
			return curr < best
		else:  # self.mode == 'max'
			return curr > best

	def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
		curr_monitored  = trainer.callback_metrics[self.monitor].detach().item()
		self.epoch += 1

		if self.best_monitor is None or self._compare(curr_monitored, self.best_monitor):
			print(f'[BestModel]: IMPROVEMENT ON EPOCH {self.epoch}')
			self.best_epoch = self.epoch
			self.best_monitor = curr_monitored
			self.best_model_params = deepcopy(pl_module.state_dict())
			for key in self.best_model_params.keys():
				self.best_model_params[key] = self.best_model_params[key].cpu()

	def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
		print(f'[BestModel]: LOAD BEST VALIDATION PARAMETERS')
		print(f'[BestModel]: BEST EPOCH {self.best_epoch}')
		print(f'[BestModel]: BEST MONITORED {self.monitor} : {self.best_monitor}')
		pl_module.load_state_dict(self.best_model_params)

		if hasattr(pl_module, 'mode'):
			clef = f'/{pl_module.mode}'
		else:
			clef = ""
		wandb.run.summary[f'bestModel{clef}/bestEpoch'] = self.best_epoch
		wandb.run.summary[f'bestModel{clef}/bestMonitored'] = self.best_monitor

		print(f'[BestModel]: best parameters restored')
