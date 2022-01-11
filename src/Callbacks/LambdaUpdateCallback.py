from pytorch_lightning import Callback


class LambdaUpdateCallback(Callback):
	def __init__(self):
		super(LambdaUpdateCallback, self).__init__()
		self.curr_epoch = 0

	def on_train_epoch_end(self, trainer, pl_module):
		updated_lambda = []
		initial_lambdas = pl_module.initial_lambdas
		for initial_lambda, star_decay, end_decay in zip(initial_lambdas, pl_module.lambdas_start_epoch, pl_module.lambdas_stop_epoch):
			if end_decay == -1 or self.curr_epoch < star_decay:
				fraction = 1
			else:
				fraction = ((end_decay - self.curr_epoch) / (end_decay - star_decay))
				if fraction < 0:
					fraction = 0
			lambda_value = initial_lambda * fraction
			updated_lambda.append(lambda_value)

		pl_module.lambda_1 = updated_lambda[0]
		pl_module.lambda_2 = updated_lambda[1]
		pl_module.lambda_3 = updated_lambda[2]
		pl_module.lambda_4 = updated_lambda[3]
		pl_module.lambda_5 = updated_lambda[4]
		pl_module.lambda_6 = updated_lambda[5]

		self.curr_epoch += 1