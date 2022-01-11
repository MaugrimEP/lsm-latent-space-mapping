from dataclasses import dataclass
from typing import Optional, List


@dataclass
class cfg_Testing:
	id_list: List[str] = None  # if not None, model_folder == ''
	bool_list: List[bool] = None
	prop_list: List[int] = None

	model_folder: str = ''  # folder where all the models are situated 'C:/Users/tmayet/MobaXterm/home/fetch_Face_LSM'

	dest_folder: str = 'C:/Users/tmayet/PycharmProjects/LSM/Testing/Inferences_results/Face'  # where we save the predictions
	test_mode: str = 'toyset'  # face | toyset | sarcopenia


cfg_testing = cfg_Testing()
