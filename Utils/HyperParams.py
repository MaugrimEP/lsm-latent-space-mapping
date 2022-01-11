from dataclasses import asdict

from jsonargparse import ArgumentParser
from functools import reduce
from typing import List, Tuple, Any, Set, get_type_hints


class HP:
	def __init__(self, name: str, values: List):
		self.name = name
		self.values = values

	def get_upgrade_list(self) -> List[Tuple[str, Any]]:
		upgraded_list = []
		for value in self.values:
			upgraded_list.append((self.name, value))
		return upgraded_list


class HP_parsed:
	def __init__(self, _list_dict_cfg: List[object], grid_search_params_list: Set[str] = None):
		"""
		:param _list_dict_cfg: List of dataclass type
		:param grid_search_params_list: List of tuple (param_name, List of values for the grid search)
		"""
		if grid_search_params_list is None:
			grid_search_params_list = set()

		list_types         = [get_type_hints(i) for i in _list_dict_cfg]
		list_dict_cfg = [asdict(i) for i in _list_dict_cfg]
		self.flat_configuration = reduce(lambda d1, d2: d1 | d2, list_dict_cfg)
		self.list_types         = reduce(lambda d1, d2: d1 | d2, list_types)

		self.parsed_args = self.parse_from_cmd()
		self._check_grid_intersection(grid_search_params_list)


	def _check_grid_intersection(self, grid_search_params_list: Set[str]):
		"""
		The parameter in the command line should not interfere with the grid search parameters
		=> there intersection has to be empty
		:param grid_search_params_list:
		:return:
		"""
		intersect = set(self.parsed_args.keys()) & grid_search_params_list
		if len(intersect) != 0:
			raise Exception("sanity check fails, the params line arguments should not intersect with the grid params")


	def parse_from_cmd(self) -> dict[str, Any]:
		"""
		We basically take all default arguments from config files
		flatten the dict
		and for each params, either we have it on the command line in the form of a list
		or either it will be the default value in the conf file
		and we do the fusion a after with the grid search
		:return:
		"""
		parser = ArgumentParser(description='HP command parser', exit_on_error=False)
		for params_name, params_type in self.list_types.items():
			parser.add_argument(f'--{params_name}', type=params_type)

		args_p = parser.parse_args()
		formatted_args = vars(args_p)

		present_args = dict()
		for params_name, list_values in formatted_args.items():
			if list_values is not None:
				present_args[params_name] = list_values

		return present_args
