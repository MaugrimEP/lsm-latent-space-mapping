from dataclasses import dataclass


@dataclass
class cfg_slurm_class:
	job_id: int = -1
	working_directory: str = "None"
	slurm_user: str = "user_nf"

cfg_slurm = cfg_slurm_class()
