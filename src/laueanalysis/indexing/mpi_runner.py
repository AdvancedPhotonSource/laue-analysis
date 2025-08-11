"""
MPI runner for functional indexing interface.
"""
import os
import yaml
from datetime import datetime
from mpi4py import MPI

from laueanalysis.indexing import index
from laueanalysis.indexing.lau_dataclasses.config import LaueConfig


def run_mpi(config=None):
    """
    Run indexing with MPI support using the functional interface.
    
    Args:
        config: Configuration dictionary or path to config file
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    start = datetime.now()
    
    # Convert config file path to config dict if needed
    if isinstance(config, str) and os.path.exists(config):
        with open(config, 'r') as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = config or {}
    
    # Convert to LaueConfig object
    laue_config = LaueConfig.from_dict(config_dict)
    
    # TODO: Implement MPI distribution logic for the functional interface
    # This will need to be redesigned based on how files should be distributed
    # across MPI processes. The old PyLaueGo.run() method would have contained
    # this logic.
    
    if rank == 0:
        print("WARNING: MPI runner needs to be redesigned for functional interface")
        print("The functional interface processes one file at a time.")
        print("MPI distribution logic needs to be implemented.")
    
    # For now, each rank could process different files, but we need the file list
    # This would typically come from the config or be generated based on patterns
    
    if rank == 0:
        print(f'runtime is {datetime.now() - start}')

