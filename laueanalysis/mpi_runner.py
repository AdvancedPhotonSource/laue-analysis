"""
MPI runner for PyLaueGo indexing.
"""
import os
import yaml
from datetime import datetime
import fire
from mpi4py import MPI

from laueanalysis.pyLaueGo import PyLaueGo


def run_mpi(config=None):
    """
    Run PyLaueGo with MPI support.
    
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
        config_dict = config
            
    pyLaueGo = PyLaueGo(config_dict, comm)
    pyLaueGo.run(rank, size)
    
    if rank == 0:
        print(f'runtime is {datetime.now() - start}')


if __name__ == '__main__':
    fire.Fire(run_mpi)
