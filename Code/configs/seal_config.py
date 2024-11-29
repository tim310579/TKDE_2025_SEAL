import os
from datetime import datetime
from dataclasses import dataclass

@dataclass
class Config():
    sampling_rate: int = 500
    
    window_size: int = 256

    model_name: str = "SEAL"

    num_worker: int = 4

    epoch_size: int = 100

    learning_rate: float = 0.0001

    batch_size: int = 12

    input_size: int = 12

    hidden_size: int = 256

    hidden_output_size: int = 1

    output_size: int = 9

    alpha: float = 0.3

    beta: float = 0.001

    gamma: float = 0.5

    message: str = "None"
    
    seed: int = 1

    # path
    root_dir: str = "/code/path/here/"
    
    model_root_dir: str = "/model/path/here/"

    data_dir: str = "Datasets"

    snippet_dir: str = "snippet"
    
    tmp_dir: str = "tmp"

    model_dir: str = "models"

    wandb_dir: str = "wandb"

    state_dir: str = "state"

    output_dir: str = "EMTSC"

    state_name: str = "state.pkl"

    dataset_name: str = "ICBEB"

    snippet_name: str = "gamboa_norm_1000.pickle"
