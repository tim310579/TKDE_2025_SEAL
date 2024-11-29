from datetime import datetime
from dataclasses import dataclass


@dataclass
class DataConfig():

    sampling_rate: int = 500

    window_size: int = 256

    segmenter: str = "gamboa"
    
    root_dir: str = "/code/path/here/"
    
    model_root_dir: str = "/model/path/here/"
    
    data_dir: str = "Datasets"

    snippet_dir: str = "snippet"

    tmp_dir: str = "tmp"

    output_dir: str = "EMTSC"

    dataset_name: str = "ptbxl"

    snippet_name: str = "christov.pickle"
