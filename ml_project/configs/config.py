from dataclasses import dataclass
from typing import Dict, List


@dataclass
class SplittingParams:
    test_size: float
    random_state: int


@dataclass
class TrainParams:
    input_data_path: str
    target_name: str
    splitting_params: SplittingParams
    model: any
    dump_model: str


@dataclass
class PredictParams:
    input_data_path: str
    model_path: str
    save_path: str


@dataclass
class MakeDatasetParams:
    input_data_path: str
    target_name: str
    features: Dict[str, List[str]]
    save_path: str
    name: str
    n_samples: int
