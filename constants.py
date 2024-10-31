import pathlib
from enum import Enum


current_directory = pathlib.Path(__file__).parent
data_path = current_directory / "data"

class Dataset_Config(Enum):
    FILE_NAME = "Final Database.csv"
    DATASET_PATH = data_path_2 = data_path / FILE_NAME
    TRAIN_BATTERIES_COUNT = 10
    TEST_BATTERIES_COUNT = 4
    BATCH_SIZE = 32
    SLIDING_WINDOW_SIZE = 20
    SMOOTHING_WINDOW_SIZE = 5

class Model_Config(Enum):
    EPOCHS = 5
    SAVED_MODELS_PATH = current_directory / 'models'
    LEARNING_RATE = 1e-3
    REGULARIZATION_PARAMETER = 1e-5
    HIDDEN_DIM = 32
    TIME_DIVS = 5

 






