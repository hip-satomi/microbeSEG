import json
import numpy as np
import random
import torch

from pathlib import Path
from src.microbe_seg_gui import run_gui


def run_microbe_seg():
    """ Run the microbe segmentation tool microbeSEG """

    random.seed()
    np.random.seed()

    # Get default settings for the omero login
    with open(Path(__file__).parent/'settings.json') as f:
        settings = json.load(f)

    # Set device for using CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device) == 'cuda':
        torch.backends.cudnn.benchmark = True
        use_gpu = True
        num_gpus = torch.cuda.device_count()
    else:
        use_gpu = False
        num_gpus = 0

    run_gui(model_path=Path(__file__).parent/'models',
            training_data_path=Path(__file__).parent/'training_dataset',
            eval_results_path=Path(__file__).parent/'evaluation',
            inference_results_path=Path(__file__).parent/'results',
            gpu=use_gpu,
            multi_gpu=True if num_gpus > 1 else False,
            omero_settings=settings)


if __name__ == "__main__":

    run_microbe_seg()
