import argparse
import json
import numpy as np
import tifffile as tiff
import torch
import torch.nn.functional as F

from pathlib import Path

from src.inference.postprocessing import boundary_postprocessing, distance_postprocessing
from src.utils.unets import build_unet, get_weights
from src.utils.utils import zero_pad_model_input


def main():

    parser = argparse.ArgumentParser(description='microbeSEG inference script')
    parser.add_argument('--img_dir', '-i', required=True, type=str, help='Directory with image files to process (.tif, .tiff)')
    parser.add_argument('--model', '-m', required=True, type=str, help='Path to model')
    parser.add_argument('--thresholds', '-t', default=[0.10, 0.45], type=float, nargs='+', help='Thresholds for distance models')
    parser.add_argument('--result_path', '-r', default=None, type=str, help='Result path')
    parser.add_argument('--channel', '-c', default=0, type=int, help='Channel to process')
    parser.add_argument('--device', '-d', default='cuda:0', help='Device to use')
    parser.add_argument('--overwrite', '-o', default=False, action='store_true', help='Overwrite existing results')
    args = parser.parse_args()

    # Path
    imgs_path = Path(args.img_dir)
    result_path = (Path(__file__).parents[2] / 'results') if args.result_path is None else Path(args.result_path)
    result_path.mkdir(exist_ok=True)

    # Check if model is available
    inference_model = Path(args.model)
    if not (inference_model.parent / f"{inference_model.stem}.pth").is_file():
        raise Exception(f'{inference_model.parent / f"{inference_model.stem}.pth"} not found!')
    if not (inference_model.parent / f"{inference_model.stem}.json").is_file():
        raise Exception(f'{inference_model.parent / f"{inference_model.stem}.json"} not found!')
    with open(inference_model.parent / f"{inference_model.stem}.json") as f:
        model_settings = json.load(f)

    # Check thresholds
    if len(args.thresholds) != 2:
        raise Exception(f"{len(args.thresholds)} threshold given, needed are 2")

    # Set device for using CPU or GPU
    if 'cuda' in args.device and not torch.cuda.is_available():
        raise ValueError('No cuda capable gpu device detected, use device "cpu"')
    device = torch.device(args.device)
    if 'cuda' in str(device):
        torch.backends.cudnn.benchmark = True

    # Get all file ids
    file_ids = list(imgs_path.glob('*.tif*'))
    if len(file_ids) == 0:
        print('No files found')
        return

    # Build net
    net = build_unet(unet_type=model_settings['architecture'][0],
                     act_fun=model_settings['architecture'][2],
                     pool_method=model_settings['architecture'][1],
                     normalization=model_settings['architecture'][3],
                     device=device,
                     num_gpus=1,  # Only batch size 1 is used at the time, so makes no sense to use more gpus
                     ch_in=1,
                     ch_out=1 if model_settings['label_type'] == 'distance' else 3,
                     filters=model_settings['architecture'][4])

    # Load weights
    net = get_weights(net=net,
                      weights=str(inference_model.parent / f"{inference_model.stem}.pth"),
                      num_gpus=1,
                      device=device)
    net.eval()
    torch.set_grad_enabled(False)

    print('--- Start inference ---')

    for i, img_id in enumerate(file_ids):

        # Load image
        img = tiff.imread(str(img_id))
        fname = result_path / img_id.stem

        # Check shape + extract channel  ---> image needs to be in shape [time dimension, height, width]
        if len(img.shape) == 2:
            img = img[None, ...]
        elif len(img.shape) == 3:
            if img.shape[-1] == 3:  # probably rgb image
                img = img[..., args.channel]
                img = img[None, ...]
            elif img.shape[0] == 3:  # probably rgb image
                img = img[args.channel, ...]
                img = img[None, ...]
        elif len(img.shape) == 4:
            img = img[:, args.channel, ...]  # channels expected at 2nd position --> change here for your data
        elif len(img.shape) == 5:
            print(f'Skip {fname.name} (not supported image shape)')
            continue

        # Check if results exist and should not be overwritten
        already_processed = False
        if (result_path / f"mask_{fname.stem}_channel{args.channel}.tif").is_file():
            already_processed = True

        if already_processed and not args.overwrite:
            print(f'Skip {fname.name} (already processed and overwriting not enabled)')
            continue

        print(f'Process {fname.name} (channel: {args.channel})')

        # Pre-allocate array for results
        results_array = np.zeros(shape=(img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint16)

        # Go through frames
        for frame in range(len(img)):

            # Get image from Omero
            img_frame = np.copy(img[frame])

            # Get frame_min and frame_max before padding/cropping
            frame_min, frame_max = np.min(img_frame), np.max(img_frame)

            # Zero padding
            img_frame, pads = zero_pad_model_input(img_frame, pad_val=frame_min)

            # Normalize crop and convert to tensor / img_batch  --> dataset and loader needed for larger batch sizes
            img_batch = 2 * (img_frame.astype(np.float32) - frame_min) / (frame_max - frame_min) - 1
            img_batch = torch.from_numpy(img_batch[None, None, :, :]).to(torch.float)
            img_batch = img_batch.to(device)

            # Prediction
            if model_settings['label_type'] == 'distance':
                try:
                    prediction_border_batch, prediction_cell_batch = net(img_batch)
                except RuntimeError:
                    prediction = np.zeros_like(img_frame, dtype=np.uint16)[pads[0]:, pads[1]:]
                    print('RuntimeError during inference (maybe not enough ram/vram?)')
                else:
                    prediction_cell_batch = prediction_cell_batch[0, 0, pads[0]:, pads[1]:, None].cpu().numpy()
                    prediction_border_batch = prediction_border_batch[0, 0, pads[0]:, pads[1]:, None].cpu().numpy()
                    prediction = distance_postprocessing(border_prediction=np.copy(prediction_border_batch),
                                                         cell_prediction=np.copy(prediction_cell_batch),
                                                         th_cell=args.thresholds[0],
                                                         th_seed=args.thresholds[1])
            else:
                try:
                    prediction_batch = net(img_batch)
                except RuntimeError:
                    prediction = np.zeros_like(img_frame, dtype=np.uint16)[pads[0]:, pads[1]:]
                    print('RuntimeError during inference (maybe not enough ram/vram?)')
                else:
                    prediction_batch = F.softmax(prediction_batch, dim=1)
                    prediction_batch = prediction_batch[:, :, pads[0]:, pads[1]:].cpu().numpy()
                    prediction_batch = np.transpose(prediction_batch[0], (1, 2, 0))
                    prediction = boundary_postprocessing(prediction_batch)

            # Fill results array
            results_array[frame] = prediction

        # Save predictions
        results_array = np.squeeze(results_array)
        tiff.imwrite(str(result_path / f"mask_{fname.stem}_channel{args.channel}.tif"), results_array)

    print('--- Finished ---')


if __name__ == "__main__":

    main()
