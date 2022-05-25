import gc
import json
import hashlib
import shutil

import numpy as np
import os
# import requests
import pandas as pd
import tifffile as tiff
import torch
import torch.nn.functional as F
import zipfile

from copy import deepcopy
from itertools import product
from multiprocessing import cpu_count
from PyQt5.QtCore import pyqtSignal, QObject, pyqtSlot, QCoreApplication
from skimage import measure

from src.utils.unets import build_unet, get_weights
from src.inference.inference_dataset import InferenceDataset, pre_processing_transforms
from src.inference.postprocessing import boundary_postprocessing, distance_postprocessing
from src.evaluation.stats_utils import get_fast_aji_plus
from src.utils.utils import border_correction


class EvalWorker(QObject):
    """ Worker class for model evaluation """
    finished = pyqtSignal()  # Signal when import is finished
    progress = pyqtSignal(int)  # Signal for updating the progress bar
    text_output = pyqtSignal(str)  # Signal for possible exceptions, e.g., user interaction to stop export
    stop_evaluation = False  # Stop evaluation process
    is_evaluating = False  # State of evaluation process

    def start_evaluation(self, path_data, path_results, models, batch_size, device, num_gpus, save_raw_pred,
                         start_message=''):
        """ Start evaluation process

        :param path_data: Path of the training set (contains dirs 'train', and 'val').
        :type path_data: pathlib Path object
        :param path_results: Path to save the evaluation results into.
        :type path_results: pathlib Path object
        :param models: List with paths of the models to evaluate.
        :type models: list
        :param batch_size: Batch size.
        :type batch_size: int
        :param device: Device to use (gpu or cpu).
        :type device: torch device
        :param num_gpus: Number of gpus to use.
        :type num_gpus: int
        :param save_raw_pred: Save also the raw cnn outputs.
        :type save_raw_pred: bool
        :param start_message: Message to print at start of evaluation.
        :type start_message: str
        :return: None
        """

        # Check if export has been stopped (folders are deleted)
        if not path_data.is_dir():
            self.progress.emit(0)
            self.finished.emit()
            return

        # Check if enough test images are available:
        if len(list((path_data / 'test').glob('mask*'))) < 2:
            self.text_output.emit('Not enough test images found. At least 2 are needed (better more)')
            self.progress.emit(0)
            self.finished.emit()
            return

        self.is_evaluating = True

        # if not (path_results / 'software').is_dir():
        #     self.download_eval_software(path_results=path_results)

        self.text_output.emit(start_message)

        trainset_scores = {'model': [],
                           'th_cell': [],
                           'th_seed': [],
                           'aji+ (mean)': [],
                           'aji+ (std)': [],
                           'test set version': []}
        # Eval multiple models
        for i, model in enumerate(models):

            # Make dirs / clean dirs
            if not path_results.is_dir():
                path_results.mkdir()
            if not (path_results / "{}_{}".format(model.parent.stem, model.stem)).is_dir():
                (path_results / "{}_{}".format(model.parent.stem, model.stem)).mkdir()
            else:
                shutil.rmtree(path_results / "{}_{}".format(model.parent.stem, model.stem))
                (path_results / "{}_{}".format(model.parent.stem, model.stem)).mkdir()

            # Look for stop signal
            QCoreApplication.processEvents()  # Update to get stop signal
            if self.stop_evaluation:
                if self.is_evaluating:  # Stop event happened outside evaluation (print stop message)
                    self.text_output.emit("Stop evaluation due to user interaction.")
                break

            # Load model json file to get architecture + filters
            with open(model.parent / "{}.json".format(model.stem)) as f:
                model_settings = json.load(f)

            # Build CNN
            net = build_unet(unet_type=model_settings['architecture'][0],
                             act_fun=model_settings['architecture'][2],
                             pool_method=model_settings['architecture'][1],
                             normalization=model_settings['architecture'][3],
                             device=device,
                             num_gpus=num_gpus,
                             ch_in=3,
                             ch_out=1 if model_settings['label_type'] == 'distance' else 3,
                             filters=model_settings['architecture'][4])

            # Load weights
            net = get_weights(net=net, weights=str(model), num_gpus=num_gpus, device=device)

            # Get test set (new dataset needed
            test_dataset = InferenceDataset(data_dir=path_data / 'test',
                                            transform=pre_processing_transforms(apply_clahe=False, scale_factor=1))

            # Get thresholds to test depending on method
            if model_settings['label_type'] == 'distance':
                th_cell, th_seed = [0.05, 0.075, 0.10, 0.125], [0.35, 0.45]
                ths = list(product(th_cell, th_seed))
            else:
                ths = [-1]

            # Inference (save results for multiple thresholds at once)
            try:
                self.inference(net=net, dataset=test_dataset, label_type=model_settings['label_type'], ths=ths,
                               batch_size=batch_size, device=device,
                               path_model=path_results / "{}_{}".format(model.parent.stem, model.stem),
                               save_raw=save_raw_pred, eval_progress=(0.5/len(models), i/len(models)))
            except:
                text = "Please, try again with smaller batch size or reduce the crop size (use the export " \
                       "and import functionalities for this)"
                self.text_output.emit(text)
                self.text_output.emit('Stop evaluation due to memory problems')
                self.text_output.emit(text)
                self.finished.emit()
                return

            # Clear memory
            del net
            gc.collect()

            # Calculate scores (keep only best threshold results)
            results = self.calc_scores(prediction_path=path_results / "{}_{}".format(model.parent.stem, model.stem),
                                       test_set_path=path_data / 'test',
                                       label_type=model_settings['label_type'])

            if results:  # check if operation was aborted
                trainset_scores['model'].append("{}: {}".format(model.parent.stem, model.stem))
                trainset_scores['th_cell'].append(results[2])
                trainset_scores['th_seed'].append(results[3])
                trainset_scores['aji+ (mean)'].append(results[0])
                trainset_scores['aji+ (std)'].append(results[1])
                trainset_scores['test set version'].append(results[4])

                # zip test set and save into evaluation folder
                with zipfile.ZipFile(path_results / "{}_{}".format(model.parent.stem, model.stem) / 'test_set.zip', 'w') as z:
                    z.write(path_data, arcname=path_data.stem, compress_type=zipfile.ZIP_DEFLATED)
                    z.write(path_data / 'test', arcname=os.path.join(path_data.stem, 'test'),
                            compress_type=zipfile.ZIP_DEFLATED)
                    for file in (path_data / 'test').glob('*'):
                        z.write(file, arcname=os.path.join(path_data.stem, 'test', file.name),
                                compress_type=zipfile.ZIP_DEFLATED)

            # Update progress bar
            self.progress.emit(int(100 * (i + 1) / len(models)))

        if not self.stop_evaluation:
            # Convert to pandas dataframe
            trainset_scores_df = pd.DataFrame(trainset_scores)
            # Get existing scores
            if (path_results.parent / '{}.csv'.format(path_results.stem)).is_file():
                trainset_scores_old_df = pd.read_csv(path_results.parent / '{}.csv'.format(path_results.stem))
                # Delete evaluation scores on old test set (hash differs)
                trainset_scores_old_df = trainset_scores_old_df[
                    trainset_scores_old_df['test set version'] == trainset_scores_df.iloc[0]['test set version']]
                # Combine old and new scores without duplicates
                trainset_scores_df = trainset_scores_df.append(trainset_scores_old_df)
                # Delete possible duplicate keys due to rounding errors ...
                trainset_scores_df = trainset_scores_df.drop_duplicates('model')
            trainset_scores_df = trainset_scores_df.sort_values(by=['model'])
            trainset_scores_df.to_csv(path_results.parent / '{}.csv'.format(path_results.stem), header=True,
                                      index=False)
            self.progress.emit(100)
        self.finished.emit()

        return

    @pyqtSlot()
    def stop_evaluation_process(self):
        """ Set internal evaluation stop state to True

        :return: None
        """
        self.stop_evaluation = True

    # def download_eval_software(self, path_results):
    #
    #     self.text_output.emit('Download evaluation software from celltrackingchallenge.net')
    #     (path_results / 'software').mkdir()
    #     with requests.get('http://public.celltrackingchallenge.net/software/EvaluationSoftware.zip',
    #                       stream=True) as r:
    #         r.raise_for_status()
    #         with open(path_results / 'software' / 'EvaluationSoftware.zip', 'wb') as f:
    #             for chunk in r.iter_content(chunk_size=8192):
    #                 f.write(chunk)
    #     # Unzip evaluation software
    #     with zipfile.ZipFile(path_results / 'software' / 'EvaluationSoftware.zip', 'r') as z:
    #         z.extractall(path_results / 'software')
    #     # Remove zip
    #     os.remove(path_results / 'software' / 'EvaluationSoftware.zip')
    #     return

    def calc_scores(self, prediction_path, test_set_path,  label_type):
        """ Calculate metrics (aggregated Jaccard index).

        :param prediction_path: Path to the predictions.
        :type prediction_path: pathlib Path object.
        :param test_set_path: Path to the ground truths.
        :type test_set_path: pathlib Path object.
        :param label_type: Segmentation method / type of the labels.
        :type label_type: str
        :return: [mean score, std score] for boundary method, [mean score, std_score, [th_cell, th_seed]] for distance
        """

        if label_type == 'distance':

            score, score_std, th_cell, th_seed, best_sub_dir = 0, 0, 0, 0, ''

            for sub_dir in prediction_path.iterdir():

                sub_dir_scores = []
                file_names = []

                pred_ids = sub_dir.glob('mask*.tif')
                for pred_id in pred_ids:

                    # Look for stop signal
                    QCoreApplication.processEvents()  # Update to get stop signal
                    if self.stop_evaluation:
                        if self.is_evaluating:  # Stop event happened outside evaluation (print stop message)
                            self.text_output.emit("Stop metric calculation.")
                        return None

                    prediction = tiff.imread(str(pred_id))
                    ground_truth = tiff.imread(str(test_set_path / pred_id.name))

                    # Apply border correction
                    prediction = border_correction(prediction)
                    ground_truth = border_correction(ground_truth)
                    if np.max(prediction) > 0:
                        aji = get_fast_aji_plus(true=measure.label(ground_truth), pred=measure.label(prediction))
                    else:
                        aji = 0
                    sub_dir_scores.append(aji)
                    file_names.append(pred_id.stem)

                sub_dir_score = np.mean(sub_dir_scores)
                sub_dir_score_std = np.std(sub_dir_scores)

                if sub_dir_score > score:
                    score = sub_dir_score
                    score_std = sub_dir_score_std
                    th_cell = float(sub_dir.stem.split('_')[0])
                    th_seed = float(sub_dir.name.split('_')[-1])
                    scores = deepcopy(sub_dir_scores)
                    best_sub_dir = sub_dir.name

            for sub_dir in prediction_path.iterdir():
                if sub_dir.name == best_sub_dir:
                    for f in sub_dir.glob('*'):
                        shutil.move(f, sub_dir.parents[0] / f.name)
                shutil.rmtree(sub_dir)

            # Save scores
            results_df = pd.DataFrame({'test image': file_names, 'aji+': scores})
            results_df = results_df.sort_values(by=['test image'])
            results_df.to_csv(prediction_path / "scores.csv", header=True, index=False)

            return score, score_std, th_cell, th_seed, hashlib.sha1(str(file_names).encode("UTF-8")).hexdigest()[:10]

        else:

            scores, file_names = [], []

            pred_ids = prediction_path.glob('mask*.tif')
            for pred_id in pred_ids:

                # Look for stop signal
                QCoreApplication.processEvents()  # Update to get stop signal
                if self.stop_evaluation:
                    if self.is_evaluating:  # Stop event happened outside evaluation (print stop message)
                        self.text_output.emit("Stop metric calculation.")
                    return None

                prediction = tiff.imread(str(pred_id))
                ground_truth = tiff.imread(str(test_set_path / pred_id.name))

                # Apply border correction
                prediction = border_correction(prediction)
                ground_truth = border_correction(ground_truth)

                if np.max(prediction) > 0:
                    aji = get_fast_aji_plus(true=measure.label(ground_truth), pred=measure.label(prediction))
                else:
                    aji = 0
                scores.append(aji)
                file_names.append(pred_id.stem)

            score = np.mean(scores)
            score_std = np.std(scores)

            # Save scores
            results_df = pd.DataFrame({'test image': file_names, 'aji+': scores})
            results_df = results_df.sort_values(by=['test image'])
            results_df.to_csv(prediction_path / "scores.csv", header=True, index=False)

            return score, score_std, -1, -1, hashlib.sha1(str(file_names).encode("UTF-8")).hexdigest()[:10]

    def inference(self, net, dataset, label_type, ths, batch_size, device, path_model, save_raw, eval_progress):
        """ Train the model.
        :param net: Model/Network to use for inference.
        :type net:
        :param dataset: Pytorch dataset.
        :type dataset: torch dataset.
        :param label_type: segmentation method / type of the labels.
        :type label_type: str
        :param ths: thresholds to evaluate in post-processing (for distance method).
        :type ths: list
        :param batch_size: Batch size.
        :type batch_size: int
        :param device: Device to use (gpu or cpu).
        :type device: torch device
        :param path_model: Path to save the results of the selected model into.
        :type path_model: pathlib Path object
        :param save_raw: Save also raw cnn outputs.
        :type save_raw: bool.
        :param eval_progress: Progress of the evaluation (needed to update the progress bar properly)
        :type eval_progress: tuple
        :return: None
        """

        # Data loader for training and validation set
        if device.type == "cpu":
            num_workers = 0
        else:
            try:
                num_workers = cpu_count() // 2
            except AttributeError:
                num_workers = 4
        num_workers = np.minimum(num_workers, 16)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                                 num_workers=num_workers)

        net.eval()
        torch.set_grad_enabled(False)

        # Predict images (iterate over images/files)
        for i, sample in enumerate(dataloader):

            if i % 5 == 0:  # Check from time to time if evaluation should be stopped
                QCoreApplication.processEvents()
                if self.stop_evaluation:
                    self.text_output.emit("Stop evaluation due to user interaction.")
                    self.is_evaluating = False
                    return

            img_batch, ids_batch, pad_batch, img_size = sample
            img_batch = img_batch.to(device)

            if batch_size > 1:  # all images in a batch have same dimensions and pads
                pad_batch = [pad_batch[i][0] for i in range(len(pad_batch))]

            # Prediction
            if label_type == 'distance':
                prediction_border_batch, prediction_cell_batch = net(img_batch)
                # Get rid of pads
                prediction_cell_batch = prediction_cell_batch[:, 0, pad_batch[0]:, pad_batch[1]:, None].cpu().numpy()
                prediction_border_batch = prediction_border_batch[:, 0, pad_batch[0]:, pad_batch[1]:, None].cpu().numpy()
            else:
                prediction_batch = net(img_batch)
                prediction_batch = F.softmax(prediction_batch, dim=1)
                # Get rid of pads
                prediction_batch = prediction_batch[..., pad_batch[0]:, pad_batch[1]:].cpu().numpy()
                prediction_batch = np.transpose(prediction_batch, (0, 2, 3, 1))

            # Go through predicted batch and apply post-processing (not parallelized)
            for h in range(len(img_batch)):

                file_id = ids_batch[h].split('img')[-1]

                for th in ths:

                    if label_type == 'distance':
                        path_results = path_model / "{}_{}".format(th[0], th[1])
                        if not path_results.is_dir():
                            path_results.mkdir()
                        prediction = distance_postprocessing(border_prediction=np.copy(prediction_border_batch[h]),
                                                             cell_prediction=np.copy(prediction_cell_batch[h]),
                                                             th_cell=th[0],
                                                             th_seed=th[1])
                        if save_raw:
                            # Combine raw predictions
                            raw_pred = np.concatenate((prediction_cell_batch[h], prediction_border_batch[h]), axis=-1)
                            raw_pred = np.transpose(raw_pred, (2, 0, 1))
                            tiff.imwrite(str(path_results / "raw{}.tif".format(file_id)), raw_pred)
                    else:
                        path_results = path_model  # No thresholds need to be evaluated
                        prediction = boundary_postprocessing(prediction_batch[h])
                        if save_raw:
                            tiff.imwrite(str(path_results / "raw{}.tif".format(file_id)), prediction_batch[h])

                    tiff.imwrite(str(path_results / "mask{}.tif".format(file_id)), prediction)

            # Update progress bar
            self.progress.emit(int(100 * (eval_progress[0] * (i + 1) * batch_size / len(dataset) + eval_progress[1])))

        return
