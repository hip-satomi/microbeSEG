import gc
import numpy as np
import os
import random
import tifffile as tiff
import time
import torch
import torch.optim as optim
import zipfile

from multiprocessing import cpu_count
from PyQt5.QtCore import pyqtSignal, QObject, pyqtSlot, QCoreApplication
from shutil import rmtree
from skimage.measure import regionprops
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from src.training.training_dataset import TrainingDataset
from src.training.losses import get_loss
from src.training.ranger2020 import Ranger
from src.training.train_data_representations import get_label
from src.training.mytransforms import augmentors
from src.utils.utils import unique_path, write_train_info
from src.utils.unets import build_unet, get_weights


class CreateLabelsWorker(QObject):
    """ Worker class for label creation """
    finished = pyqtSignal()  # Signal when import is finished
    progress = pyqtSignal(int)  # Signal for updating the progress bar
    text_output = pyqtSignal(str)  # Signal for possible exceptions, e.g., user interaction to stop label creation
    stop_label_creation = False  # Stop worker

    def create_labels(self, path, label_type):
        """ Create labels out of ground truth masks

        :param path: Path to the training set
        :type path: pathlib Path object
        :param label_type: Type of the label / segmentation method
        :type label_type: str
        :return: None
        """

        # Check if export has been stopped (folders are deleted)
        if len(list(path.glob('*'))) == 0:
            self.progress.emit(0)
            self.finished.emit()
            return

        self.text_output.emit('Create labels')

        # Get mask ids
        mask_ids_train = list((path / 'train').glob('mask*.tif'))
        mask_ids_val = list((path / 'val').glob('mask*.tif'))
        if len(mask_ids_val) < 2 or len(mask_ids_train) < 2:
            self.text_output.emit("The training and the validation set should each contain at least two annotated "
                                  "images! Stop")
            self.progress.emit(0)
            self.finished.emit()
            return
        mask_ids = mask_ids_train + mask_ids_val

        # Go through mask ids
        for i, mask_id in enumerate(mask_ids):

            QCoreApplication.processEvents()  # Update to get stop signal
            if self.stop_label_creation:
                self.text_output.emit("Stop label creation due to user interaction.\nDelete local folder.")
                rmtree(str(path))
                break

            mask = tiff.imread(str(mask_id))

            # Get major axis length information and calculate label
            if label_type == 'distance':
                major_axes = []
                props = regionprops(mask)
                for cell in props:
                    major_axes.append(cell.major_axis_length)
                max_mal = int(np.ceil(np.max(np.array(major_axes))))
            else:
                max_mal = 0  # not needed

            # Calculate labels
            label = get_label(mask=mask, label_type=label_type, max_mal=max_mal)

            # Save labels (check label type for dtype)
            fname = mask_id.name.split('mask_')[-1]

            if label_type == 'distance':
                tiff.imwrite(str(mask_id.parent / 'cell_dist_{}'.format(fname)), label[0])
                tiff.imwrite(str(mask_id.parent / 'neighbor_dist_{}'.format(fname)), label[1])
            else:
                tiff.imwrite(str(mask_id.parent / '{}_{}'.format(label_type, fname)), label)

            # Update progress bar
            self.progress.emit(int(100 * (i + 1) / len(mask_ids)))

        if self.stop_label_creation:
            self.progress.emit(0)
        else:
            self.progress.emit(100)
        self.finished.emit()

        return

    @pyqtSlot()
    def stop_label_creation_process(self):
        """ Set internal export stop state to True

        :return: None
        """
        self.stop_label_creation = True


class TrainWorker(QObject):
    """ Worker class for model training """
    finished = pyqtSignal()  # Signal when import is finished
    progress = pyqtSignal(int)  # Signal for updating the progress bar
    text_output_main_gui = pyqtSignal(str)  # Signal for possible exceptions, e.g., user interaction to stop export
    text_output = pyqtSignal(str)  # Signal for reporting the training progress
    stop_training = False  # Stop training process
    is_training = False  # State of training process

    def start_training(self, path_data, path_models, label_type, iterations, optimizer, batch_size, device, num_gpus):
        """ Start training process

        :param path_data: Path of the training set (contains dirs 'train', and 'val')
        :type path_data: pathlib Path object
        :param path_models: Path to save the trained models into
        :type path_models: pathlib Path object
        :param label_type: Segmentation method / label type
        :type label_type: str
        :param iterations: Number of models to train
        :type iterations: int
        :param optimizer: Optimizer to use (ranger or adam)
        :type optimizer: str
        :param batch_size: Batch size
        :type batch_size: int
        :param device: Device to use (gpu or cpu)
        :type device: torch device
        :param num_gpus: Number of gpus to use
        :type num_gpus: int
        :return: None
        """

        # Check if export has been stopped (folders are deleted)
        if len(list(path_data.glob('*'))) == 0 or len(list((path_data / 'train').glob('mask*'))) < 2 \
                or len(list((path_data / 'val').glob('mask*'))) < 2:
            self.progress.emit(0)
            self.finished.emit()
            return

        # Send text message for training start
        self.text_output_main_gui.emit('Start training')
        self.is_training = True

        # Train multiple models
        for i in range(iterations):

            # Look for stop signal
            QCoreApplication.processEvents()  # Update to get stop signal
            if self.stop_training:
                if self.is_training:  # Stop event happened outside train method (print stop message)
                    self.text_output_main_gui.emit("Stop training due to user interaction.")
                break

            # Get model name
            run_name = unique_path(path_models, label_type + '_model_{:02d}.pth').stem

            # Get activation function
            act_fun = 'mish' if optimizer == 'ranger' else 'relu'

            if label_type in ['boundary', 'distance']:

                filters = [64, 1024]  # Number of kernels (start and maximum)

                try_training = True
                while try_training:
                    try:
                        # Get CNN settings
                        train_configs = {'architecture': ('DU' if label_type == 'distance' else 'U',
                                                          "conv",
                                                          act_fun,
                                                          'bn',
                                                          filters),
                                         'batch_size': batch_size,
                                         'label_type': label_type,
                                         'loss': 'smooth_l1' if label_type == 'distance' else 'ce_dice',
                                         'num_gpus': num_gpus,
                                         'optimizer': optimizer,
                                         'run_name': run_name}

                        # Build CNN
                        net = build_unet(unet_type=train_configs['architecture'][0],
                                         act_fun=train_configs['architecture'][2],
                                         pool_method=train_configs['architecture'][1],
                                         normalization=train_configs['architecture'][3],
                                         device=device,
                                         num_gpus=num_gpus,
                                         ch_in=1,
                                         ch_out=1 if label_type == 'distance' else 3,
                                         filters=train_configs['architecture'][4])

                        # Load training and validation set
                        data_transforms = augmentors(label_type=train_configs['label_type'],
                                                     min_value=0,  # data are already normalized
                                                     max_value=65535)
                        train_configs['data_transforms'] = str(data_transforms)
                        datasets = {x: TrainingDataset(root_dir=path_data,
                                                       label_type=label_type,
                                                       mode=x,
                                                       transform=data_transforms[x])
                                    for x in ['train', 'val']}

                        # Get number of training epochs depending on dataset size (roughly to decrease training time):
                        crop_size = tiff.imread(list((path_data / 'train').glob('*.tif'))[0]).shape[0]
                        train_configs['max_epochs'] = get_max_epochs(len(datasets['train']) + len(datasets['val']),
                                                                     crop_size=crop_size)

                        # Train model
                        best_loss = self.train(net=net, datasets=datasets, configs=train_configs, device=device,
                                               path_models=path_models, train_progress=(1/iterations, i/iterations))

                        # Fine-tune with cosine annealing for Ranger models
                        if train_configs['optimizer'] == 'ranger' and self.is_training:
                            # Build model again
                            net = build_unet(unet_type=train_configs['architecture'][0],
                                             act_fun=train_configs['architecture'][2],
                                             pool_method=train_configs['architecture'][1],
                                             normalization=train_configs['architecture'][3],
                                             device=device,
                                             num_gpus=num_gpus,
                                             ch_in=1,
                                             ch_out=1 if label_type == 'distance' else 3,
                                             filters=train_configs['architecture'][4])
                            # Get best weights as starting point
                            net = get_weights(net=net, weights=str(path_models / '{}.pth'.format(run_name)),
                                              num_gpus=num_gpus, device=device)

                            # Train further
                            _ = self.train(net=net, datasets=datasets, configs=train_configs, device=device,
                                           path_models=path_models, best_loss=best_loss,
                                           train_progress=(1/iterations, (0.9+i)/iterations))

                        try_training = False

                        if self.is_training:

                            # Update progress bar (training could be stopped early)
                            self.progress.emit(int(100 * (i + 1) / iterations))

                            # Write information to json-file
                            write_train_info(configs=train_configs, path=path_models)

                            # Pack training set into zip (only if not interrupted)
                            with zipfile.ZipFile(path_models / '{}_trainset.zip'.format(run_name), 'w') as z:
                                z.write(path_data, arcname=path_data.stem, compress_type=zipfile.ZIP_DEFLATED)
                                for sub_dir in path_data.iterdir():
                                    if sub_dir.stem == 'test':
                                        continue
                                    z.write(sub_dir, arcname=os.path.join(path_data.stem, sub_dir.stem),
                                            compress_type=zipfile.ZIP_DEFLATED)
                                    for file in sub_dir.glob('*'):
                                        z.write(file, arcname=os.path.join(path_data.stem, sub_dir.stem, file.name),
                                                compress_type=zipfile.ZIP_DEFLATED)

                    except RuntimeError:  # out of memory
                        if batch_size > 8:
                            text = "Model does not fit on RAM/VRAM. Reduce batch size from {} to 8".format(batch_size)
                            batch_size = 8
                        elif batch_size > 4:
                            text = "Model does not fit on RAM/VRAM. Reduce batch size from {} to 4".format(batch_size)
                            batch_size = 4
                        elif filters[0] > 32:
                            text = "Model does not fit on RAM/VRAM. Reduce number of kernels"
                            filters = [32, 512]
                        elif filters[-1] == 512:
                            text = "Model does not fit on RAM/VRAM. Reduce model depth"
                        else:
                            text = "Please, try again with smaller batch size or reduce the crop size (use the export " \
                                   "and import functionalities for this)"
                            self.text_output_main_gui.emit(text)
                            self.text_output.emit('Stop training due to memory problems')
                            try_training = False
                        self.text_output_main_gui.emit(text)

            else:
                continue

        if not self.stop_training:
            self.progress.emit(100)
        self.finished.emit()

        return

    @pyqtSlot()
    def stop_training_process(self):
        """ Set internal training stop state to True

        :return: None
        """
        self.stop_training = True

    def train(self, net, datasets, configs, device, path_models, train_progress, best_loss=1e4):
        """ Train the model.

        :param net: Model/Network to train
        :type net:
        :param datasets: Dictionary containing the training and the validation data set
        :type datasets: dict
        :param configs: Dictionary with configurations of the training process
        :type configs: dict
        :param device: Use (multiple) GPUs or CPU
        :type device: torch device
        :param path_models: Path to the directory to save the models
        :type path_models: pathlib Path object
        :param train_progress: Progress of the training (to update progress bar correctly)
        :type train_progress: tuple
        :param best_loss: Best loss (only needed for second run to see if val loss further improves)
        :type best_loss: float
        :return: None
        """

        if best_loss < 1e3:  # second Ranger run
            second_run = True
            self.text_output.emit('Start 2nd run with cosine annealing')
        else:
            second_run = False
            self.text_output.emit('-' * 10)
            self.text_output.emit('{}'.format(configs['run_name']))
            self.text_output.emit('-' * 10)
            self.text_output.emit('Train/validate on {}/{} images'.format(len(datasets['train']), len(datasets['val'])))

        # Data loader for training and validation set
        apply_shuffling = {'train': True, 'val': False}
        if device.type == "cpu":
            num_workers = 0
        else:
            try:
                num_workers = cpu_count() // 2
            except AttributeError:
                num_workers = 4
        num_workers = np.minimum(num_workers, 16)
        dataloader = {x: torch.utils.data.DataLoader(datasets[x],
                                                     batch_size=configs['batch_size'],
                                                     shuffle=apply_shuffling[x],
                                                     pin_memory=True,
                                                     worker_init_fn=seed_worker,
                                                     num_workers=num_workers)
                      for x in ['train', 'val']}

        # Loss function and optimizer
        criterion = get_loss(configs['loss'], label_type=configs['label_type'])

        max_epochs = configs['max_epochs']

        # Optimizer
        if configs['optimizer'] == 'adam':
            optimizer = optim.Adam(net.parameters(),
                                   lr=8e-4,
                                   betas=(0.9, 0.999),
                                   eps=1e-08,
                                   weight_decay=0,
                                   amsgrad=True)
            scheduler = ReduceLROnPlateau(optimizer,
                                          mode='min',
                                          factor=0.25,
                                          patience=configs['max_epochs'] // 20,
                                          verbose=False,
                                          min_lr=3e-6)
            break_condition = 2 * configs['max_epochs'] // 20 + 5

        elif configs['optimizer'] == 'ranger':

            lr = 6e-3
            if second_run:

                optimizer = Ranger(net.parameters(),
                                   lr=0.09 * lr,
                                   alpha=0.5, k=6, N_sma_threshhold=5,  # Ranger options
                                   betas=(.95, 0.999), eps=1e-6, weight_decay=0,  # Adam options
                                   # Gradient centralization on or off, applied to conv layers only or conv + fc layers
                                   use_gc=True, gc_conv_only=False, gc_loc=True)

                scheduler = CosineAnnealingLR(optimizer,
                                              T_max=configs['max_epochs'] // 10,
                                              eta_min=3e-5,
                                              last_epoch=-1,
                                              verbose=False)
                break_condition = configs['max_epochs'] // 10 + 1
                max_epochs = configs['max_epochs'] // 10
            else:
                optimizer = Ranger(net.parameters(),
                                   lr=lr,
                                   alpha=0.5, k=6, N_sma_threshhold=5,  # Ranger options
                                   betas=(.95, 0.999), eps=1e-6, weight_decay=0,  # Adam options
                                   # Gradient centralization on or off, applied to conv layers only or conv + fc layers
                                   use_gc=True, gc_conv_only=False, gc_loc=True)
                scheduler = ReduceLROnPlateau(optimizer,
                                              mode='min',
                                              factor=0.25,
                                              patience=configs['max_epochs'] // 10,
                                              verbose=False,
                                              min_lr=0.075 * lr)
                break_condition = 2 * configs['max_epochs'] // 10 + 5
        else:
            raise Exception('Optimizer not known')

        # Auxiliary variables for training process
        epochs_wo_improvement, train_loss, val_loss, = 0, [], []
        since = time.time()

        # Training process
        for epoch in range(max_epochs):

            QCoreApplication.processEvents()
            if self.stop_training:
                self.text_output_main_gui.emit("Stop training due to user interaction.\nRemove last model.")
                self.text_output.emit("Stop training due to user interaction.")
                try:
                    os.remove(str(path_models / "{}.pth".format(configs['run_name'])))
                except FileNotFoundError:
                    pass
                self.is_training = False
                break

            start = time.time()

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    net.train()  # Set model to training mode
                else:
                    net.eval()  # Set model to evaluation mode

                running_loss = 0.0

                # Iterate over data
                for samples in dataloader[phase]:

                    # Get img_batch and label_batch and put them on GPU if available
                    if configs['label_type'] == 'distance':
                        img_batch, border_label_batch, cell_label_batch = samples
                        img_batch = img_batch.to(device)
                        cell_label_batch = cell_label_batch.to(device)
                        border_label_batch = border_label_batch.to(device)
                    else:
                        img_batch, label_batch = samples
                        img_batch = img_batch.to(device)
                        label_batch = label_batch.to(device)

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass (track history if only in train)
                    with torch.set_grad_enabled(phase == 'train'):
                        if configs['label_type'] == 'distance':
                            border_pred_batch, cell_pred_batch = net(img_batch)
                            loss_border = criterion['border'](border_pred_batch, border_label_batch)
                            loss_cell = criterion['cell'](cell_pred_batch, cell_label_batch)
                            loss = loss_border + loss_cell
                        else:
                            pred_batch = net(img_batch)
                            loss = criterion(pred_batch, label_batch)

                        # Backward (optimize only if in training phase)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # Statistics
                    running_loss += float(loss.item() * img_batch.size(0))

                epoch_loss = running_loss / len(datasets[phase])

                if phase == 'train':
                    train_loss.append(epoch_loss)
                else:
                    val_loss.append(epoch_loss)

                    if epoch_loss < best_loss:
                        train_output = '{} / {}: Loss train / val: {:.4f} / {:.4f} --> save'.format(epoch+1,
                                                                                                    max_epochs,
                                                                                                    train_loss[-1],
                                                                                                    epoch_loss,
                                                                                                    best_loss)
                        best_loss = epoch_loss

                        # The state dict of data parallel (multi GPU) models need to get saved in a way that allows to
                        # load them also on single GPU or CPU
                        if configs['num_gpus'] > 1:
                            torch.save(net.module.state_dict(), str(path_models / (configs['run_name'] + '.pth')))
                        else:
                            torch.save(net.state_dict(), str(path_models / (configs['run_name'] + '.pth')))
                        epochs_wo_improvement = 0

                    else:
                        train_output = '{} / {}: Loss train / val: {:.4f} / {:.4f}'.format(epoch+1,
                                                                                           max_epochs,
                                                                                           train_loss[-1],
                                                                                           epoch_loss)
                        epochs_wo_improvement += 1

                    self.text_output.emit(train_output)

                    if configs['optimizer'] == 'ranger' and second_run:
                        scheduler.step()

                    else:
                        scheduler.step(epoch_loss)

            if configs['optimizer'] == 'ranger':
                if not second_run:
                    self.progress.emit(
                        int(100 * (epoch + 1) / (1.1 * max_epochs) * train_progress[0] + 100 * train_progress[1]))
                else:
                    self.progress.emit(
                        int(100 * (epoch + 1) / (10 * max_epochs) * train_progress[0] + 100 * train_progress[1]))
            else:
                self.progress.emit(int(100 * (epoch + 1) / max_epochs * train_progress[0] + 100 * train_progress[1]))

            # Break training if plateau is reached
            if epochs_wo_improvement == break_condition:
                self.text_output.emit('{} epochs without val loss improvement --> break'.format(epochs_wo_improvement))
                break

        # Total training time
        if not self.stop_training:
            time_elapsed = time.time() - since
            self.text_output.emit('Training completed in {:.0f}min {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

            # Save loss
            stats = np.transpose(np.array([list(range(1, len(train_loss) + 1)), train_loss, val_loss]))
            if second_run:
                f = open(str(path_models / (configs['run_name'] + '_loss.txt')), 'a')
                f.write('\n')
                np.savetxt(f, X=stats, fmt=['%3i', '%2.5f', '%2.5f'], delimiter=',')
                f.close()
                configs['training_time_run_2'], configs['trained_epochs_run2'] = time_elapsed, epoch + 1
            else:
                np.savetxt(fname=str(path_models / (configs['run_name'] + '_loss.txt')), X=stats,
                           fmt=['%3i', '%2.5f', '%2.5f'],
                           header='Epoch, training loss, validation loss', delimiter=',')
                configs['training_time'], configs['trained_epochs'] = time_elapsed, epoch + 1

        # Clear memory
        del net, loss, optimizer, scheduler
        gc.collect()

        return best_loss


def get_max_epochs(n_samples, crop_size):
    """ Get maximum amount of training epochs.

    :param n_samples: number of training samples
    :type n_samples: int
    :param crop_size: Size of the crops / training images
    :type crop_size: int
    :return: maximum amount of training epochs
    """

    # Some heuristics (made for 320x320 px crops)
    if n_samples >= 1000:
        max_epochs = 200
    elif n_samples >= 500:
        max_epochs = 240
    elif n_samples >= 200:
        max_epochs = 320
    elif n_samples >= 100:
        max_epochs = 400
    elif n_samples >= 50:
        max_epochs = 480
    else:
        max_epochs = 560

    max_epochs *= np.sqrt(320 / crop_size)  # scale a bit for larger/smaller crops (initially thought for 320x320)
    max_epochs = int(max_epochs - max_epochs % 20)

    return max_epochs


def seed_worker(worker_id):
    """ Fix pytorch seeds on linux

    https://pytorch.org/docs/stable/notes/randomness.html
    https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/

    :param worker_id:
    :return:
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
