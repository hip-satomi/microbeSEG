import json
import numpy as np

from copy import deepcopy
from omero.gateway import BlitzGateway
from omero.rtypes import rstring
from PyQt5.QtCore import pyqtSignal, QObject, pyqtSlot
from random import randint
from skimage.draw import polygon_perimeter

import torch
import torch.nn.functional as F

from src.inference.postprocessing import boundary_postprocessing, distance_postprocessing
from src.utils.unets import build_unet, get_weights
from src.utils.hull_polygon import get_indices_pandas, cv2_countour


class DataCropWorker(QObject):
    """ Worker class for training crop creation """
    finished = pyqtSignal()  # Signal when cropping is finished
    text_output = pyqtSignal(str)  # Signal for possible exceptions, e.g., too small crops
    crops = pyqtSignal(object)  # Signal for sending the crops
    stop_creation = False
    
    def __init__(self, img_list, crop_size, trainset_id, train_path, omero_username, omero_password, omero_host,
                 omero_port, group_id, pre_labeling=False, model=None, device=None, num_gpus=None, ths=None):
        """

        :param img_list: List with information about the images to crop
        :type img_list: list
        :param crop_size: Crop size to use for cropping
        :type crop_size: int
        :param trainset_id: OMERO id of the training dataset to use
        :type trainset_id: int
        :param train_path: Local path for training
        :type train_path: pathlib Path object
        :param omero_username: OMERO username
        :type omero_username: str
        :param omero_password: OMERO password
        :type omero_password: str
        :param omero_host: OMERO host
        :type omero_host: str
        :param omero_port: OMERO port
        :type omero_port: str
        :param group_id: Current user group id
        :type group_id: int
        :param pre_labeling: Apply pre-labeling
        :type pre_labeling: bool
        :param model: Model to use for pre-labeling
        :type model: pathlib Path Object
        :param device: Device to use (gpu or cpu)
        :type device: torch device
        :param num_gpus: Number of gpus to use
        :type num_gpus: int
        :param ths: Thresholds to use for pre-labeling
        :type ths: list
        """
        
        super().__init__()
        
        self.img_list = img_list
        self.crop_size = crop_size
        self.trainset_id = trainset_id
        self.train_path = train_path
        self.img_idx = 0
        self.crop = None
        self.pre_labeling = pre_labeling
        self.omero_username = omero_username
        self.omero_password = omero_password
        self.omero_host = omero_host
        self.omero_port = omero_port
        self.group_id = group_id
        self.conn = None

        if self.pre_labeling:
            self.device = device

            # Load model json file to get architecture + filters
            with open(model) as f:
                self.model_settings = json.load(f)
            # Build net
            self.net = build_unet(unet_type=self.model_settings['architecture'][0],
                                  act_fun=self.model_settings['architecture'][2],
                                  pool_method=self.model_settings['architecture'][1],
                                  normalization=self.model_settings['architecture'][3],
                                  device=self.device,
                                  num_gpus=num_gpus,
                                  ch_in=1,
                                  ch_out=1 if self.model_settings['label_type'] == 'distance' else 3,
                                  filters=self.model_settings['architecture'][4])
            self.net = get_weights(net=self.net,
                                   weights=str(model.parent / "{}.pth".format(model.stem)),
                                   num_gpus=num_gpus,
                                   device=device)
            if self.model_settings['label_type'] == 'distance':
                self.ths = ths
            else:
                self.ths = None

        # Connect to omero
        self.connect()

        # Load first crop --> blocks main thread if called during initialization
        # self.next_crop()

    def connect(self):
        """ Connect to OMERO server """
        self.conn = BlitzGateway(self.omero_username, self.omero_password,
                                 host=self.omero_host,
                                 port=self.omero_port,
                                 secure=True)
        self.conn.connect()

        if self.group_id is not None:
            self.conn.setGroupForSession(self.group_id)

    def disconnect(self):
        """ Disconnect from OMERO server"""
        try:
            self.conn.close()
        except:
            pass  # probably already closed / timeout

    def get_crop(self):
        """ Emit crop(s) """

        # Emit loaded crop
        self.crops.emit(deepcopy(self.crop))

        # Get next crop
        if self.img_idx == len(self.img_list):
            self.text_output.emit('No more crops can be found. Start process again to see not accepted crops.')
            self.crop_creation_finished()
        else:
            self.next_crop()

    def next_crop(self):
        """ Generate next crop(s) """

        # Try to get crops
        cropping = True
        while cropping and self.img_idx < len(self.img_list):

            # Load image from omero
            try:
                img = self.conn.getObject("Image", self.img_list[self.img_idx]['id'])
            except Exception as e:  # probably timeout  --> reconnect and try again
                self.disconnect()
                self.connect()
                img = self.conn.getObject("Image", self.img_list[self.img_idx]['id'])

            img = img.getPrimaryPixels().getPlane(0,
                                                  self.img_list[self.img_idx]['channel'],
                                                  self.img_list[self.img_idx]['frame'])

            # Check which dimension is larger --> crop_dimension
            if img.shape[0] > img.shape[1]:
                crop_dim = 0
            else:
                crop_dim = 1

            # Check if 3 crops are possible
            if img.shape[crop_dim] > 3 * self.crop_size:
                n_crops = 3
            elif img.shape[crop_dim] > 2 * self.crop_size:
                n_crops = 2
            else:
                n_crops = 1

            # Get frame_min and frame_max before padding/cropping
            img_min, img_max, img_mean, img_std = np.min(img), np.max(img), np.mean(img), np.std(img)

            # Check if padding is needed
            if 0.9 * self.crop_size > img.shape[0] or 0.9 * self.crop_size > img.shape[1]:  # Skip too small images
                self.img_idx += 1
                continue
            x_pads = np.maximum(0, self.crop_size - img.shape[1])
            y_pads = np.maximum(0, self.crop_size - img.shape[0])
            img = np.pad(img, ((0, y_pads), (0, x_pads)), mode='constant', constant_values=img_min)

            # Get crop starting points
            crops = []
            for i in range(n_crops):

                c = img.shape[crop_dim] // n_crops

                if x_pads > 0 and x_pads > 0:
                    a, b = 0, 0
                elif crop_dim == 0 and y_pads == 0 and img.shape[crop_dim] > self.crop_size:
                    a = randint(i * c,
                                np.minimum(img.shape[crop_dim] - self.crop_size, (i + 1) * c - self.crop_size))
                    b = randint(0, img.shape[1] - self.crop_size)
                elif crop_dim == 1 and x_pads == 0 and img.shape[crop_dim] > self.crop_size:
                    a = randint(0, img.shape[0] - self.crop_size)
                    b = randint(i * c,
                                np.minimum(img.shape[crop_dim] - self.crop_size, (i + 1) * c - self.crop_size))
                else:
                    a, b = 0, 0

                # Crop
                crop = img[a:a + self.crop_size, b:b + self.crop_size]

                # In gui shown crop
                crop_show = 255 * (crop.astype(np.float32) - img_min) / (img_max - img_min)
                crop_show = crop_show.astype(np.uint8)

                if self.pre_labeling:

                    # Predict crop
                    prediction = self.inference(crop, min_val=img_min, max_val=img_max)

                    # Get polygon coordinates for uploading --> roi
                    roi_show = np.concatenate((crop_show[..., None], crop_show[..., None], crop_show[..., None]),
                                              axis=-1)
                    # Create polygon ROIs for each cell
                    prediction_ids = get_indices_pandas(prediction)
                    roi = []
                    if np.max(prediction) > 0:
                        try:
                            outlines = np.zeros_like(crop, dtype=bool)
                            for m_key, prediction_idx in prediction_ids.items():
                                try:
                                    polygon_point_list = cv2_countour(prediction_idx)
                                except AssertionError:
                                    continue
                                for polygon_points in polygon_point_list:
                                    points = ""
                                    for col_idx in range(polygon_points.shape[1]):
                                        points += "{},{} ".format(polygon_points[1, col_idx], polygon_points[0, col_idx])
                                    roi.append(rstring(points))
                                    cell_perimeter = polygon_perimeter(polygon_points[0], polygon_points[1],
                                                                       shape=(self.crop_size, self.crop_size),
                                                                       clip=True)
                                    outlines[cell_perimeter[0], cell_perimeter[1]] = True

                            # Create overlay to show in gui --> roi_show
                            roi_show[outlines, 0] = 255
                            roi_show[outlines, 1] = 255
                            roi_show[outlines, 2] = 0
                        except:
                            pass
                else:
                    roi, roi_show = None, None

                # Add info and crops as dict to list
                crops.append({'project': str(self.img_list[self.img_idx]['project']),
                              'dataset': str(self.img_list[self.img_idx]['dataset']),
                              'image': str(self.img_list[self.img_idx]['name']),
                              'image_id': self.img_list[self.img_idx]['id'],  # convert to string later --> 'used' key
                              'crop_size': str(self.crop_size),
                              'channel': self.img_list[self.img_idx]['channel'],   # convert to string later
                              'frame': self.img_list[self.img_idx]['frame'],   # convert to string later --> 'used' key
                              'max_frame': str(img_max),
                              'mean_frame': str(img_mean),
                              'min_frame': str(img_min),
                              'std_frame': str(img_std),
                              'pre_labeled': str(False),  # Set to true when segmentation has been selected
                              'x_start': str(b),
                              'y_start': str(a),
                              'img': img[a:a + self.crop_size, b:b + self.crop_size],
                              'img_show': crop_show,
                              'roi': roi,
                              'roi_show': roi_show})

            self.img_idx += 1
            cropping = False
            self.crop = crops

    def inference(self, crop, min_val, max_val):
        """ Prediction of crops for pre-labeling

        :param crop: Crop to predict
        :type crop:
        :param min_val: Minimum image/frame value for normalization
        :type min_val: int
        :param max_val: Maximum image/frame value for normalization
        :type max_val: int
        :return: prediction
        """

        self.net.eval()
        torch.set_grad_enabled(False)

        # Normalize crop and convert to tensor / img_batch
        img_batch = 2 * (crop.astype(np.float32) - min_val) / (max_val - min_val) - 1
        img_batch = torch.from_numpy(img_batch[None, None, :, :]).to(torch.float)
        img_batch = img_batch.to(self.device)

        # Prediction
        if self.model_settings['label_type'] == 'distance':
            try:
                prediction_border_batch, prediction_cell_batch = self.net(img_batch)
            except RuntimeError:
                prediction = np.zeros_like(crop, dtype=np.uint16)
                self.text_output.emit('RuntimeError during inference (maybe not enough ram/vram?)')
            else:
                prediction_cell_batch = prediction_cell_batch[0, 0, ..., None].cpu().numpy()
                prediction_border_batch = prediction_border_batch[0, 0, ..., None].cpu().numpy()
                prediction = distance_postprocessing(border_prediction=np.copy(prediction_border_batch),
                                                     cell_prediction=np.copy(prediction_cell_batch),
                                                     th_cell=self.ths[0],
                                                     th_seed=self.ths[1])
        else:
            try:
                prediction_batch = self.net(img_batch)
            except RuntimeError:
                prediction = np.zeros_like(crop, dtype=np.uint16)
                self.text_output.emit('RuntimeError during inference (maybe not enough ram/vram?)')
            else:
                prediction_batch = F.softmax(prediction_batch, dim=1)
                prediction_batch = prediction_batch.cpu().numpy()
                prediction_batch = np.transpose(prediction_batch[0], (1, 2, 0))
                prediction = boundary_postprocessing(prediction_batch)

        return prediction

    @pyqtSlot()
    def crop_creation_finished(self):
        """ Close connection and send finished signal """
        self.disconnect()
        self.finished.emit()

    @pyqtSlot()
    def stop_crop_process(self):
        """ Set internal stop state to True

        :return: None
        """
        self.stop_creation = True
