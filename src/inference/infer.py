import json
import numpy as np
import omero.model
import tifffile as tiff
import torch
import torch.nn.functional as F

from copy import copy
from datetime import datetime
from omero.constants import metadata
from omero.gateway import BlitzGateway, MapAnnotationWrapper
from omero.rtypes import rstring, rint
from PyQt5.QtCore import pyqtSignal, QObject, pyqtSlot, QCoreApplication

from src.inference.postprocessing import boundary_postprocessing, distance_postprocessing
from src.utils.unets import build_unet, get_weights
from src.utils.data_import import create_roi
from src.utils.hull_polygon import get_indices_pandas, cv2_countour
from src.utils.utils import zero_pad_model_input


class InferWorker(QObject):
    """ Worker class for training crop creation """
    finished = pyqtSignal()  # Signal when inference is finished
    text_output = pyqtSignal(str)  # Signal for reporting which file is processed
    progress = pyqtSignal(int)  # Signal for updating the progress bar

    stop_inference = False

    def __init__(self, img_id_list, inference_path, omero_username, omero_password, omero_host, omero_port, group_id,
                 model, device, ths, upload=True, overwrite=True, sliding_window=False):
        """

        :param img_id_list: List of omero image ids
        :type img_id_list: list
        :param inference_path: Local results path
        :type inference_path: pathlib Path object
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
        :param model: Model to use for inference
        :type model: pathlib Path object
        :param device: Device to use (gpu or cpu)
        :type device: torch device
        :param ths: Thresholds for the post-processing
        :type ths: list
        :param upload: Upload results to OMERO
        :type upload: bool
        :param overwrite: Overwrite results on OMERO
        :type overwrite: bool
        :param sliding_window: Use sliding window for inference (not implemented yet)
        :type sliding_window: bool
        """

        super().__init__()

        self.img_id_list = img_id_list
        self.omero_username = omero_username
        self.omero_password = omero_password
        self.omero_host = omero_host
        self.omero_port = omero_port
        self.conn = None
        self.inference_path = inference_path  # path for local results
        self.channel = 'rgb'
        self.upload = upload
        self.overwrite = overwrite
        self.sliding_window = sliding_window
        self.device = device
        self.group_id = group_id
        self.net = None

        # Load model json file to get architecture + filters
        with open(model) as f:
            self.model_settings = json.load(f)
        self.model = model
        self.model_name = "{}: {}".format(model.parent.stem, model.stem)

        if self.model_settings['label_type'] == 'distance':
            self.ths = ths
        else:
            self.ths = None

        # Connect to omero
        self.connect()

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

    def start_inference(self):
        """ Start inference """

        self.text_output.emit('\nInference')

        # Build net
        self.net = build_unet(unet_type=self.model_settings['architecture'][0],
                              act_fun=self.model_settings['architecture'][2],
                              pool_method=self.model_settings['architecture'][1],
                              normalization=self.model_settings['architecture'][3],
                              device=self.device,
                              num_gpus=1,  # Only batch size 1 is used at the time, so makes no sense to use more gpus
                              ch_in=3 if self.channel == 'rgb' else 1,
                              ch_out=1 if self.model_settings['label_type'] == 'distance' else 3,
                              filters=self.model_settings['architecture'][4])

        # Load weights
        self.net = get_weights(net=self.net,
                               weights=str(self.model.parent / "{}.pth".format(self.model.stem)),
                               num_gpus=1,
                               device=self.device)

        for i, img_id in enumerate(self.img_id_list):

            # Load image from omero
            try:
                img_ome = self.conn.getObject("Image", img_id)
            except Exception as e:  # probably timeout  --> reconnect and try again
                self.disconnect()
                self.connect()
                img_ome = self.conn.getObject("Image", img_id)

            # When using copy & paste in omero, only links are created and the parent project may be not clear
            parents = img_ome.listParents()
            if len(parents) > 1:
                parent_projects = []
                for pp in parents:
                    if isinstance(pp.getParent(), omero.gateway.ProjectWrapper):
                        parent_projects.append(pp.getParent().getName())
                    else:
                        parent_projects.append(pp.getName())
            else:
                parent_projects = img_ome.getProject().getName()

            if self.upload and not img_ome.canAnnotate():
                self.text_output.emit(f'  Skip {parent_projects}: {img_ome.getName()} (no write permission)')
                continue

            # Check if z stack
            if img_ome.getSizeZ() > 1:
                self.text_output.emit(f'  Skip {parent_projects}: {img_ome.getName()} (is z-stack)')
                continue

            # Get image from Omero
            if img_ome.getSizeC() != 3:
                self.text_output.emit(f'  Skip {parent_projects}: {img_ome.getName()} (no rgb image)')
                continue

            # Check if results exist and should not be overwritten
            already_processed = False
            if self.upload:  # Check if annotation is available --> already processed once
                for ann in img_ome.listAnnotations():
                    if ann.OMERO_TYPE == omero.model.MapAnnotationI:
                        keys_values = ann.getValue()
                        for key, value in keys_values:
                            if key in ['inference_model', 'inference_date']:
                                already_processed = True
                if self.overwrite:  # Delete possible annotations
                    roi_service = self.conn.getRoiService()
                    result = roi_service.findByImage(img_ome.getId(), None)
                    roi_ids = [roi.id.val for roi in result.rois if type(roi.getShape(0)) == omero.model.PolygonI]
                    if roi_ids:
                        self.conn.deleteObjects("Roi", roi_ids)

                    # Delete attachments from label tool and analysis
                    to_delete = []
                    for ann in img_ome.listAnnotations(ns='microbeseg.analysis.namespace'):
                        to_delete.append(ann.getId())
                    for ann in img_ome.listAnnotations():
                        if ann.OMERO_TYPE == omero.model.FileAnnotationI:
                            if ann.getFileName() in ['simpleSegmentation.json', 'GUISegmentation.json']:
                                to_delete.append(ann.getId())
                    if to_delete:
                        self.conn.deleteObjects('Annotation', to_delete, wait=True)

            else:  # Check if local results are available
                if isinstance(parent_projects, list):
                    result_path = self.inference_path / parent_projects[0]
                else:
                    result_path = self.inference_path / img_ome.getProject().getName()
                fname = result_path / img_ome.getName()
                if (result_path / f"{fname.stem}_channel{self.channel}.tif").is_file():
                    already_processed = True

            if already_processed and not self.overwrite:
                self.text_output.emit(f'  Skip {parent_projects}: {img_ome.getName()} (already processed and '
                                      f'overwriting not enabled)')
                continue

            if not self.upload:
                if isinstance(parent_projects, list):
                    result_path = self.inference_path / parent_projects[0]
                else:
                    result_path = self.inference_path / img_ome.getProject().getName()
                result_path.mkdir(exist_ok=True)

            # Pre-allocate array for results
            results_array = np.zeros(shape=(img_ome.getSizeT(), img_ome.getSizeY(), img_ome.getSizeX()),
                                     dtype=np.uint16)

            # Go through frames
            for frame in range(img_ome.getSizeT()):

                # Check for stop signal
                QCoreApplication.processEvents()  # Update to get stop signal
                if self.stop_inference:
                    self.text_output.emit("Stop inference due to user interaction.")
                    self.progress.emit(0)
                    self.disconnect()
                    self.finished.emit()
                    return

                # Get image from Omero
                img = np.zeros(shape=(img_ome.getSizeY(), img_ome.getSizeX(), 3))
                zct_list = []
                for z in range(img_ome.getSizeZ()):  # all slices (1 anyway)
                    for c in range(img_ome.getSizeC()):  # all channels
                        zct_list.append((z, c, frame))
                for h, plane in enumerate(img_ome.getPrimaryPixels().getPlanes(zct_list)):
                    img[:, :, zct_list[h][1]] = plane

                # Get frame_min and frame_max before padding/cropping
                img_min, img_max, img_mean, img_std = np.min(img), np.max(img), np.mean(img), np.std(img)

                # Zero padding
                img, pads = zero_pad_model_input(img, pad_val=img_min)

                # Predict crop
                prediction = self.inference(img, min_val=img_min, max_val=img_max, pads=pads)

                # Fill results array
                results_array[frame] = prediction

                # Create polygon ROIs for each cell and upload segmentation
                if self.upload:
                    update_service = self.conn.getUpdateService()
                    img_ome = self.conn.getObject("Image", img_ome.getId())  # Reload needed
                    mask_polygon = omero.model.PolygonI()
                    mask_polygon.theZ, mask_polygon.theT, mask_polygon.fillColor = rint(0), rint(frame), rint(0)
                    # mask_polygon.theC = rint(self.channel)
                    mask_polygon.strokeColor = rint(
                        int.from_bytes([255, 255, 0, 255], byteorder='big', signed=True))
                    prediction_ids = get_indices_pandas(prediction)
                    rois = []
                    if np.max(prediction) > 0:
                        try:
                            for m_key, prediction_idx in prediction_ids.items():
                                try:
                                    polygon_point_list = cv2_countour(prediction_idx)
                                except AssertionError:
                                    continue
                                for polygon_points in polygon_point_list:
                                    points = ""
                                    for col_idx in range(polygon_points.shape[1]):
                                        points += "{},{} ".format(polygon_points[1, col_idx], polygon_points[0, col_idx])
                                    mask_polygon.points = rstring(points)
                                    rois.append(copy(mask_polygon))

                            update_service.saveAndReturnObject(create_roi(img_ome, rois))
                        except:
                            pass

                # Upload metadata
                if frame == 0 and self.upload:
                    now = datetime.now()
                    key_value_data = [['inference_model', self.model_name],
                                      ['inference_date', now.strftime("%m/%d/%Y, %H:%M:%S")]]
                    for ann in img_ome.listAnnotations():
                        if ann.OMERO_TYPE == omero.model.MapAnnotationI:
                            keys_values = ann.getValue()
                            for key, value in keys_values:
                                if key not in ['inference_model', 'inference_date', 'last_modification']:
                                    key_value_data.append([key, value])
                            if ann.canEdit():
                                self.conn.deleteObjects("Annotation", [ann.getId()], wait=True)
                            else:
                                self.text_output.emit(
                                    f'  Problems with deleting annotations (probably from another user), redundant'
                                    f' results possible. Please check on OMERO.web.')
                    map_ann = MapAnnotationWrapper(self.conn)
                    map_ann.setNs(
                        metadata.NSCLIENTMAPANNOTATION)  # Use 'client' namespace to allow editing in Insight & web
                    map_ann.setValue(key_value_data)
                    map_ann.save()
                    img_ome.linkAnnotation(map_ann)

                # Update progress bar
                self.progress.emit(int(100 * ((i + (frame + 1) / img_ome.getSizeT()) / len(self.img_id_list))))

            if not self.upload:  # Save locally
                fname = result_path / img_ome.getName()
                tiff.imwrite(str(result_path / f"{fname.stem}_channel{self.channel}.tif"), results_array)

        self.disconnect()
        self.progress.emit(100)
        self.finished.emit()

    def inference(self, img, min_val, max_val, pads):
        """ Prediction of crops for pre-labeling

        :param img: Crop to predict
        :type img:
        :param min_val: Minimum image/frame value for normalization
        :type min_val: int
        :param max_val: Maximum image/frame value for normalization
        :type max_val: int
        :return: prediction
        :param pads: Number of padded zeros in each dimension (need to be removed after inference)
        :type pads: list
        """

        self.net.eval()
        torch.set_grad_enabled(False)

        # Normalize crop and convert to tensor / img_batch
        img = np.transpose(img, (2, 0, 1))  # color channel on first position
        img_batch = 2 * (img.astype(np.float32) - min_val) / (max_val - min_val) - 1
        img_batch = torch.from_numpy(img_batch[None, :, :]).to(torch.float)
        img_batch = img_batch.to(self.device)

        # Prediction
        if self.model_settings['label_type'] == 'distance':
            try:
                prediction_border_batch, prediction_cell_batch = self.net(img_batch)
            except RuntimeError:
                prediction = np.zeros_like(img, dtype=np.uint16)[0, pads[0]:, pads[1]:]
                self.text_output.emit('RuntimeError during inference (maybe not enough ram/vram?)')
            else:
                prediction_cell_batch = prediction_cell_batch[0, 0, pads[0]:, pads[1]:, None].cpu().numpy()
                prediction_border_batch = prediction_border_batch[0, 0, pads[0]:, pads[1]:, None].cpu().numpy()
                prediction = distance_postprocessing(border_prediction=np.copy(prediction_border_batch),
                                                     cell_prediction=np.copy(prediction_cell_batch),
                                                     th_cell=self.ths[0],
                                                     th_seed=self.ths[1])
        else:
            try:
                prediction_batch = self.net(img_batch)
            except RuntimeError:
                prediction = np.zeros_like(img, dtype=np.uint16)[0, pads[0]:, pads[1]:]
                self.text_output.emit('RuntimeError during inference (maybe not enough ram/vram?)')
            else:
                prediction_batch = F.softmax(prediction_batch, dim=1)
                prediction_batch = prediction_batch[:, :, pads[0]:, pads[1]:].cpu().numpy()
                prediction_batch = np.transpose(prediction_batch[0], (1, 2, 0))
                prediction = boundary_postprocessing(prediction_batch)

        return prediction

    @pyqtSlot()
    def inference_finished(self):
        """ Close connection and send finished signal """
        self.disconnect()
        self.finished.emit()

    @pyqtSlot()
    def stop_inference_process(self):
        """ Set internal stop state to True

        :return: None
        """
        self.stop_inference = True
