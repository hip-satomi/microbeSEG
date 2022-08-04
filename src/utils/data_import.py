import json
import numpy as np
import omero.model

from copy import copy
from omero.gateway import BlitzGateway, MapAnnotationWrapper
from omero.constants import metadata
from omero.rtypes import rint, rstring
from PyQt5.QtCore import pyqtSignal, QObject, pyqtSlot, QCoreApplication
from random import random
from skimage.io import imread

from src.utils import utils
from src.utils.hull_polygon import get_indices_pandas, cv2_countour


class DataImportWorker(QObject):
    """ Worker class for dataset import """
    finished = pyqtSignal()  # Signal when import is finished
    text_output = pyqtSignal(str)  # Signal for possible exceptions, e.g., too small crops
    progress = pyqtSignal(int)  # Signal for updating the progress bar
    stop_import = False

    def import_data(self, img_ids, keep_normalization, crop_size, trainset_id, train_path, omero_username,
                    omero_password,  omero_host, omero_port, group_id, p_train, p_val, p_test):
        """ Import data to Omero

        :param img_ids: List of image indices to import
        :type img_ids: list
        :param keep_normalization: Keep initial normalization of the imported images
        :type keep_normalization: bool
        :param crop_size: Crop size of the Omero dataset to import the data into
        :type crop_size: int
        :param trainset_id: id of the Omero dataset to import the data into
        :type trainset_id
        :param train_path: Local path to save training data info
        :type train_path: pathlib Path object
        :param omero_username: Omero username
        :type omero_username: str
        :param omero_password: Omero password
        :type omero_password: str
        :param omero_host: Omero host
        :type omero_host: str
        :param omero_port: Omero port
        :type omero_port: str
        :param group_id: Current user group id
        :type group_id: int
        :param p_train: Probability to save crop into train set
        :type p_train: float
        :param p_val: Probability to save crop into val set
        :type p_val: float
        :param p_test: Probability to save crop into test set
        :type p_test: float
        :return: None
        """

        # Connect to Omero
        conn = BlitzGateway(omero_username, omero_password, host=omero_host, port=omero_port, secure=True)
        conn.connect()

        if group_id is not None:
            conn.setGroupForSession(group_id)

        self.text_output.emit('\nImporting data')

        # Get information for filenaming
        split_info = []
        for ann in conn.getObject("Dataset", trainset_id).listAnnotations(ns='split.info.namespace'):
            with open(train_path / 'split_info.json', 'wb') as outfile:
                for chunk in ann.getFileInChunks():
                    outfile.write(chunk)
            with open(train_path / 'split_info.json', 'r') as infile:
                split_info = json.load(infile)
        if not split_info:
            split_info = {'used': [],
                          'num_ext': 0}
        if 'num_ext' not in split_info:
            split_info['num_ext'] = 0

        # Load images and masks, create ROIs and load to Omero
        for i, img_id in enumerate(img_ids):

            QCoreApplication.processEvents()  # Update to get stop signal
            if self.stop_import:
                self.text_output.emit("Stop data import due to user interaction.")
                break

            # Load image if type is supported
            if img_id.suffix.lower() in ['.png', '.tiff', '.tif', '.jpeg', '.jpg']:
                img = np.squeeze(imread(str(img_id)))
            else:
                continue

            # Load corresponding mask
            mask_id = img_id.parent / "mask{}".format(img_id.name.split('img')[-1])
            if mask_id.suffix.lower() in ['.png', '.tiff', '.tif', '.jpeg', '.jpg']:
                try:  # Mask and image could have different file formats
                    mask = np.squeeze(imread(str(mask_id)))
                except FileNotFoundError:
                    self.text_output.emit("  {}: No mask found (different file formats?)".format(img_id.name))
                    continue
            else:
                continue

            # Discard 3D images and convert rgb/multi-channel images to grayscale
            if len(img.shape) == 3 and img.shape[-1] <= 3:
                img = np.mean(img, axis=-1).astype(img.dtype)
                self.text_output.emit("  {}: rgb image converted to grayscale".format(img_id.name))
            if len(img.shape) >= 3:
                self.text_output.emit("  {}: 3D image --> skip".format(img_id.name))
                continue

            # Some masks are pseudo-rgb images
            if len(mask.shape) == 3 and mask.shape[-1] <= 3:
                mask = np.mean(mask, axis=-1).astype(mask.dtype)
                self.text_output.emit("  {}: rgb mask converted to grayscale".format(img_id.name))
            if len(mask.shape) >= 3:
                self.text_output.emit("  {}: mask shape not supported --> skip".format(img_id.name))
                continue

            # Get crop information before cropping/padding
            if keep_normalization and np.issubdtype(img.dtype, np.unsignedinteger):
                min_frame, max_frame = np.iinfo(img.dtype).min, np.iinfo(img.dtype).max
            else:
                min_frame, max_frame = np.min(img), np.max(img)
            mean_frame, std_frame = np.mean(img), np.std(img)
            x_start, y_start = 0, 0

            # Image < crop size: pad
            pads = [0, 0]
            if img.shape[0] < crop_size:
                pads[0] = crop_size - img.shape[0]
            if img.shape[1] < crop_size:
                pads[1] = crop_size - img.shape[1]
            if pads[0] > img.shape[0] or pads[1] > img.shape[1]:
                self.text_output.emit("  {}: too much pads needed --> skip".format(img_id.name))
                continue
            img = np.pad(img, ((int(np.ceil(pads[0]/2)), int(np.floor(pads[0]/2))),
                               (int(np.ceil(pads[1]/2)), int(np.floor(pads[1]/2)))),
                         mode='constant')
            mask = np.pad(mask, ((int(np.ceil(pads[0]/2)), int(np.floor(pads[0]/2))),
                                 (int(np.ceil(pads[1]/2)), int(np.floor(pads[1]/2)))),
                          mode='constant')

            # Image > crop size: crop
            img_list, mask_list = [], []
            if img.shape[0] > crop_size or img.shape[1] > crop_size:

                num_crops_y, num_crops_x = img.shape[0] // crop_size, img.shape[1] // crop_size

                # Remove image borders if possible (annotation errors are more likely on image border)
                border_y = np.maximum(0, (img.shape[0] - num_crops_y * crop_size) / 2)
                border_x = np.maximum(0, (img.shape[1] - num_crops_x * crop_size) / 2)

                if border_y > 0:  # crop is in y direction
                    img = img[int(np.floor(border_y)):int(np.floor(-border_y)), ...]
                    mask = mask[int(np.floor(border_y)):int(np.floor(-border_y)), ...]
                if border_x > 0:  # crop is in x direction
                    img = img[:, int(np.floor(border_x)):int(np.floor(-border_x))]
                    mask = mask[:, int(np.floor(border_x)):int(np.floor(-border_x))]

                num_cells, area_cells = len(utils.get_nucleus_ids(mask)), np.sum(mask > 0)

                for h in range(num_crops_y):

                    for w in range(num_crops_x):

                        y_start, x_start = h * crop_size, w * crop_size
                        img_crop = img[y_start:y_start+crop_size, x_start:x_start+crop_size]
                        mask_crop = mask[y_start:y_start + crop_size, x_start:x_start + crop_size]

                        # avoid empty crops (or crops with only very few information)
                        num_cells_crop = len(utils.get_nucleus_ids(mask_crop))
                        if num_cells_crop == 0 or (np.sum(mask_crop > 0) < (area_cells / num_cells)):
                            continue
                        img_list.append([np.copy(img_crop),
                                         np.copy(mask_crop),
                                         x_start + int(np.floor(border_x)),
                                         y_start + int(np.floor(border_y))])
            else:
                img_list.append([np.copy(img), np.copy(mask), x_start, y_start])

            # train/val/test set assignment. Different crops from same image should belong to the same set
            random_number = random()
            if random_number < p_test:
                import_set = 'test'
            elif random_number < p_test + p_val:
                import_set = 'val'
            else:
                import_set = 'train'

            # Import
            for img, mask, x_start, y_start in img_list:

                # Upload image, increase counter for file naming and update progress
                omero_img = conn.createImageFromNumpySeq(utils.plane_gen(img),
                                                         "img_ext{:03d}.tif".format(split_info['num_ext']),
                                                         1, 1, 1,
                                                         description='imported training image crop',
                                                         dataset=conn.getObject("Dataset", trainset_id))

                # Increase counter and send progress update
                split_info['num_ext'] += 1
                self.progress.emit(int(100 * (i+1) / len(img_ids)))

                # Attach metadata to uploaded image
                key_value_data = [["crop_size", str(crop_size)],
                                  ["set", import_set],
                                  ["project", "imported_data"],
                                  ["dataset", img_id.parent.stem],
                                  ["image", "ext_{}".format(img_id.name)],
                                  ["image_id", "imported"],
                                  ["frame", str(0)],
                                  ["channel", str(0)],
                                  ["pre_labeled", str(False)],
                                  ["x_start", str(x_start)],
                                  ["y_start", str(y_start)],
                                  ["min_frame", str(min_frame)],
                                  ["max_frame", str(max_frame)],
                                  ["mean_frame", str(mean_frame)],
                                  ["std_frame", str(std_frame)]]
                map_ann = MapAnnotationWrapper(conn)
                map_ann.setNs(metadata.NSCLIENTMAPANNOTATION)  # Use 'client' namespace to allow editing in Insight/web
                map_ann.setValue(key_value_data)
                map_ann.save()
                omero_img.linkAnnotation(map_ann)

                # Initialize update service for ROIs and set polygon settings
                update_service = conn.getUpdateService()
                omero_img = conn.getObject("Image", omero_img.getId())
                mask_polygon = omero.model.PolygonI()
                mask_polygon.theZ, mask_polygon.theT, mask_polygon.fillColor = rint(0), rint(0), rint(0)
                mask_polygon.strokeColor = rint(int.from_bytes([255, 255, 0, 255], byteorder='big', signed=True))

                # Create polygon ROIs for each cell
                mask_ids = get_indices_pandas(mask)
                mask_polygon_list = []
                for m_key, mask_idx in mask_ids.items():
                    try:
                        polygon_point_list = cv2_countour(mask_idx)
                    except AssertionError:
                        self.text_output.emit("img_ext{:03d}: roi error for mask {}".format(split_info['num_ext']-1,
                                                                                            m_key))
                    for polygon_points in polygon_point_list:
                        points = ""
                        for col_idx in range(polygon_points.shape[1]):
                            points += "{},{} ".format(polygon_points[1, col_idx], polygon_points[0, col_idx])
                        mask_polygon.points = rstring(points)
                        mask_polygon_list.append(copy(mask_polygon))

                # Attach to uploaded image (advantage: all ROIs at once, but can only be viewed with webviewer)
                update_service.saveAndReturnObject(create_roi(omero_img, mask_polygon_list))

        # Update json file on Omero server
        with open(train_path / 'split_info.json', 'w', encoding='utf-8') as outfile:
            json.dump(split_info, outfile, ensure_ascii=False, indent=2)
        namespace = 'split.info.namespace'
        file_ann = conn.createFileAnnfromLocalFile(str(train_path / 'split_info.json'),
                                                   mimetype='application/json',
                                                   ns=namespace,
                                                   desc='info about train/test split and already used frames')
        dataset = conn.getObject("Dataset", trainset_id)
        to_delete = []
        for ann in dataset.listAnnotations(ns=namespace):
            to_delete.append(ann.id)
        if to_delete:
            conn.deleteObjects('Annotation', to_delete, wait=True)
        conn.getObject("Dataset", trainset_id).linkAnnotation(file_ann)

        if not self.stop_import:
            self.progress.emit(100)

        conn.close()
        self.finished.emit()

    @pyqtSlot()
    def stop_import_process(self):
        """ Set internal import stop state to True

        :return: None
        """
        self.stop_import = True


def create_roi(omero_img, shapes):
    """ Create Omero ROI

    :param omero_img: Omero image wrapper to attach the ROI to.
    :type omero_img:
    :param shapes: List of mask polygons.
    :type shapes: list
    :return: Omero ROI
    """
    # create an ROI, link it to Image
    roi = omero.model.RoiI()
    # use the omero.model.ImageI that underlies the 'image' wrapper
    roi.setImage(omero_img._obj)
    # for shape in shapes:
    #     roi.addShape(shape)
    roi.addAllShapeSet(shapes)
    return roi
