import numpy as np
import omero.model
import tifffile as tiff

from omero.gateway import BlitzGateway
from PyQt5.QtCore import pyqtSignal, QObject, pyqtSlot, QCoreApplication
from shutil import rmtree
from skimage.draw import polygon


class DataExportWorker(QObject):
    """ Worker class for dataset import """
    finished = pyqtSignal()  # Signal when import is finished
    progress = pyqtSignal(int)  # Signal for updating the progress bar
    text_output = pyqtSignal(str)  # Signal for possible exceptions, e.g., user interaction to stop export
    stop_export = False

    def export_data(self, trainset_id, train_path, omero_username, omero_password,  omero_host, omero_port, group_id):
        """ Import data to Omero

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
        :return: None
        """

        # Connect to Omero
        conn = BlitzGateway(omero_username, omero_password, host=omero_host, port=omero_port, secure=True)
        conn.connect()
        if group_id is not None:
            conn.setGroupForSession(group_id)

        trainset = conn.getObject("Dataset", trainset_id)
        trainset_path = train_path / trainset.getName()
        trainset_length = len(list(trainset.listChildren()))

        self.text_output.emit('\nDownloading data')

        # Get images and masks
        roi_service = conn.getRoiService()
        for i, file in enumerate(trainset.listChildren()):

            QCoreApplication.processEvents()  # Update to get stop signal
            if self.stop_export:
                self.text_output.emit("Stop training set export due to user interaction.\nDelete local folder.")
                rmtree(str(trainset_path))
                break

            mask_rois = roi_service.findByImage(file.getId(), None)
            mask = np.zeros(shape=(file.getSizeY(), file.getSizeX()), dtype=np.uint16)
            cell_id = 1
            for mask_roi in mask_rois.rois:
                for s in mask_roi.copyShapes():
                    if type(s) == omero.model.PolygonI:
                        # t = s.getTheT().getValue()
                        r, c = make_coordinates(s.getPoints().getValue(), crop_size=file.getSizeX())
                        rr, cc = polygon(r, c)
                        mask[rr, cc] = cell_id
                        cell_id += 1

            if cell_id == 1:  # no roi found
                continue

            # Get meta data
            frame_min, frame_max, subset, pre_labeled, corrected = 0, 65535, '', False, False
            for ann in file.listAnnotations():
                if ann.OMERO_TYPE == omero.model.MapAnnotationI:
                    keys_values = ann.getValue()
                    for key, value in keys_values:
                        if key == 'set':
                            subset = value
                        elif key == 'min_frame':
                            frame_min = int(value)
                        elif key == 'max_frame':
                            frame_max = int(value)
                        elif key == 'pre_labeled':
                            if value != 'False':
                                pre_labeled = True
                        elif key == 'last_modification':
                            corrected = value

            # Keep only pre-labeled images that are corrected
            if pre_labeled and not corrected:
                self.text_output.emit("   {}: pre-labeled but not corrected --> skip.".format(file.getName()))
                continue

            # Get image
            if file.getSizeC() > 1:
                img = np.zeros(shape=(file.getSizeY(), file.getSizeX(), file.getSizeC()))
                zct_list = []
                for z in range(file.getSizeZ()):  # all slices (1 anyway)
                    for c in range(file.getSizeC()):  # all channels
                        for t in range(file.getSizeT()):  # all time-points (1 anyway)
                            zct_list.append((z, c, t))
                for h, plane in enumerate(file.getPrimaryPixels().getPlanes(zct_list)):
                    img[:, :, zct_list[h][1]] = plane
            else:
                self.text_output.emit("Not a rgb data set --> break.")
                break

            # Normalization
            img = 65535 * (img.astype(np.float32) - frame_min) / (frame_max - frame_min)
            img = np.clip(img, 0, 65535).astype(np.uint16)

            # Save image, mask and label
            fname = file.getName().split('img_')[1]
            tiff.imwrite(str(trainset_path / subset / 'img_{}'.format(fname)), img)
            tiff.imwrite(str(trainset_path / subset / 'mask_{}'.format(fname)), mask)

            self.progress.emit(int(100 * (i+1) / trainset_length))

        if self.stop_export:
            self.progress.emit(0)
        else:
            self.progress.emit(100)

        conn.close()
        self.finished.emit()

    @pyqtSlot()
    def stop_export_process(self):
        """ Set internal export stop state to True

        :return: None
        """
        self.stop_export = True


def make_coordinates(polystr, crop_size):
    """ Convert polygon string to coordinates

    :param polystr: String with coordinates.
    :type polystr: str
    :param crop_size: Size of the (square) crop.
    :type crop_size: int
    :return: List of coordinates.
    """
    r, c = [], []

    for textCoord in polystr.split(' '):
        coord = textCoord.split(',')
        if len(coord) == 1:
            continue
        r.append(np.minimum(np.maximum(int(round(float(coord[1]))), 0), crop_size-1))
        c.append(np.minimum(np.maximum(int(round(float(coord[0]))), 0), crop_size-1))

    return r, c
