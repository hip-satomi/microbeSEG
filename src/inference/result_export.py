import numpy as np
import omero.model
import tifffile as tiff

from omero.gateway import BlitzGateway
from PyQt5.QtCore import pyqtSignal, QObject, pyqtSlot, QCoreApplication
from skimage.draw import polygon, polygon_perimeter
from skimage.measure import label


class ResultExportWorker(QObject):
    """ Worker class for dataset import """
    finished = pyqtSignal()  # Signal when import is finished
    progress = pyqtSignal(int)  # Signal for updating the progress bar
    text_output = pyqtSignal(str)  # Signal for possible exceptions, e.g., user interaction to stop export
    stop_export = False

    def __init__(self, img_id_list, inference_path, omero_username, omero_password,  omero_host, omero_port, group_id):
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
        """

        super().__init__()

        self.img_id_list = img_id_list
        self.omero_username = omero_username
        self.omero_password = omero_password
        self.omero_host = omero_host
        self.omero_port = omero_port
        self.group_id = group_id
        self.conn = None
        self.inference_path = inference_path  # path for local results
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

    def export_data(self):
        """ Export results from Omero """

        self.text_output.emit('\nExport data and results')

        for i, img_id in enumerate(self.img_id_list):

            QCoreApplication.processEvents()  # Update to get stop signal
            if self.stop_export:
                self.text_output.emit("Stop result export due to user interaction.")
                self.progress.emit(0)
                self.disconnect()
                self.finished.emit()
                return

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

            # Check if z stack
            if img_ome.getSizeZ() > 1:
                self.text_output.emit(f'  Skip {parent_projects}: {img_ome.getName()} (is z-stack)')
                continue

            self.progress.emit(int(100 * ((i + 1 / 10) / len(self.img_id_list))))

            # Get images and masks
            roi_service = self.conn.getRoiService()
            mask_rois = roi_service.findByImage(img_ome.getId(), None)
            mask = np.zeros(shape=(img_ome.getSizeT(), img_ome.getSizeY(), img_ome.getSizeX()), dtype=np.uint16)
            mask_outlines = np.zeros(shape=(img_ome.getSizeT(), img_ome.getSizeY(), img_ome.getSizeX()), dtype=bool)
            cell_id = 1
            for mask_roi in mask_rois.rois:
                for s in mask_roi.copyShapes():
                    if type(s) == omero.model.PolygonI:
                        t = s.getTheT().getValue()
                        r, c = make_coordinates(s.getPoints().getValue(),
                                                size_x=img_ome.getSizeX(),
                                                size_y=img_ome.getSizeY())
                        # Fill mask
                        rr, cc = polygon(r, c)
                        mask[t, rr, cc] = cell_id
                        # Fill mask_outlines
                        cell_perimeter = polygon_perimeter(r, c, shape=(img_ome.getSizeY(), img_ome.getSizeX()),
                                                           clip=True)
                        mask_outlines[t, cell_perimeter[0], cell_perimeter[1]] = True
                        cell_id += 1
                        if cell_id == 66535:
                            mask = mask.astype(np.int32)
            self.progress.emit(int(100 * ((i + 1 / 4) / len(self.img_id_list))))

            QCoreApplication.processEvents()  # Update to get stop signal
            if self.stop_export:
                self.text_output.emit("Stop result export due to user interaction.")
                self.progress.emit(0)
                self.disconnect()
                self.finished.emit()
                return

            for frame in range(len(mask)):
                mask[frame] = label(mask[frame], background=0)
            if np.max(mask) <= 65535 and mask.dtype == np.int32:
                mask = mask.astype(np.uint16)

            if np.max(mask) == 0:
                self.text_output.emit(f'  Skip {parent_projects}: {img_ome.getName()} (no segmentation results found)')
                continue

            # Get image
            if img_ome.getSizeC() > 1:
                img = np.zeros(shape=(img_ome.getSizeT(), img_ome.getSizeY(), img_ome.getSizeX(), img_ome.getSizeC()))
                zct_list = []
                for z in range(img_ome.getSizeZ()):  # all slices (1 anyway)
                    for c in range(img_ome.getSizeC()):  # all channels
                        for t in range(img_ome.getSizeT()):  # all time-points
                            zct_list.append((z, c, t))
                for h, plane in enumerate(img_ome.getPrimaryPixels().getPlanes(zct_list)):
                    img[zct_list[h][2], :, :, zct_list[h][1]] = plane
            else:
                zct_list = []
                for z in range(img_ome.getSizeZ()):  # all slices (1 anyway)
                    for c in range(img_ome.getSizeC()):  # all channels
                        for t in range(img_ome.getSizeT()):  # all time-points
                            zct_list.append((z, c, t))
                img = []
                for plane in img_ome.getPrimaryPixels().getPlanes(zct_list):
                    img.append(plane)
                img = np.array(img)
            self.progress.emit(int(100 * ((i + 2 / 4) / len(self.img_id_list))))

            QCoreApplication.processEvents()  # Update to get stop signal
            if self.stop_export:
                self.text_output.emit("Stop result export due to user interaction.")
                self.progress.emit(0)
                self.disconnect()
                self.finished.emit()
                return

            # Overlay
            img_overlay = np.clip(255 * img.astype(np.float32) / np.max(img), 0, 255).astype(np.uint8)
            if img_ome.getSizeC() == 1:
                img_overlay = np.concatenate((img_overlay[..., None], img_overlay[..., None], img_overlay[..., None]),
                                             axis=-1)

            img_overlay[mask_outlines, 0] = 255
            img_overlay[mask_outlines, 1] = 255
            img_overlay[mask_outlines, 2] = 0

            self.progress.emit(int(100 * ((i + 3 / 4) / len(self.img_id_list))))

            # Save image, mask and label
            if isinstance(parent_projects, list):
                result_path = self.inference_path / parent_projects[0]
            else:
                result_path = self.inference_path / img_ome.getProject().getName()
            result_path.mkdir(exist_ok=True)
            fname = result_path / img_ome.getName()
            tiff.imwrite(str(result_path / f"{fname.stem}.tif"), img, compress=1)
            tiff.imsave(str(result_path / f"{fname.stem}_mask.tif"), mask, compress=1)
            tiff.imsave(str(result_path / f"{fname.stem}_overlay.tif"), img_overlay, compress=1)
            tiff.imsave(str(result_path / f"{fname.stem}_outlines.tif"), mask_outlines)

            # Get and save analysis results (counts ...)
            for ann in img_ome.listAnnotations(ns='microbeseg.analysis.namespace'):
                with open(result_path / f"{fname.stem}_analysis.csv", 'wb') as outfile:
                    for chunk in ann.getFileInChunks():
                        outfile.write(chunk)

            self.progress.emit(int(100 * (i+1) / len(self.img_id_list)))

        self.disconnect()
        self.progress.emit(100)
        self.finished.emit()

    @pyqtSlot()
    def stop_export_process(self):
        """ Set internal export stop state to True

        :return: None
        """
        self.stop_export = True


def make_coordinates(polystr, size_x, size_y):
    """ Convert polygon string to coordinates

    :param polystr: String with coordinates
    :type polystr: str
    :param size_x: Image width in px
    :type size_x: int
    :param size_y: image height in px
    :type size_y: int
    :return: Lists of coordinates
    """
    r, c = [], []

    for textCoord in polystr.split(' '):
        coord = textCoord.split(',')
        if len(coord) == 1:
            continue
        r.append(np.minimum(np.maximum(int(round(float(coord[1]))), 0), size_y-1))
        c.append(np.minimum(np.maximum(int(round(float(coord[0]))), 0), size_x-1))

    return r, c