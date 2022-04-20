import numpy as np
import omero.model
import os
import pandas as pd

from omero.gateway import BlitzGateway
from PyQt5.QtCore import pyqtSignal, QObject, pyqtSlot, QCoreApplication
from skimage.draw import polygon, polygon_perimeter
from skimage.measure import label, regionprops


class AnalysisWorker(QObject):
    """ Worker class for dataset import """
    finished = pyqtSignal()  # Signal when import is finished
    progress = pyqtSignal(int)  # Signal for updating the progress bar
    text_output = pyqtSignal(str)  # Signal for possible exceptions, e.g., user interaction to stop export
    stop_analysis = False

    def __init__(self, img_id_list, results_path, omero_username, omero_password,  omero_host, omero_port, group_id):
        """

        :param img_id_list: List of omero image ids
        :type img_id_list: list
        :param results_path: Local results path
        :type results_path: pathlib Path object
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
        self.results_path = results_path
        self.omero_username = omero_username
        self.omero_password = omero_password
        self.omero_host = omero_host
        self.omero_port = omero_port
        self.group_id = group_id
        self.conn = None

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

    def analyze_data(self):
        """ Analyze data from Omero """

        self.text_output.emit('\nAnalyze results')

        for i, img_id in enumerate(self.img_id_list):

            QCoreApplication.processEvents()  # Update to get stop signal
            if self.stop_analysis:
                self.text_output.emit("Stop analysis due to user interaction.")
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

            if not self.conn.canWrite(img_ome):
                self.text_output.emit(f'  Skip {img_ome.getProject().getName()}: {img_ome.getName()} '
                                      f'(no write permission)')
                continue

            # Check if z stack
            if img_ome.getSizeZ() > 1:
                self.text_output.emit(f'  Skip {img_ome.getProject().getName()}: {img_ome.getName()} (is z-stack)')
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

            self.progress.emit(int(100 * ((i + 3 / 4) / len(self.img_id_list))))

            for frame in range(len(mask)):
                mask[frame] = label(mask[frame], background=0)
            if np.max(mask) <= 65535 and mask.dtype == np.int32:
                mask = mask.astype(np.uint16)

            if np.max(mask) == 0:
                self.text_output.emit(f'  Skip {img_ome.getProject().getName()}: {img_ome.getName()} '
                                      f'(no segmentation results found)')
                continue

            # Analyze
            results = {'frame': [],
                       'counts': [],
                       'mean_area': [],
                       'total_area': [],
                       'mean_minor_axis_length': [],
                       'mean_major_axis_length': []}
            for frame in range(len(mask)):
                results['frame'].append(frame)
                results['counts'].append(np.max(mask[frame]))
                results['total_area'].append(np.sum(mask[frame]))

                props = regionprops(mask[frame])
                areas, major_axes, minor_axes = [], [], []
                for nucleus in props:
                    areas.append(nucleus.area)
                    major_axes.append(nucleus.axis_major_length)
                    minor_axes.append(nucleus.axis_minor_length)
                results['mean_area'].append(np.mean(np.array(areas)))
                results['mean_minor_axis_length'].append(np.mean(np.array(minor_axes)))
                results['mean_major_axis_length'].append(np.mean(np.array(major_axes)))

            # Convert to pandas data frame and save temporarily (needed for upload)
            results_df = pd.DataFrame(results)
            result_path = self.results_path / img_ome.getProject().getName()
            fname = result_path / img_ome.getName()
            results_df.to_csv(self.results_path / f"{fname.stem}_analysis.csv", index=False)

            # Delete old results if available
            namespace = 'microbeseg.analysis.namespace'
            to_delete = []
            for ann in img_ome.listAnnotations(ns=namespace):
                to_delete.append(ann.getId())
            if to_delete:
                self.conn.deleteObjects('Annotation', to_delete, wait=True)

            # Upload new results
            file_ann = self.conn.createFileAnnfromLocalFile(str(self.results_path / f"{fname.stem}_analysis.csv"),
                                                            mimetype='text/csv',
                                                            ns=namespace,
                                                            desc='microbeSEG analysis results')
            img_ome.linkAnnotation(file_ann)

            # Remove temp file
            os.remove(str(self.results_path / f"{fname.stem}_analysis.csv"))

            self.progress.emit(int(100 * (i+1) / len(self.img_id_list)))

        self.disconnect()
        self.progress.emit(100)
        self.finished.emit()

    @pyqtSlot()
    def stop_analysis_process(self):
        """ Set internal export stop state to True

        :return: None
        """
        self.stop_analysis = True


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
