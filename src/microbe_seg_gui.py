""" Graphical user interface of microbeSEG """
import json
import numpy as np
import omero.model
import os
import pandas as pd
import sys
import torch
import urllib
import webbrowser

from copy import copy
from functools import partial
from pathlib import Path
from random import random, shuffle
from shutil import rmtree
from skimage.transform import resize

from omero.gateway import BlitzGateway, DatasetWrapper, MapAnnotationWrapper, ProjectWrapper
from omero.constants import metadata
from omero.rtypes import rint

from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPalette, QColor, QIntValidator, QKeySequence, QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot

from src.utils.data_cropping import DataCropWorker
from src.utils.data_export import DataExportWorker
from src.utils.data_import import DataImportWorker, create_roi
from src.utils.utils import plane_gen_rgb
from src.evaluation.eval import EvalWorker
from src.inference.analysis import AnalysisWorker
from src.inference.infer import InferWorker
from src.inference.result_export import ResultExportWorker
from src.training.train import CreateLabelsWorker, TrainWorker


class CropSelectionDialog(QDialog):
    """ QDialog subclass with close event which sends signal """

    crop_selection_closed_signal = pyqtSignal()

    def closeEvent(self, event):
        """ Close event """
        self.crop_selection_closed_signal.emit()
        self.close()


class MicrobeSegMainWindow(QWidget):
    """ microbeSEG GUI """

    get_crop_signal = pyqtSignal()  # Signal for worker to create next crop
    stop_cropping_signal = pyqtSignal()  # Signal to stop worker

    def __init__(self, gpu, multi_gpu, omero_settings, model_path, train_path, eval_path, results_path, parent=None):
        """
        
        :param gpu: GPU available or not.
        :type gpu: bool
        :param multi_gpu: multiple GPUs available or not.
        :type multi_gpu: bool
        :param omero_settings: pre-filled settings for the login window and url of the annotation tool
        :type omero_settings: dict
        :param model_path: Path for the trained models
        :type model_path: Path
        :param train_path: Local path for the training dataset
        :type train_path: Path
        :param results_path: Local path for the inference results
        :type results_path: Path
        :param parent:
        """

        super().__init__(parent)

        self.setWindowIcon(QIcon(str(model_path.parent / 'doc' / 'window-logo.png')))
        # Path for trained models
        self.model_path = model_path
        # Path for exported training data sets
        self.train_path = train_path
        # Path for evaluation results
        self.eval_path = eval_path
        # Path for local inference results
        self.results_path = results_path
        # Url of the annotation tool
        self.annotation_tool_url = omero_settings['annotation_tool_url']
        # Screen height and width of the used monitor (needed for rescaling on low resolution screens)
        self.screen_height = QApplication.desktop().screenGeometry().height()
        self.screen_width = QApplication.desktop().screenGeometry().width()
        # Omero connection
        self.conn = None
        # User group
        self.group_id = None
        # List for omero projects
        self.projects = []
        # List for omero datasets
        self.datasets = []
        # Lists for omero files
        self.files, self.file_ids = [], []
        # Inference model list
        self.inference_models, self.inference_model, self.inference_model_ths = [], None, None
        # Prelabel_model_list
        self.prelabel_models, self.prelabel_model, self.prelabel_model_ths = [], None, None,
        self.scores_df = None
        # Selected omero training set
        self.trainset, self.trainset_id, self.trainset_project_id, self.trainset_length = None, None, None, 0
        # Color channel
        self.color_channel = 'rgb'
        # List of available trained models
        self.trained_model_list = []
        # Crop selection
        self.crop_idx, self.crop_size = 0, 0
        self.crop_list, self.split_info, self.crops = [], {}, []
        self.img, self.prelabel_pixmaps = np.zeros(0), []
        self.create_crops, self.stop_crop_creation = False, False
        # Workers and threads
        self.import_thread, self.import_worker = None, None
        self.export_thread, self.export_worker = None, None
        self.train_thread, self.train_worker = None, None
        self.eval_thread, self.eval_worker = None, None
        self.infer_thread, self.infer_worker = None, None
        self.result_export_thread, self.result_export_worker = None, None
        self.result_analysis_thread, self.result_analysis_worker = None, None
        self.label_thread, self.label_worker = None, None
        self.crop_thread, self.crop_worker = None, None
        # GUI states
        self.create_labels_state = False
        self.import_data_state = False
        self.export_data_state = False
        self.train_state = False
        self.eval_state = False
        self.infer_state = False

        # Window title
        self.setWindowTitle("microbeSEG")

        # Login window
        self.omero_login_dialog = QDialog()
        self.omero_login_dialog.setWindowTitle('Connect to OMERO')
        self.omero_login_dialog.setWindowIcon(QIcon(str(self.model_path.parent / 'doc' / 'window-logo.png')))
        self.omero_username_label, self.omero_password_label = QLabel('Username'), QLabel('Password')
        self.omero_host_label, self.omero_port_label = QLabel('Host'), QLabel('Port')
        self.omero_username_edit = QLineEdit(omero_settings['omero_username'])
        self.omero_password_edit = QLineEdit('')
        self.omero_password_edit.setEchoMode(QLineEdit.Password)
        self.omero_host_edit = QLineEdit(omero_settings['omero_host'])
        self.omero_port_edit = QLineEdit(omero_settings['omero_port'])
        self.omero_connect_button = QPushButton('Connect')

        # OMERO box
        self.omero_project_button, self.omero_dataset_button = QPushButton('Project(s)'), QPushButton('Dataset(s)')
        self.omero_project_button.setMinimumWidth(125), self.omero_dataset_button.setMinimumWidth(125)
        self.omero_project_button.setMinimumHeight(25), self.omero_dataset_button.setMinimumHeight(25)
        self.omero_file_button, self.omero_trainset_button = QPushButton('File(s)'), QPushButton('Training set')
        self.omero_file_button.setMinimumWidth(125), self.omero_trainset_button.setMinimumWidth(125)
        self.omero_file_button.setMinimumHeight(25), self.omero_trainset_button.setMinimumHeight(25)
        self.omero_project_edit, self.omero_dataset_edit = QTextEdit(), QTextEdit()
        self.omero_project_edit.setReadOnly(True), self.omero_dataset_edit.setReadOnly(True)
        self.omero_project_edit.setMaximumHeight(25), self.omero_dataset_edit.setMaximumHeight(25)
        self.omero_file_text_edit, self.omero_trainset_text_edit = QTextEdit(), QTextEdit()
        self.omero_file_text_edit.setReadOnly(True), self.omero_trainset_text_edit.setReadOnly(True)
        self.omero_file_text_edit.setMaximumHeight(25), self.omero_trainset_text_edit.setMaximumHeight(25)

        # OMERO box - project selection menu
        self.omero_project_list = QListWidget()
        self.omero_project_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.omero_project_selection = QDialog()
        self.omero_project_selection.setWindowIcon(QIcon(str(self.model_path.parent / 'doc' / 'window-logo.png')))
        self.omero_project_selection.setWindowTitle('Select project(s)')
        self.omero_project_selection_select_all_button = QPushButton('Select all')
        self.omero_project_selection_deselect_all_button = QPushButton('Deselect all')
        self.omero_project_selection_button = QPushButton('Ok')

        # OMERO box - dataset selection menu
        self.omero_dataset_list = QListWidget()
        self.omero_dataset_list.setMinimumWidth(350)
        self.omero_dataset_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.omero_dataset_selection = QDialog()
        self.omero_dataset_selection.setWindowIcon(QIcon(str(self.model_path.parent / 'doc' / 'window-logo.png')))
        self.omero_dataset_selection.setWindowTitle('Select dataset(s)')
        self.omero_dataset_selection_select_all_button = QPushButton('Select all')
        self.omero_dataset_selection_deselect_all_button = QPushButton('Deselect all')
        self.omero_dataset_selection_button = QPushButton('Ok')

        # OMERO box - file selection menu
        self.omero_file_list = QListWidget()
        self.omero_file_list.setMinimumWidth(350)
        self.omero_file_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.omero_file_selection = QDialog()
        self.omero_file_selection.setWindowIcon(QIcon(str(self.model_path.parent / 'doc' / 'window-logo.png')))
        self.omero_file_selection.setWindowTitle('Select file(s)')
        self.omero_file_selection_select_all_button = QPushButton('Select all')
        self.omero_file_selection_deselect_all_button = QPushButton('Deselect all')
        self.omero_file_selection_button = QPushButton('Ok')

        # OMERO box - trainset selection menu
        self.omero_trainset_list = QListWidget()
        self.omero_trainset_selection = QDialog()
        self.omero_trainset_selection.setWindowIcon(QIcon(str(self.model_path.parent / 'doc' / 'window-logo.png')))
        self.omero_trainset_selection.setWindowTitle('Select training and test set')
        self.omero_new_trainset_name_label, self.omero_new_trainset_name_edit = QLabel('Name:'), QLineEdit()
        self.omero_new_trainset_add_button = QPushButton('Add')
        self.omero_trainset_selection_button = QPushButton('Ok')
        self.crop_size_label = QLabel('Crop size [px]:')
        self.crop_size_128_rbutton, self.crop_size_256_rbutton = QRadioButton("128 x 128"), QRadioButton("256 x 256")
        self.crop_size_320_rbutton, self.crop_size_512_rbutton = QRadioButton("320 x 320"), QRadioButton("512 x 512")
        self.crop_size_768_rbutton, self.crop_size_1024_rbutton = QRadioButton("768 x 768"), QRadioButton("1024 x 1024")
        self.crop_size_320_rbutton.setChecked(True)
        self.crop_size_320_rbutton.setMinimumWidth(120), self.crop_size_1024_rbutton.setMinimumWidth(120)
        crop_size_group = QButtonGroup(self)
        crop_size_group.addButton(self.crop_size_128_rbutton), crop_size_group.addButton(self.crop_size_256_rbutton)
        crop_size_group.addButton(self.crop_size_320_rbutton), crop_size_group.addButton(self.crop_size_512_rbutton)
        crop_size_group.addButton(self.crop_size_768_rbutton), crop_size_group.addButton(self.crop_size_1024_rbutton)

        # Settings box: color channel selection
        self.color_channel_label = QLabel('Channel:')
        self.color_channel_label.setMinimumWidth(70)
        self.color_channel_1_rbutton, self.color_channel_2_rbutton = QRadioButton("1"), QRadioButton("2")
        self.color_channel_3_rbutton = QRadioButton("3")
        self.color_channel_rgb_rbutton = QRadioButton("rgb")
        self.color_channel_rgb_rbutton.setChecked(True)
        self.color_channel_1_rbutton.setVisible(False), self.color_channel_2_rbutton.setVisible(False)
        self.color_channel_3_rbutton.setVisible(False)
        color_group = QButtonGroup(self)
        color_group.addButton(self.color_channel_1_rbutton), color_group.addButton(self.color_channel_2_rbutton)
        color_group.addButton(self.color_channel_3_rbutton), color_group.addButton(self.color_channel_rgb_rbutton)

        # Settings box: group selection menu
        self.group_list = QListWidget()
        self.group_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.group_selection = QDialog()
        self.group_selection.setWindowIcon(QIcon(str(self.model_path.parent / 'doc' / 'window-logo.png')))
        self.group_selection.setWindowTitle('Select group')
        self.group_selection_button = QPushButton('Ok')
        self.group_button = QPushButton("Change group")

        # Settings box: device selection
        self.device_label = QLabel('Device:')
        self.device_label.setMinimumWidth(70)
        self.device_cpu_rbutton, self.device_gpu_rbutton = QRadioButton("cpu"), QRadioButton("gpu")
        device_group = QButtonGroup(self)
        device_group.addButton(self.device_cpu_rbutton), device_group.addButton(self.device_gpu_rbutton)
        self.device_multi_gpu_checkbox = QCheckBox("use multiple gpus")
        if not gpu:  # No gpu detected
            self.device_cpu_rbutton.setChecked(True)
            self.device_gpu_rbutton.setCheckable(False), self.device_gpu_rbutton.setEnabled(False)
            self.device_multi_gpu_checkbox.setCheckable(False), self.device_multi_gpu_checkbox.setEnabled(False)
        else:
            self.device_gpu_rbutton.setChecked(True)
            if not multi_gpu:
                self.device_multi_gpu_checkbox.setCheckable(False), self.device_multi_gpu_checkbox.setEnabled(False)

        # Button row
        self.train_data_button = QPushButton('Training data')
        self.train_button = QPushButton('Training')
        self.eval_button = QPushButton('Evaluation')
        self.inference_button = QPushButton('Inference')

        # Text output
        self.output_edit = QTextEdit('Press F1 for help.')
        self.output_edit.setReadOnly(True)

        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.showMessage('Ready')

        # Submenu - training data
        self.train_data = QDialog()
        self.train_data.setWindowIcon(QIcon(str(self.model_path.parent / 'doc' / 'window-logo.png')))
        self.train_data.setWindowModality(Qt.ApplicationModal), self.train_data.setWindowTitle('Add Training Data')
        self.train_data_prelabel_checkbox = QCheckBox('Pre-labeling')
        self.train_data_all_frames_checkbox = QCheckBox('Use all frames')
        self.train_data_prelabel_model_button = QPushButton('Model')
        self.train_data_prelabel_model_button.setVisible(False)
        self.train_data_prelabel_model_edit = QLineEdit('')
        self.train_data_prelabel_model_edit.setAlignment(Qt.AlignRight)
        self.train_data_prelabel_model_edit.setReadOnly(True)
        self.train_data_prelabel_model_edit.setVisible(False)
        self.train_data_prelabel_checkbox.setEnabled(False)
        self.crop_data_set_selection_label = QLabel("Add to:")
        self.crop_data_train_set_checkbox = QCheckBox("train")
        self.crop_data_train_set_checkbox.setChecked(True)
        self.crop_data_val_set_checkbox = QCheckBox("val")
        self.crop_data_val_set_checkbox.setChecked(True)
        self.crop_data_test_set_checkbox = QCheckBox("test")
        self.crop_data_test_set_checkbox.setChecked(True)
        self.train_data_crop_button = QPushButton('Create crops')
        self.train_data_annotate_button = QPushButton('Annotate')
        self.import_data_normalization_checkbox = QCheckBox("Keep normalization")
        self.import_data_set_selection_label = QLabel("Add to:")
        self.import_data_train_set_checkbox = QCheckBox("train")
        self.import_data_train_set_checkbox.setChecked(True)
        self.import_data_val_set_checkbox = QCheckBox("val")
        self.import_data_val_set_checkbox.setChecked(True)
        self.import_data_test_set_checkbox = QCheckBox("test")
        self.import_data_button, self.export_data_button = QPushButton('Import'), QPushButton('Export')
        self.import_data_progress_bar, self.export_data_progress_bar = QProgressBar(self), QProgressBar(self)
        self.import_data_progress_bar.setValue(0), self.import_data_progress_bar.setMaximum(100)
        self.export_data_progress_bar.setValue(0), self.export_data_progress_bar.setMaximum(100)
        self.import_data_progress_bar.hide(), self.export_data_progress_bar.hide()

        # Submenu - prelabel model selection
        self.prelabel_model_list = QListWidget()
        self.prelabel_model_list.setMinimumWidth(350)
        self.prelabel_model_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.prelabel_model_selection = QDialog()
        self.prelabel_model_selection.setWindowIcon(QIcon(str(self.model_path.parent / 'doc' / 'window-logo.png')))
        self.prelabel_model_selection.setWindowTitle('Select model')
        self.prelabel_model_selection_button = QPushButton('Ok')

        # Submenu - training data: crop selection
        self.select_crops = CropSelectionDialog()
        self.select_crops.setWindowIcon(QIcon(str(self.model_path.parent / 'doc' / 'window-logo.png')))
        self.select_crops.setWindowTitle('Select Crops')
        self.select_crops_overlay_checkbox = QCheckBox('Show overlay')
        self.select_crops_overlay_checkbox.setVisible(False)
        self.select_crops_img_left = QLabel('')
        self.select_crops_img_center = QLabel('')
        self.select_crops_img_right = QLabel('')
        self.select_crops_filename_label = QLabel('')
        self.select_crops_img_left.setAlignment(Qt.AlignCenter)
        self.select_crops_img_center.setAlignment(Qt.AlignCenter)
        self.select_crops_img_right.setAlignment(Qt.AlignCenter)
        self.select_crops_img_left_checkbox = QCheckBox('')
        self.select_crops_img_center_checkbox = QCheckBox('')
        self.select_crops_img_right_checkbox = QCheckBox('')
        self.select_crops_filename_label.setAlignment(Qt.AlignCenter)
        self.select_crops_num_crops_label, self.select_crops_counter_label = QLabel('Number of crops:'), QLabel('0')
        self.select_crops_accept_button = QPushButton('Select / next')
        self.select_crops_unchecked_checkbox = QCheckBox('skip')
        self.select_crops_partially_checkbox = QCheckBox('upload image only')
        self.select_crops_partially_checkbox.setTristate(True)
        self.select_crops_partially_checkbox.setCheckState(Qt.PartiallyChecked)
        self.select_crops_partially_checkbox.setVisible(False)
        self.select_crops_checked_checkbox = QCheckBox('upload image')
        self.select_crops_checked_checkbox.setChecked(True)

        # Submenu - train
        self.train_settings = QDialog()
        self.train_settings.setWindowIcon(QIcon(str(self.model_path.parent / 'doc' / 'window-logo.png')))
        self.train_settings.setWindowModality(Qt.ApplicationModal), self.train_settings.setWindowTitle('Training')
        self.train_settings_batchsize_label = QLabel('Batch size:')
        self.train_settings_batchsize_label.setMinimumWidth(85)
        self.train_settings_batchsize_edit = QLineEdit('4')
        self.train_settings_batchsize_edit.setAlignment(Qt.AlignRight)
        self.train_settings_batchsize_edit.setValidator(QIntValidator(1, 16))
        self.train_settings_iterations_label = QLabel('Iterations:')
        self.train_settings_iterations_label.setMinimumWidth(85)
        self.train_settings_iterations_line_edit = QLineEdit('5')
        self.train_settings_iterations_line_edit.setAlignment(Qt.AlignRight)
        self.train_settings_iterations_line_edit.setValidator(QIntValidator(1, 11))
        self.train_settings_optimizer_label = QLabel('Optimizer:')
        self.train_settings_optimizer_adam_rbutton = QRadioButton('Adam')
        self.train_settings_optimizer_ranger_rbutton = QRadioButton('Ranger')
        self.train_settings_optimizer_ranger_rbutton.setChecked(True)
        optimizer_group = QButtonGroup(self)
        optimizer_group.addButton(self.train_settings_optimizer_adam_rbutton)
        optimizer_group.addButton(self.train_settings_optimizer_ranger_rbutton)
        self.train_settings_method_boundary_rbutton = QRadioButton('boundary')
        self.train_settings_method_distance_rbutton = QRadioButton('distance')
        self.train_settings_method_distance_rbutton.setChecked(True)
        train_method_group = QButtonGroup(self)
        train_method_group.addButton(self.train_settings_method_boundary_rbutton)
        train_method_group.addButton(self.train_settings_method_distance_rbutton)
        self.train_settings_train_button = QPushButton('Train')
        self.train_export_data_progress_bar = QProgressBar(self)
        self.train_export_data_progress_bar.setValue(0), self.train_export_data_progress_bar.setMaximum(100)
        self.train_export_data_progress_bar.setFormat('Data export'), self.train_export_data_progress_bar.hide()
        self.train_create_labels_progress_bar = QProgressBar(self)
        self.train_create_labels_progress_bar.setValue(0), self.train_create_labels_progress_bar.setMaximum(100)
        self.train_create_labels_progress_bar.setFormat('Label creation'), self.train_create_labels_progress_bar.hide()
        self.train_training_progress_bar = QProgressBar(self)
        self.train_training_progress_bar.setValue(0), self.train_training_progress_bar.setMaximum(100)
        self.train_training_progress_bar.setFormat('Training'), self.train_training_progress_bar.hide()
        self.train_progress_edit = QTextEdit('')
        self.train_progress_edit.setReadOnly(True), self.train_progress_edit.setFontPointSize(7)
        self.train_progress_edit.hide(), self.train_progress_edit.setMinimumWidth(325)
        self.train_progress_edit.setMinimumHeight(100), self.train_progress_edit.setMaximumHeight(100)

        # Submenu - eval
        self.evaluation_menu = QDialog()
        self.evaluation_menu.setWindowIcon(QIcon(str(self.model_path.parent / 'doc' / 'window-logo.png')))
        self.evaluation_menu.setWindowModality(Qt.ApplicationModal), self.evaluation_menu.setWindowTitle('Evaluation')
        self.evaluation_batch_size_label = QLabel('Batch size:')
        self.evaluation_batch_size_label.setMinimumWidth(80)
        self.evaluation_batch_size_edit = QLineEdit('1')
        self.evaluation_batch_size_edit.setAlignment(Qt.AlignRight)
        self.evaluation_batch_size_edit.setValidator(QIntValidator(1, 16))
        self.evaluation_batch_size_label.setVisible(False)
        self.evaluation_batch_size_edit.setVisible(False)
        self.evaluation_save_raw_checkbox = QCheckBox('Save raw predictions')
        self.evaluation_model_list = QListWidget()
        self.evaluation_model_list.setMinimumWidth(350)
        self.evaluation_model_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.evaluation_model_list_select_all_button = QPushButton('Select all')
        self.evaluation_model_list_deselect_all_button = QPushButton('Deselect all')
        self.evaluation_menu_eval_button = QPushButton('Evaluate')
        self.evaluation_menu_export_data_progress_bar = QProgressBar(self)
        self.evaluation_menu_export_data_progress_bar.setValue(0)
        self.evaluation_menu_export_data_progress_bar.setMaximum(100)
        self.evaluation_menu_export_data_progress_bar.setFormat('Data export')
        self.evaluation_menu_export_data_progress_bar.hide()
        self.evaluation_menu_eval_data_progress_bar = QProgressBar(self)
        self.evaluation_menu_eval_data_progress_bar.setValue(0)
        self.evaluation_menu_eval_data_progress_bar.setMaximum(100)
        self.evaluation_menu_eval_data_progress_bar.setFormat('Evaluation')
        self.evaluation_menu_eval_data_progress_bar.hide()

        # Submenu - inference
        self.inference_menu = QDialog()
        self.inference_menu.setWindowIcon(QIcon(str(self.model_path.parent / 'doc' / 'window-logo.png')))
        self.inference_menu.setWindowModality(Qt.ApplicationModal), self.inference_menu.setWindowTitle('Inference')
        self.inference_batch_size_label = QLabel('Batch size:')
        self.inference_batch_size_label.setMinimumWidth(80)
        self.inference_batch_size_edit = QLineEdit('1')
        self.inference_batch_size_edit.setAlignment(Qt.AlignRight)
        self.inference_batch_size_edit.setValidator(QIntValidator(1, 16))
        self.inference_batch_size_label.setVisible(False)
        self.inference_batch_size_edit.setVisible(False)
        self.inference_menu_model_button = QPushButton('Model')
        self.inference_menu_model_edit = QLineEdit('')
        self.inference_menu_model_edit.setAlignment(Qt.AlignRight)
        self.inference_menu_model_edit.setReadOnly(True)
        self.inference_menu_upload_checkbox = QCheckBox('Upload to OMERO')
        self.inference_menu_upload_checkbox.setChecked(True)
        self.inference_menu_overwrite_checkbox = QCheckBox('Overwrite')
        self.inference_menu_sliding_window_checkbox = QCheckBox('Sliding window')
        self.inference_menu_sliding_window_checkbox.setVisible(False)
        self.inference_menu_process_button = QPushButton('Process selected files')
        self.inference_menu_correct_button = QPushButton('Correct')
        self.inference_menu_export_button = QPushButton('Export')
        self.inference_menu_analyze_button = QPushButton('Analyze')
        self.inference_progress_bar = QProgressBar(self)
        self.inference_progress_bar.setValue(0)
        self.inference_progress_bar.setMaximum(100)
        self.inference_progress_bar.setFormat('Inference')
        self.inference_progress_bar.hide()
        self.result_export_progress_bar = QProgressBar(self)
        self.result_export_progress_bar.setValue(0)
        self.result_export_progress_bar.setMaximum(100)
        self.result_export_progress_bar.setFormat('Result export')
        self.result_export_progress_bar.hide()
        self.analysis_progress_bar = QProgressBar(self)
        self.analysis_progress_bar.setValue(0)
        self.analysis_progress_bar.setMaximum(100)
        self.analysis_progress_bar.setFormat('Analysis')
        self.analysis_progress_bar.hide()

        # Submenu - inference model selection
        self.inference_model_list = QListWidget()
        self.inference_model_list.setMinimumWidth(350)
        self.inference_model_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.inference_model_selection = QDialog()
        self.inference_model_selection.setWindowIcon(QIcon(str(self.model_path.parent / 'doc' / 'window-logo.png')))
        self.inference_model_selection.setWindowTitle('Select model')
        self.inference_model_selection_button = QPushButton('Ok')

        # Init layout and color
        self.init_ui()
        self.init_dark_mode()

        # Shortcuts
        QShortcut(QKeySequence(Qt.Key_F1), self, self.help_shortcut_clicked)
        QShortcut(QKeySequence(Qt.Key_Space), self.select_crops, self.select_crops_accept_button_clicked)
        QShortcut(QKeySequence(Qt.Key_1), self.select_crops, self.select_crops_img_left_checkbox.click)
        QShortcut(QKeySequence(Qt.Key_2), self.select_crops, self.select_crops_img_center_checkbox.click)
        QShortcut(QKeySequence(Qt.Key_3), self.select_crops, self.select_crops_img_right_checkbox.click)
        QShortcut(QKeySequence(Qt.Key_S), self.select_crops, self.show_overlay_sc_pressed)
        self.stop_import_export_sc = QShortcut(QKeySequence("Ctrl+C"), self.train_data)
        self.stop_training_sc = QShortcut(QKeySequence("Ctrl+C"), self.train_settings)
        self.stop_evaluation_sc = QShortcut(QKeySequence("Ctrl+C"), self.evaluation_menu)
        self.stop_inference_sc = QShortcut(QKeySequence("Ctrl+C"), self.inference_menu)

        # Connect omero related buttons
        self.omero_connect_button.clicked.connect(self.omero_connect_button_clicked)
        self.omero_project_button.clicked.connect(self.omero_project_button_clicked)
        self.omero_project_selection_select_all_button.clicked.connect(
            self.omero_project_selection_select_all_button_clicked)
        self.omero_project_selection_deselect_all_button.clicked.connect(
            self.omero_project_selection_deselect_all_button_clicked)
        self.omero_project_selection_button.clicked.connect(self.omero_project_selection_button_clicked)
        self.omero_dataset_button.clicked.connect(self.omero_dataset_button_clicked)
        self.omero_dataset_selection_select_all_button.clicked.connect(
            self.omero_dataset_selection_select_all_button_clicked)
        self.omero_dataset_selection_deselect_all_button.clicked.connect(
            self.omero_dataset_selection_deselect_all_button_clicked)
        self.omero_dataset_selection_button.clicked.connect(self.omero_dataset_selection_button_clicked)
        self.omero_file_button.clicked.connect(self.omero_file_button_clicked)
        self.omero_file_selection_select_all_button.clicked.connect(self.omero_file_selection_select_all_button_clicked)
        self.omero_file_selection_deselect_all_button.clicked.connect(
            self.omero_file_selection_deselect_all_button_clicked)
        self.omero_file_selection_button.clicked.connect(self.omero_file_selection_button_clicked)
        self.omero_trainset_button.clicked.connect(self.omero_trainset_button_clicked)
        self.omero_new_trainset_add_button.clicked.connect(self.omero_new_trainset_add_button_clicked)
        self.omero_trainset_selection_button.clicked.connect(self.omero_trainset_selection_button_clicked)
        self.group_button.clicked.connect(self.group_button_clicked)
        self.group_selection_button.clicked.connect(self.group_selection_button_clicked)

        # Connect training data related buttons
        self.train_data_button.clicked.connect(self.train_data_button_clicked)
        self.train_data_crop_button.clicked.connect(self.train_data_crop_button_clicked)
        self.train_data_prelabel_checkbox.clicked.connect(self.prelabel_checkbox_clicked)
        self.select_crops_accept_button.clicked.connect(self.select_crops_accept_button_clicked)
        self.train_data_annotate_button.clicked.connect(self.train_data_annotate_button_clicked)
        self.select_crops_img_left.mousePressEvent = self.select_crops_img_left_clicked
        self.select_crops_img_center.mousePressEvent = self.select_crops_img_center_clicked
        self.select_crops_img_right.mousePressEvent = self.select_crops_img_right_clicked
        self.select_crops_overlay_checkbox.clicked.connect(self.select_crops_overlay_checkbox_clicked)
        self.select_crops_partially_checkbox.clicked.connect(self.select_crops_partially_checkbox_clicked)
        self.select_crops_checked_checkbox.clicked.connect(self.select_crops_checked_checkbox_clicked)
        self.select_crops_unchecked_checkbox.clicked.connect(self.select_crops_unchecked_checkbox_clicked)
        self.train_data_prelabel_model_button.clicked.connect(self.train_data_prelabel_model_button_clicked)
        self.prelabel_model_selection_button.clicked.connect(self.prelabel_model_selection_button_clicked)

        # Connect import/export related buttons
        self.import_data_button.clicked.connect(self.train_data_import_button_clicked)
        self.export_data_button.clicked.connect(self.train_data_export_button_clicked)

        # Connect training related buttons
        self.train_button.clicked.connect(self.train_button_clicked)
        self.train_settings_train_button.clicked.connect(self.train_settings_train_button_clicked)

        # Connect evaluation related buttons
        self.eval_button.clicked.connect(self.eval_button_clicked)
        self.evaluation_menu_eval_button.clicked.connect(self.eval_menu_eval_button_clicked)
        self.evaluation_model_list_select_all_button.clicked.connect(
            self.evaluation_model_list_select_all_button_clicked)
        self.evaluation_model_list_deselect_all_button.clicked.connect(
            self.evaluation_model_list_deselect_all_button_clicked)

        # Connect inference related buttons
        self.inference_button.clicked.connect(self.inference_button_clicked)
        self.inference_menu_model_button.clicked.connect(self.inference_menu_model_button_clicked)
        self.inference_model_selection_button.clicked.connect(self.inference_model_selection_button_clicked)
        self.inference_menu_correct_button.clicked.connect(self.inference_menu_correct_button_clicked)
        self.inference_menu_process_button.clicked.connect(self.inference_menu_process_button_clicked)
        self.inference_menu_export_button.clicked.connect(self.inference_menu_export_button_clicked)
        self.inference_menu_analyze_button.clicked.connect(self.inference_menu_analyze_button_clicked)

        self.show()
        self.omero_login_dialog.exec()
        if not self.conn:
            sys.exit()  # Login windows is just closed --> close whole application

    def init_ui(self):
        """ Initialize user interface """

        # Login window layout
        lw_layout_1, lw_layout_2 = QVBoxLayout(), QFormLayout()
        lw_layout_2.addRow(self.omero_username_label, self.omero_username_edit)
        lw_layout_2.addRow(self.omero_password_label, self.omero_password_edit)
        lw_layout_2.addRow(self.omero_host_label, self.omero_host_edit)
        lw_layout_2.addRow(self.omero_port_label, self.omero_port_edit)
        lw_layout_1.addLayout(lw_layout_2)
        lw_layout_1.addWidget(self.omero_connect_button)
        self.omero_login_dialog.setLayout(lw_layout_1)

        # OMERO box layout
        paths_box, path_box_layout = QGroupBox("File && training set selection"), QFormLayout()
        path_box_layout.addRow(self.omero_project_button, self.omero_project_edit)
        path_box_layout.addRow(self.omero_dataset_button, self.omero_dataset_edit)
        path_box_layout.addRow(self.omero_file_button, self.omero_file_text_edit)
        path_box_layout.addRow(self.omero_trainset_button, self.omero_trainset_text_edit)
        paths_box.setLayout(path_box_layout)

        # OMERO project selection layout
        omero_project_selection_layout = QVBoxLayout()
        omero_project_selection_layout.addWidget(self.omero_project_list)
        omero_project_selection_all_layout = QHBoxLayout()
        omero_project_selection_all_layout.addWidget(self.omero_project_selection_select_all_button)
        omero_project_selection_all_layout.addWidget(self.omero_project_selection_deselect_all_button)
        omero_project_selection_layout.addLayout(omero_project_selection_all_layout)
        omero_project_selection_layout.addWidget(self.omero_project_selection_button)
        self.omero_project_selection.setLayout(omero_project_selection_layout)

        # OMERO dataset selection layout
        omero_dataset_selection_layout = QVBoxLayout()
        omero_dataset_selection_layout.addWidget(self.omero_dataset_list)
        omero_dataset_selection_all_layout = QHBoxLayout()
        omero_dataset_selection_all_layout.addWidget(self.omero_dataset_selection_select_all_button)
        omero_dataset_selection_all_layout.addWidget(self.omero_dataset_selection_deselect_all_button)
        omero_dataset_selection_layout.addLayout(omero_dataset_selection_all_layout)
        omero_dataset_selection_layout.addWidget(self.omero_dataset_selection_button)
        self.omero_dataset_selection.setLayout(omero_dataset_selection_layout)

        # OMERO file selection layout
        omero_file_selection_layout = QVBoxLayout()
        omero_file_selection_layout.addWidget(self.omero_file_list)
        omero_file_selection_all_layout = QHBoxLayout()
        omero_file_selection_all_layout.addWidget(self.omero_file_selection_select_all_button)
        omero_file_selection_all_layout.addWidget(self.omero_file_selection_deselect_all_button)
        omero_file_selection_layout.addLayout(omero_file_selection_all_layout)
        omero_file_selection_layout.addWidget(self.omero_file_selection_button)
        self.omero_file_selection.setLayout(omero_file_selection_layout)

        # OMERO trainset selection layout
        omero_trainset_selection_layout = QVBoxLayout()
        omero_trainset_selection_layout.addWidget(self.omero_trainset_list)
        omero_new_trainset_box = QGroupBox("New train and test set")
        omero_new_trainset_layout = QVBoxLayout()
        crop_size_layout_1, crop_size_layout_2 = QHBoxLayout(), QHBoxLayout()
        crop_size_layout_1.addWidget(self.crop_size_128_rbutton)
        crop_size_layout_1.addWidget(self.crop_size_256_rbutton)
        crop_size_layout_1.addWidget(self.crop_size_320_rbutton)
        crop_size_layout_2.addWidget(self.crop_size_512_rbutton)
        crop_size_layout_2.addWidget(self.crop_size_768_rbutton)
        crop_size_layout_2.addWidget(self.crop_size_1024_rbutton)
        omero_new_trainset_layout.addWidget(self.crop_size_label)
        omero_new_trainset_layout.addLayout(crop_size_layout_1), omero_new_trainset_layout.addLayout(crop_size_layout_2)
        omero_new_trainset_name_layout = QHBoxLayout()
        omero_new_trainset_name_layout.addWidget(self.omero_new_trainset_name_label)
        omero_new_trainset_name_layout.addWidget(self.omero_new_trainset_name_edit)
        omero_new_trainset_name_layout.addWidget(self.omero_new_trainset_add_button)
        omero_new_trainset_layout.addLayout(omero_new_trainset_name_layout)
        omero_new_trainset_box.setLayout(omero_new_trainset_layout)
        omero_trainset_selection_layout.addWidget(omero_new_trainset_box)
        omero_trainset_selection_layout.addWidget(self.omero_trainset_selection_button)
        self.omero_trainset_selection.setLayout(omero_trainset_selection_layout)

        # Settings box: Color channel selection layout
        settings_box = QGroupBox("Settings")
        color_channel_layout = QHBoxLayout()
        color_channel_layout.setAlignment(Qt.AlignLeft)
        color_channel_layout.addWidget(self.color_channel_label)
        color_channel_layout.addWidget(self.color_channel_1_rbutton)
        color_channel_layout.addWidget(self.color_channel_2_rbutton)
        color_channel_layout.addWidget(self.color_channel_3_rbutton)
        color_channel_layout.addWidget(self.color_channel_rgb_rbutton)

        # Settings box: device selection layout
        device_layout = QHBoxLayout()
        device_layout.addWidget(self.device_label)
        device_layout.addWidget(self.device_cpu_rbutton)
        device_layout.addWidget(self.device_gpu_rbutton)

        # Settings box: group selection layout
        group_selection_layout = QVBoxLayout()
        group_selection_layout.addWidget(self.group_list)
        group_selection_layout.addWidget(self.group_selection_button)
        self.group_selection.setLayout(group_selection_layout)

        # Settings box layout
        settings_box_layout = QVBoxLayout()
        settings_box_layout.addLayout(color_channel_layout)
        settings_box_layout.addLayout(device_layout)
        settings_box_layout.addWidget(self.device_multi_gpu_checkbox)
        settings_box_layout.addWidget(self.group_button)
        # settings_box_layout.addStretch(1)
        settings_box.setLayout(settings_box_layout)

        # Button row
        bottom_buttons_layout = QHBoxLayout()
        bottom_buttons_layout.addWidget(self.train_data_button)
        bottom_buttons_layout.addWidget(self.train_button)
        bottom_buttons_layout.addWidget(self.eval_button)
        bottom_buttons_layout.addWidget(self.inference_button)

        # Main Layout
        main_layout = QGridLayout()
        main_layout.addWidget(paths_box, 0, 0)
        main_layout.addWidget(settings_box, 0, 1)
        main_layout.addLayout(bottom_buttons_layout, 1, 0, 1, 2)
        main_layout.addWidget(self.output_edit, 2, 0, 1, 2)
        main_layout.addWidget(self.status_bar, 3, 0, 1, 2)
        main_layout.setRowStretch(1, 1)
        main_layout.setRowStretch(2, 1)
        main_layout.setColumnStretch(0, 3)
        main_layout.setColumnStretch(1, 1)
        self.setLayout(main_layout)

        # Submenu - train data: add and annotate crops layout
        add_crops_box = QGroupBox("Add and annotate crops")
        add_crops_layout = QVBoxLayout()
        add_crops_layout.addWidget(self.train_data_all_frames_checkbox)
        add_crops_layout.addWidget(self.train_data_prelabel_checkbox)
        add_crops_prelabel_model_layout = QHBoxLayout()
        add_crops_prelabel_model_layout.addWidget(self.train_data_prelabel_model_button)
        add_crops_prelabel_model_layout.addWidget(self.train_data_prelabel_model_edit)
        add_crops_layout.addLayout(add_crops_prelabel_model_layout)
        crop_data_set_selection_layout = QHBoxLayout()
        crop_data_set_selection_layout.addWidget(self.crop_data_set_selection_label)
        crop_data_set_selection_layout.addWidget(self.crop_data_train_set_checkbox)
        crop_data_set_selection_layout.addWidget(self.crop_data_val_set_checkbox)
        crop_data_set_selection_layout.addWidget(self.crop_data_test_set_checkbox)
        add_crops_layout.addLayout(crop_data_set_selection_layout)
        add_crops_layout.addWidget(self.train_data_crop_button)
        add_crops_layout.addWidget(self.train_data_annotate_button)
        add_crops_box.setLayout(add_crops_layout)

        # Submenu - prelabel model selection layout
        prelabel_model_selection_layout = QVBoxLayout()
        prelabel_model_selection_layout.addWidget(self.prelabel_model_list)
        prelabel_model_selection_layout.addWidget(self.prelabel_model_selection_button)
        self.prelabel_model_selection.setLayout(prelabel_model_selection_layout)

        # Submenu - train data: import data layout
        import_data_box = QGroupBox("Import annotated data set")
        import_data_layout = QVBoxLayout()
        import_data_layout.addWidget(self.import_data_normalization_checkbox)
        import_data_set_selection_layout = QHBoxLayout()
        import_data_set_selection_layout.addWidget(self.import_data_set_selection_label)
        import_data_set_selection_layout.addWidget(self.import_data_train_set_checkbox)
        import_data_set_selection_layout.addWidget(self.import_data_val_set_checkbox)
        import_data_set_selection_layout.addWidget(self.import_data_test_set_checkbox)
        import_data_layout.addLayout(import_data_set_selection_layout)
        import_data_layout.addWidget(self.import_data_button)
        import_data_layout.addWidget(self.import_data_progress_bar)
        import_data_box.setLayout(import_data_layout)

        # Submenu - train data: export data layout
        export_data_box = QGroupBox("Export training set")
        export_data_layout = QVBoxLayout()
        export_data_layout.addWidget(self.export_data_button)
        export_data_layout.addWidget(self.export_data_progress_bar)
        export_data_box.setLayout(export_data_layout)

        # Submenu - train data layout
        train_data_layout = QVBoxLayout()
        train_data_layout.addWidget(add_crops_box)
        train_data_layout.addWidget(import_data_box)
        train_data_layout.addWidget(export_data_box)
        self.train_data.setLayout(train_data_layout)

        # Submenu - train data - crop selection
        select_crops_layout = QVBoxLayout()
        select_crops_layout.addWidget(self.select_crops_filename_label)
        select_crops_counter_layout = QHBoxLayout()
        select_crops_counter_layout.setAlignment(Qt.AlignLeft)
        select_crops_counter_layout.addWidget(self.select_crops_num_crops_label)
        select_crops_counter_layout.addWidget(self.select_crops_counter_label)
        select_crops_layout.addLayout(select_crops_counter_layout)
        select_crops_layout.addWidget(self.select_crops_overlay_checkbox)
        select_crops_img_layout = QGridLayout()
        select_crops_img_layout.addWidget(self.select_crops_img_left, 0, 0)
        select_crops_img_layout.addWidget(self.select_crops_img_center, 0, 1)
        select_crops_img_layout.addWidget(self.select_crops_img_right, 0, 2)
        select_crops_img_layout.addWidget(self.select_crops_img_left_checkbox, 1, 0, Qt.AlignCenter)
        select_crops_img_layout.addWidget(self.select_crops_img_center_checkbox, 1, 1, Qt.AlignCenter)
        select_crops_img_layout.addWidget(self.select_crops_img_right_checkbox, 1, 2, Qt.AlignCenter)
        select_crops_layout.addLayout(select_crops_img_layout)
        vertical_spacer = QSpacerItem(3, 3, QSizePolicy.Minimum, QSizePolicy.Expanding)
        select_crops_layout.addSpacerItem(vertical_spacer)
        select_crops_layout.addWidget(self.select_crops_accept_button)
        select_crops_layout.addSpacerItem(vertical_spacer)
        select_crops_manual_layout = QHBoxLayout()
        select_crops_manual_layout.setAlignment(Qt.AlignLeft)
        select_crops_manual_layout.addWidget(self.select_crops_unchecked_checkbox)
        select_crops_manual_layout.addWidget(self.select_crops_partially_checkbox)
        select_crops_manual_layout.addWidget(self.select_crops_checked_checkbox)
        select_crops_layout.addLayout(select_crops_manual_layout)
        self.select_crops.setLayout(select_crops_layout)

        # Submenu - train settings: method selection layout
        train_method_box = QGroupBox("Method")
        train_method_layout = QVBoxLayout()
        train_method_layout.addWidget(self.train_settings_method_boundary_rbutton)
        train_method_layout.addWidget(self.train_settings_method_distance_rbutton)
        train_method_box.setLayout(train_method_layout)

        # Submenu - train settings: batch size and iterations layout
        train_batch_size_layout = QHBoxLayout()
        train_batch_size_layout.addWidget(self.train_settings_batchsize_label)
        train_batch_size_layout.addWidget(self.train_settings_batchsize_edit)
        train_iterations_layout = QHBoxLayout()
        train_iterations_layout.addWidget(self.train_settings_iterations_label)
        train_iterations_layout.addWidget(self.train_settings_iterations_line_edit)

        # Submenu - train settings: optimizer layout
        train_optimizer_layout = QHBoxLayout()
        train_optimizer_layout.addWidget(self.train_settings_optimizer_label)
        train_optimizer_layout.addWidget(self.train_settings_optimizer_adam_rbutton)
        train_optimizer_layout.addWidget(self.train_settings_optimizer_ranger_rbutton)

        # Submenu - train settings layout
        train_layout = QVBoxLayout()
        train_layout.addLayout(train_batch_size_layout)
        train_layout.addLayout(train_iterations_layout)
        train_layout.addLayout(train_optimizer_layout)
        train_layout.addWidget(train_method_box)
        train_layout.addWidget(self.train_settings_train_button)
        train_layout.addWidget(self.train_export_data_progress_bar)
        train_layout.addWidget(self.train_create_labels_progress_bar)
        train_layout.addWidget(self.train_training_progress_bar)
        train_layout.addWidget(self.train_progress_edit)
        self.train_settings.setLayout(train_layout)

        # Submenu - inference layout
        inference_predict_box = QGroupBox("Predict")
        inference_predict_layout = QVBoxLayout()

        inference_model_layout = QHBoxLayout()
        inference_model_layout.addWidget(self.inference_menu_model_button)
        inference_model_layout.addWidget(self.inference_menu_model_edit)

        inference_batch_size_layout = QHBoxLayout()
        inference_batch_size_layout.addWidget(self.inference_batch_size_label)
        inference_batch_size_layout.addWidget(self.inference_batch_size_edit)

        inference_predict_layout.addLayout(inference_model_layout)
        inference_predict_layout.addLayout(inference_batch_size_layout)
        inference_predict_layout.addWidget(self.inference_menu_upload_checkbox)
        inference_predict_layout.addWidget(self.inference_menu_overwrite_checkbox)
        inference_predict_layout.addWidget(self.inference_menu_sliding_window_checkbox)
        inference_predict_layout.addWidget(self.inference_menu_process_button)
        inference_predict_layout.addWidget(self.inference_progress_bar)
        inference_predict_box.setLayout(inference_predict_layout)

        inference_correct_box = QGroupBox("Correct and analyze results")
        inference_correct_layout = QVBoxLayout()
        inference_correct_layout.addWidget(self.inference_menu_correct_button)
        inference_correct_layout.addWidget(self.inference_menu_analyze_button)
        inference_correct_layout.addWidget(self.analysis_progress_bar)
        inference_correct_box.setLayout(inference_correct_layout)

        inference_export_box = QGroupBox("Export data and results")
        inference_export_layout = QVBoxLayout()
        inference_export_layout.addWidget(self.inference_menu_export_button)
        inference_export_layout.addWidget(self.result_export_progress_bar)
        inference_export_box.setLayout(inference_export_layout)

        inference_layout = QVBoxLayout()
        inference_layout.addWidget(inference_predict_box)
        inference_layout.addWidget(inference_correct_box)
        inference_layout.addWidget(inference_export_box)
        self.inference_menu.setLayout(inference_layout)

        # Submenu - inference model selection layout
        inference_model_selection_layout = QVBoxLayout()
        inference_model_selection_layout.addWidget(self.inference_model_list)
        inference_model_selection_layout.addWidget(self.inference_model_selection_button)
        self.inference_model_selection.setLayout(inference_model_selection_layout)

        # Submenu - evaluation layout
        evaluation_layout = QVBoxLayout()
        evaluation_model_selection_box = QGroupBox("Select models for evaluation")
        evaluation_model_selection_vlayout = QVBoxLayout()
        evaluation_model_selection_vlayout.addWidget(self.evaluation_model_list)
        evaluation_model_selection_hlayout = QHBoxLayout()
        evaluation_model_selection_hlayout.addWidget(self.evaluation_model_list_select_all_button)
        evaluation_model_selection_hlayout.addWidget(self.evaluation_model_list_deselect_all_button)
        evaluation_model_selection_vlayout.addLayout(evaluation_model_selection_hlayout)
        evaluation_model_selection_box.setLayout(evaluation_model_selection_vlayout)
        evaluation_eval_box = QGroupBox("Evaluate selected models")
        evaluation_eval_layout = QVBoxLayout()
        evaluation_batch_size_layout = QHBoxLayout()
        evaluation_batch_size_layout.addWidget(self.evaluation_batch_size_label)
        evaluation_batch_size_layout.addWidget(self.evaluation_batch_size_edit)
        evaluation_eval_layout.addLayout(evaluation_batch_size_layout)
        evaluation_eval_layout.addWidget(self.evaluation_save_raw_checkbox)
        evaluation_eval_layout.addWidget(self.evaluation_menu_eval_button)
        evaluation_eval_layout.addWidget(self.evaluation_menu_export_data_progress_bar)
        evaluation_eval_layout.addWidget(self.evaluation_menu_eval_data_progress_bar)
        evaluation_eval_box.setLayout(evaluation_eval_layout)
        evaluation_layout.addWidget(evaluation_model_selection_box)
        evaluation_layout.addWidget(evaluation_eval_box)
        self.evaluation_menu.setLayout(evaluation_layout)

    @staticmethod
    def get_dark_palette():
        """ Get palette for dark mode """
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        dark_palette.setColor(QPalette.Disabled, QPalette.Window, Qt.black)
        dark_palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(88, 88, 88))
        dark_palette.setColor(QPalette.Disabled, QPalette.Base, QColor(53, 53, 53))
        return dark_palette

    def init_dark_mode(self):
        """ Initialize dark mode """
        self.setPalette(self.get_dark_palette())
        self.omero_login_dialog.setPalette(self.get_dark_palette())
        self.omero_project_selection.setPalette(self.get_dark_palette())
        self.omero_dataset_selection.setPalette(self.get_dark_palette())
        self.omero_file_selection.setPalette(self.get_dark_palette())
        self.omero_trainset_selection.setPalette(self.get_dark_palette())
        self.train_settings.setPalette(self.get_dark_palette())
        self.group_selection.setPalette(self.get_dark_palette())
        self.train_data.setPalette(self.get_dark_palette())
        self.select_crops.setPalette(self.get_dark_palette())
        self.inference_menu.setPalette(self.get_dark_palette())
        self.inference_model_selection.setPalette(self.get_dark_palette())
        self.prelabel_model_selection.setPalette(self.get_dark_palette())
        self.evaluation_menu.setPalette(self.get_dark_palette())

    def closeEvent(self, event):
        """ Close event """
        if self.is_ready():
            event.accept()
        else:
            self.message_box(title='Gui busy', text='Currently some calculations are done. Please stop the calculations'
                                                    ' first before closing microbeSEG (with ctrl+c in the corresponding'
                                                    ' submenu) or wait until they are finished.')
            event.ignore()

    def connect(self):
        """ Connect to OMERO server """
        self.conn = BlitzGateway(self.omero_username_edit.text(), self.omero_password_edit.text(),
                                 host=self.omero_host_edit.text(), port=self.omero_port_edit.text(),
                                 secure=True)

        try:
            conn_status = self.conn.connect()
            if self.group_id is not None:
                self.conn.setGroupForSession(self.group_id)
        except:
            self.message_box(title='OMERO error', text='No OMERO server found. Check inputs or connection.')
            return False
        else:
            if not conn_status:
                self.message_box(title='OMERO error', text='OMERO server found but no connection possible. '
                                                           'Check inputs.')
            return conn_status

    def crop_creation_started(self):
        """ Set inference state to True if pre-labeling is applied """
        if self.train_data_prelabel_checkbox.isChecked():
            self.infer_state = True
        self.status_bar.showMessage('Busy')
        self.create_crops = True
        self.select_crops_img_left.setVisible(False)
        self.select_crops_img_center.setVisible(False)
        self.select_crops_img_right.setVisible(False)
        self.select_crops_img_left_checkbox.setVisible(False)
        self.select_crops_img_center_checkbox.setVisible(False)
        self.select_crops_img_right_checkbox.setVisible(False)
        if self.crop_size <= 320:  # probably 3 crops
            self.select_crops.setMinimumWidth(int(3 * self.crop_size))
        elif self.crop_size <= 768:  # probably 2-3 crops
            self.select_crops.setMinimumWidth(int(2.25 * self.crop_size))
        else:  # probably 1-2 crop
            self.select_crops.setMinimumWidth(int(1.5 * self.crop_size))
        self.select_crops.setMinimumHeight(self.crop_size)
        QApplication.processEvents()

    def crop_creation_finished(self):
        """ Set inference state to False if pre-labeling has been applied """
        if self.train_data_prelabel_checkbox.isChecked():
            self.infer_state = False
        if self.is_ready():
            self.status_bar.showMessage('Ready')
        self.create_crops = False

    def data_export_finished(self):
        """ Set export state to False after import """
        self.export_data_state = False
        if self.is_ready():
            self.status_bar.showMessage('Ready')

    def data_export_started(self):
        """ Set export state to True """
        self.export_data_state = True
        self.status_bar.showMessage('Busy')

    def data_import_finished(self):
        """ Set import state to False after import """
        self.import_data_state = False
        if self.is_ready():
            self.status_bar.showMessage('Ready')

    def data_import_started(self):
        """ Set import state to True """
        self.import_data_state = True
        self.status_bar.showMessage('Busy')
        self.import_data_progress_bar.setValue(0)
        self.import_data_progress_bar.show()

    def disconnect(self):
        """ Disconnect from OMERO server"""
        try:
            self.conn.close()
        except:
            pass  # probably already closed

    def eval_button_clicked(self):
        """ Open evaluation menu """

        # Check if training and test set has been selected:
        if not self.trainset_id:
            self.message_box(title='Training Set Error', text='Select a training set first')
            return

        # Get all available models
        self.trained_model_list = []
        for model_dir in self.model_path.iterdir():
            if model_dir.stem == self.trainset:
                preselect = True
            else:
                preselect = False
            for model in sorted(model_dir.glob('*.pth')):
                if (model.parent / "{}.json".format(model.stem)).is_file():  # without json file model cannot be loaded
                    self.trained_model_list.append([model,
                                                    preselect,
                                                    "{}: {}".format(model.parent.stem, model.stem)])

        # Preselected models should be on the top of the list
        self.trained_model_list = sorted(self.trained_model_list, key=lambda hlist: not hlist[1])

        # Fill list
        self.evaluation_model_list.clear()
        for idx, model in enumerate(self.trained_model_list):
            self.evaluation_model_list.addItem(model[-1])
            if model[1]:
                self.evaluation_model_list.item(idx).setSelected(True)

        self.evaluation_menu.show()

    def eval_menu_eval_button_clicked(self):
        """ Evaluate all trained models """
        # Avoid evaluation of multiple dataset at once (or when a model is trained)
        if not self.is_ready():
            self.message_box(title='GUI busy', text='Try again when current calculation is finished.')
            return

        # Get selected models
        models = []
        if len(self.evaluation_model_list.selectedItems()) > 0:
            print_message = ""
            for model in self.evaluation_model_list.selectedItems():
                for i in range(len(self.trained_model_list)):
                    if self.trained_model_list[i][-1] == model.text():
                        models.append(self.trained_model_list[i][0])
                print_message += "  '{}'\n".format(model.text())
            # self.output_edit.append("\nEvaluate\n{} on training set '{}'".format(print_message, self.trainset))
        else:
            self.message_box(title='Model selection error', text="Select at least one model for evaluation!")
            return

        # Get paths and remove existing train/val/test set (could be old
        trainset_path = self.set_up_train_val_test_dirs()

        # Reset and show progress bars
        self.evaluation_menu_export_data_progress_bar.setValue(0), self.evaluation_menu_export_data_progress_bar.show()
        self.evaluation_menu_eval_data_progress_bar.setValue(0), self.evaluation_menu_eval_data_progress_bar.show()

        # Get worker threads and workers for data export and evaluation
        self.export_thread, self.export_worker = QThread(parent=self), DataExportWorker()
        self.eval_thread, self.eval_worker = QThread(parent=self), EvalWorker()
        self.export_worker.moveToThread(self.export_thread)
        self.eval_worker.moveToThread(self.eval_thread)

        # Connect data export signals
        self.export_thread.started.connect(self.data_export_started)
        self.export_thread.started.connect(partial(self.export_worker.export_data,
                                                   self.trainset_id,
                                                   self.train_path,
                                                   self.omero_username_edit.text(),
                                                   self.omero_password_edit.text(),
                                                   self.omero_host_edit.text(),
                                                   self.omero_port_edit.text(),
                                                   self.group_id))
        self.stop_evaluation_sc.activated.connect(self.export_worker.stop_export_process)
        self.export_worker.text_output.connect(self.get_worker_information)
        self.export_worker.progress.connect(self.evaluation_menu_export_data_progress_bar.setValue)
        self.export_worker.finished.connect(self.data_export_finished)
        self.export_worker.finished.connect(self.export_thread.quit)
        self.export_worker.finished.connect(self.export_worker.deleteLater)
        self.export_thread.finished.connect(self.eval_thread.start)
        self.export_thread.finished.connect(self.export_thread.deleteLater)

        # Connect evaluation signals
        self.eval_thread.started.connect(self.eval_started)
        self.eval_thread.started.connect(partial(self.eval_worker.start_evaluation,
                                                 trainset_path,
                                                 self.eval_path / self.trainset,
                                                 models,
                                                 int(self.evaluation_batch_size_edit.text()),
                                                 self.get_device(),
                                                 self.get_num_gpus(),
                                                 self.evaluation_save_raw_checkbox.isChecked(),
                                                 "Evaluate\n{} on training set '{}'".format(print_message,
                                                                                            self.trainset)))
        self.stop_evaluation_sc.activated.connect(self.eval_worker.stop_evaluation_process)
        self.eval_worker.text_output.connect(self.get_worker_information)
        self.eval_worker.progress.connect(self.evaluation_menu_eval_data_progress_bar.setValue)
        self.eval_worker.finished.connect(self.eval_finished)
        self.eval_worker.finished.connect(self.eval_thread.quit)
        self.eval_worker.finished.connect(self.eval_worker.deleteLater)
        self.eval_thread.finished.connect(self.eval_thread.deleteLater)

        # Start the export thread (which starts the evaluation thread)
        self.export_thread.start()

    def evaluation_model_list_select_all_button_clicked(self):
        """ Select all available models """
        self.evaluation_model_list.selectAll()

    def evaluation_model_list_deselect_all_button_clicked(self):
        """ Clear model selection """
        self.evaluation_model_list.clearSelection()

    def eval_started(self):
        """ Set evaluation state to True """
        self.eval_state = True
        self.status_bar.showMessage('Busy')

    def eval_finished(self):
        """ Set eval state to False after evaluation """
        self.eval_state = False
        if self.is_ready():
            self.status_bar.showMessage('Ready')

    def get_color_channel(self):
        """ Get selected color channel """
        if self.color_channel_1_rbutton.isChecked():
            self.color_channel = 0
        elif self.color_channel_2_rbutton.isChecked():
            self.color_channel = 1
        elif self.color_channel_3_rbutton.isChecked():
            self.color_channel = 2
        elif self.color_channel_rgb_rbutton.isChecked():
            self.color_channel = 'rgb'

    @staticmethod
    def get_crop_key_value_data(crop_dict):
        """ Get metadata to attach to a selected image crop.

        :param crop_dict: Information about the crop
        :type crop_dict: dict
        :return: list of key value pairs
        """
        key_value_data = []
        for key in crop_dict:
            if key in ['img', 'img_show', 'roi', 'roi_show']:
                continue
            key_value_data.append([key, str(crop_dict[key])])
        return key_value_data

    def get_crop_size(self):
        """ Get selected crop size """
        if self.crop_size_128_rbutton.isChecked():
            self.crop_size = 128
        elif self.crop_size_256_rbutton.isChecked():
            self.crop_size = 256
        elif self.crop_size_320_rbutton.isChecked():
            self.crop_size = 320
        elif self.crop_size_512_rbutton.isChecked():
            self.crop_size = 512
        elif self.crop_size_768_rbutton.isChecked():
            self.crop_size = 768
        elif self.crop_size_1024_rbutton.isChecked():
            self.crop_size = 1024

    def get_crop_train_split(self):
        """ Get probabilities for the train/val/test set assignment of a created crop

        :return: probability for train, probability for val, probability for test
        """
        p_train, p_val, p_test = 0, 0, 0
        if self.crop_data_train_set_checkbox.isChecked():
            p_train = 1
            if self.crop_data_val_set_checkbox.isChecked():
                p_train = 0.8
                p_val = 0.2
            if self.crop_data_test_set_checkbox.isChecked():
                p_train *= 0.75
                p_val *= 0.75
                p_test = 0.25
        elif self.crop_data_val_set_checkbox.isChecked():
            p_val = 1
            if self.crop_data_test_set_checkbox.isChecked():
                p_val *= 0.75
                p_test = 0.25
        elif self.crop_data_test_set_checkbox.isChecked():
            p_test = 1
        return p_train, p_val, p_test

    def get_device(self):
        """ Get selected device

        :return: torch device
        """
        if self.device_gpu_rbutton.isChecked():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return device

    def get_import_train_split(self):
        """ Get probabilities for the train/val/test set assignment of an imported image

        :return: probability for train, probability for val, probability for test
        """
        p_train, p_val, p_test = 0, 0, 0
        if self.import_data_train_set_checkbox.isChecked():
            p_train = 1
            if self.import_data_val_set_checkbox.isChecked():
                p_train = 0.8
                p_val = 0.2
            if self.import_data_test_set_checkbox.isChecked():
                p_train *= 0.75
                p_val *= 0.75
                p_test = 0.25
        elif self.import_data_val_set_checkbox.isChecked():
            p_val = 1
            if self.import_data_test_set_checkbox.isChecked():
                p_val *= 0.75
                p_test = 0.25
        elif self.import_data_test_set_checkbox.isChecked():
            p_test = 1
        return p_train, p_val, p_test

    def get_num_gpus(self):
        """ Get number of gpus to use

        :return: number of gpus (int)
        """
        num_gpus = 0
        if self.device_multi_gpu_checkbox.isChecked():
            num_gpus = torch.cuda.device_count()
        elif self.device_gpu_rbutton.isChecked():
            num_gpus = 1
        return num_gpus

    def get_optimizer(self):
        """ Get selected optimizer

        :return: optimizer (str)
        """

        if self.train_settings_optimizer_adam_rbutton.isChecked():
            optimizer = 'adam'
        elif self.train_settings_optimizer_ranger_rbutton.isChecked():
            optimizer = 'ranger'
        else:
            raise ValueError("No optimizer selected")

        return optimizer

    def get_segmentation_method(self):
        """ Get selected segmentation method

        :return: segmentation method / label type
        """
        if self.train_settings_method_boundary_rbutton.isChecked():
            label_type = 'boundary'
        elif self.train_settings_method_distance_rbutton.isChecked():
            label_type = 'distance'
        else:
            self.message_box(title='Method Selection Error', text='Select a method!')
            label_type = None
        return label_type

    def get_trained_models(self, scores_df):
        """ Get sorted list of trained models

        :param scores_df: Dataframe with scores and thresholds
        :type scores_df: pandas DataFrame
        :return: sorted (with scores) list of trained models
        """

        # Walk through model dir. If model has score: show score
        models = []
        for sub_dir in self.model_path.iterdir():
            models.extend(sorted(sub_dir.glob('*.json')))
        trained_models = []
        for model in models:
            if "{}: {}".format(model.parent.stem, model.stem) in scores_df.model.values:
                model_row = scores_df.loc[scores_df['model'] == "{}: {}".format(model.parent.stem, model.stem)]
                trained_models.append(["{}: {} ({:.2f}+/-{:.2f})".format(model.parent.stem,
                                                                         model.stem,
                                                                         model_row['aji+ (mean)'].values[0],
                                                                         model_row['aji+ (std)'].values[0]),
                                       model,
                                       model_row['aji+ (mean)'].values[0]])
            else:
                trained_models.append(["{}: {} (-)".format(model.parent.stem, model.stem), model, 0])
        trained_models = sorted(trained_models, key=lambda score: score[-1], reverse=True)

        return trained_models

    def get_training_information(self, text_input):
        """ Print data import information """
        self.train_progress_edit.append(text_input)

    def get_worker_information(self, text_input):
        """ Print data import information """
        self.output_edit.append(text_input)

    def group_button_clicked(self):
        """ Select available OMERO projects """

        # Connect to server and fill QListWidget with available groups
        conn_status = self.connect()
        if not conn_status:
            return

        # Clear QListWidget
        self.group_list.clear()

        # Get groups of user
        group_list = []
        for g in self.conn.getGroupsMemberOf():
            group_list.append([g.getName(), g.getId()])
        group_list = sorted(group_list, key=lambda x: x[0].lower())

        # Fill QListWidget
        idx = 0
        for group_name, group_id in group_list:
            self.group_list.addItem(group_name)
            if len(self.group_list) > 0 and group_id == self.group_id:
                self.group_list.item(idx).setSelected(True)
            idx += 1

        # Run project selection menu
        self.group_selection.exec()

        # Check which projects/... have been selected
        selected_group = [group_list[list_item.row()][1] for list_item in self.group_list.selectedIndexes()][0]
        if selected_group != self.group_id:
            self.omero_project_list.clear(), self.omero_project_list.clearSelection(), self.omero_project_edit.clear()
            self.omero_dataset_list.clear(), self.omero_dataset_list.clearSelection(), self.omero_dataset_edit.clear()
            self.omero_file_list.clear(), self.omero_file_list.clearSelection(), self.omero_file_text_edit.clear()
            self.projects, self.datasets, self.files, self.file_ids = [], [], [], []
            self.omero_trainset_list.clear(), self.omero_trainset_text_edit.clear()
            self.trainset, self.trainset_id, self.trainset_project_id, self.trainset_length = None, None, None, 0

            # Set selected group
            self.group_id = selected_group

        self.disconnect()

    def group_selection_button_clicked(self):
        """ Select group and close window """
        self.group_selection.close()

    def help_shortcut_clicked(self):
        """ Open help """
        text = 'File & training set selection:\n'\
               '   Project(s): select your OMERO project(s).\n'\
               '   Dataset(s): select your OMERO dataset(s).\n'\
               '   File(s): select your OMERO file(s).\n'\
               '   Training set: select or create a training set.\n\n'\
               'Settings:\n'\
               '   Channel: image/color channel to process.\n'\
               '   Device: device to use for training/evaluation/inference.\n'\
               '   Use multiple gpus: use multiple gpus (training only).\n\n'\
               'Training data:\n'\
               '   Pre-labeling: show (and upload) preliminary segmentation.\n' \
               '   Pre-labeling model: select model for pre-labeling.\n'\
               '   Create crops: create crops from the selected files.\n'\
               '   Annotate: open image annotation tool in browser.\n' \
               '   Import: import annotated data set into selected training set.\n' \
               '   Export: export training data set.\n\n'\
               'Training:\n'\
               '   Batch size: training batch size (reduce for small memory).\n'\
               '   Iterations: number of models to train.\n'\
               '   Optimizer: select optimizer for the training.\n'\
               '   Method: select method.\n'\
               '   Train: train new models on the selected training set.\n\n'\
               'Evaluation:\n'\
               '   Save raw predictions: Save raw CNN outputs.\n'\
               '   Evaluate: calculate metrics for selected models on the test set.\n\n'\
               'Inference:\n'\
               '   Model: select model for inference.\n' \
               '   Upload to OMERO: enable to upload results.\n'\
               '   Overwrite: Enable to overwrite existing results.\n'\
               '   Process selected files: start inference.\n' \
               '   Correct: open annotation tool for corrections.\n' \
               '   Analyze: analyize selected and processed files (cell counts/...)\n'\
               '   Export: export data and results.\n\n'\
               'Press "ctrl + c" in the corresponding menu to stop training/evaluation/inference.\n\n'\
               'Contact: tim.scherr@kit.edu\n\n'
        self.message_box(title='Help', text=text)

    def inference_button_clicked(self):
        """ Inference using best model """

        # Check if some files are selected
        if not self.file_ids:
            self.message_box(title='File Error', text='Select some files!')
            return

        # Check if training set is selected for showing evaluation scores
        if not self.trainset_id:
            self.message_box(title='Warning', text='No training set selected. No evaluation scores can be viewed')
            self.scores_df = pd.DataFrame(columns=['model'])
        else:
            if (self.eval_path / '{}.csv'.format(self.trainset)).is_file():
                self.scores_df = pd.read_csv(self.eval_path / '{}.csv'.format(self.trainset))
            else:
                self.scores_df = pd.DataFrame(columns=['model'])
                self.message_box(title='Warning', text='No evaluation scores for the selected training set found. '
                                                       'Evaluation scores can only be viewed after evaluation!')

        # Get trained models (and scores)
        self.inference_models = self.get_trained_models(scores_df=self.scores_df)

        if len(self.inference_models) == 0:
            self.message_box(title='Model Warning', text='No trained models found!')

        # Fill QListWidget
        self.inference_model_list.clear()
        self.inference_menu_model_edit.setText('')
        for idx, inference_model in enumerate(self.inference_models):
            self.inference_model_list.addItem(inference_model[0])
            if self.inference_model and inference_model[1] == self.inference_model:
                self.inference_model_list.item(idx).setSelected(True)
                self.inference_menu_model_edit.setText("{}: {}".format(inference_model[1].parent.stem,
                                                                       inference_model[1].stem))
        # Preselect best model (only if evaluated, only if not already manually selected before)
        if self.inference_models and not self.inference_model:
            if self.inference_models[0][-1] > 0:
                self.inference_model_list.item(0).setSelected(True)
                self.inference_model_list.setCurrentRow(0)
                self.inference_model = self.inference_models[0][1]
                self.inference_menu_model_edit.setText("{}: {}".format(self.inference_models[0][1].parent.stem,
                                                                       self.inference_models[0][1].stem))
        if self.inference_model:
            if f"{self.inference_model.parent.stem}: {self.inference_model.stem}" in self.scores_df.model.values:
                model_row = self.scores_df.loc[self.scores_df['model'] == "{}: {}".format(
                    self.inference_model.parent.stem,
                    self.inference_model.stem)]
                self.inference_model_ths = [model_row['th_cell'].values[0], model_row['th_seed'].values[0]]
            else:
                self.inference_model_ths = [0.10, 0.45]  # standard thresholds for distance method
            
        self.inference_menu.show()

    def inference_menu_analyze_button_clicked(self):
        """ Analyze segmented data from OMERO """

        # Avoid export when inference is running
        if self.infer_state:
            self.message_box(title='GUI busy', text='Try again when current inference/export is finished.')
            return

        # Reset and show progress bars
        self.analysis_progress_bar.setValue(0), self.analysis_progress_bar.show()

        # Get worker threads and workers for data export and evaluation
        self.result_analysis_thread = QThread(parent=self)
        self.result_analysis_worker = AnalysisWorker(copy(self.file_ids),
                                                     self.results_path,
                                                     self.omero_username_edit.text(),
                                                     self.omero_password_edit.text(),
                                                     self.omero_host_edit.text(),
                                                     self.omero_port_edit.text(),
                                                     self.group_id)
        self.result_analysis_worker.moveToThread(self.result_analysis_thread)

        # Connect inference signals
        self.result_analysis_thread.started.connect(self.infer_started)
        self.result_analysis_thread.started.connect(self.result_analysis_worker.analyze_data)
        self.stop_inference_sc.activated.connect(self.result_analysis_worker.stop_analysis_process)
        self.result_analysis_worker.text_output.connect(self.get_worker_information)
        self.result_analysis_worker.progress.connect(self.analysis_progress_bar.setValue)
        self.result_analysis_worker.finished.connect(self.infer_finished)
        self.result_analysis_worker.finished.connect(self.result_analysis_thread.quit)
        self.result_analysis_worker.finished.connect(self.result_analysis_worker.deleteLater)
        self.result_analysis_thread.finished.connect(self.result_analysis_thread.deleteLater)

        # Start the export thread (which starts the evaluation thread)
        self.result_analysis_thread.start()

        return 0

    def inference_menu_correct_button_clicked(self):
        """ Open annotation tool """
        self.open_annotation_tool(mode='correct')

    def inference_menu_export_button_clicked(self):
        """ Export segmented data from OMERO """

        # Avoid export when inference is running
        if self.infer_state:
            self.message_box(title='GUI busy', text='Try again when current inference/export is finished.')
            return

        # Reset and show progress bars
        self.result_export_progress_bar.setValue(0), self.result_export_progress_bar.show()

        # Get worker threads and workers for data export and evaluation
        self.result_export_thread = QThread(parent=self)
        self.result_export_worker = ResultExportWorker(copy(self.file_ids),
                                                       self.results_path,
                                                       self.omero_username_edit.text(),
                                                       self.omero_password_edit.text(),
                                                       self.omero_host_edit.text(),
                                                       self.omero_port_edit.text(),
                                                       self.group_id)
        self.result_export_worker.moveToThread(self.result_export_thread)

        # Connect inference signals
        self.result_export_thread.started.connect(self.infer_started)
        self.result_export_thread.started.connect(self.result_export_worker.export_data)
        self.stop_inference_sc.activated.connect(self.result_export_worker.stop_export_process)
        self.result_export_worker.text_output.connect(self.get_worker_information)
        self.result_export_worker.progress.connect(self.result_export_progress_bar.setValue)
        self.result_export_worker.finished.connect(self.infer_finished)
        self.result_export_worker.finished.connect(self.result_export_thread.quit)
        self.result_export_worker.finished.connect(self.result_export_worker.deleteLater)
        self.result_export_thread.finished.connect(self.result_export_thread.deleteLater)

        # Start the export thread (which starts the evaluation thread)
        self.result_export_thread.start()

    def inference_menu_model_button_clicked(self):
        """ Open model selection menu """
        self.inference_model_selection.exec()
        for list_item in self.inference_model_list.selectedIndexes():
            self.inference_model = self.inference_models[list_item.row()][1]
            if f"{self.inference_model.parent.stem}: {self.inference_model.stem}" in self.scores_df.model.values:
                model_row = self.scores_df.loc[self.scores_df['model'] == "{}: {}".format(
                    self.inference_model.parent.stem,
                    self.inference_model.stem)]
                self.inference_model_ths = [model_row['th_cell'].values[0], model_row['th_seed'].values[0]]
            else:
                self.inference_model_ths = [0.10, 0.45]  # standard thresholds for distance method
            self.inference_menu_model_edit.setText("{}: {}".format(self.inference_models[list_item.row()][1].parent.stem,
                                                                   self.inference_models[list_item.row()][1].stem))

    def inference_model_selection_button_clicked(self):
        """ Close model selection menu """
        self.inference_model_selection.close()

    def inference_menu_process_button_clicked(self):
        """ Process selected files with selected model """

        # Avoid evaluation of multiple dataset at once (or when a model is trained)
        if not self.is_ready():
            self.message_box(title='GUI busy', text='Try again when current calculation is finished.')
            return

        # Check if model is selected
        if not self.inference_model:
            self.message_box(title='Model selection error', text="Select a model for inference!")
            return

        # Reset and show progress bars
        self.inference_progress_bar.setValue(0), self.inference_progress_bar.show()

        # Get selected color channel
        self.get_color_channel()

        # Get worker threads and workers for data export and evaluation
        self.infer_thread = QThread(parent=self)
        self.infer_worker = InferWorker(copy(self.file_ids),
                                        self.results_path,
                                        self.omero_username_edit.text(),
                                        self.omero_password_edit.text(),
                                        self.omero_host_edit.text(),
                                        self.omero_port_edit.text(),
                                        self.group_id,
                                        copy(self.inference_model),
                                        self.get_device(),
                                        # self.get_num_gpus(),
                                        copy(self.inference_model_ths),
                                        self.color_channel,
                                        self.inference_menu_upload_checkbox.isChecked(),
                                        self.inference_menu_overwrite_checkbox.isChecked(),
                                        self.inference_menu_sliding_window_checkbox.isChecked())
        self.infer_worker.moveToThread(self.infer_thread)

        # Connect inference signals
        self.infer_thread.started.connect(self.infer_started)
        self.infer_thread.started.connect(self.infer_worker.start_inference)
        self.stop_inference_sc.activated.connect(self.infer_worker.stop_inference_process)
        self.infer_worker.text_output.connect(self.get_worker_information)
        self.infer_worker.progress.connect(self.inference_progress_bar.setValue)
        self.infer_worker.finished.connect(self.infer_finished)
        self.infer_worker.finished.connect(self.infer_thread.quit)
        self.infer_worker.finished.connect(self.infer_worker.deleteLater)
        self.infer_thread.finished.connect(self.infer_thread.deleteLater)

        # Start the export thread (which starts the evaluation thread)
        self.infer_thread.start()

    def infer_started(self):
        """ Set evaluation state to True """
        self.infer_state = True
        self.status_bar.showMessage('Busy')

    def infer_finished(self):
        """ Set eval state to False after evaluation """
        self.infer_state = False
        if self.is_ready():
            self.status_bar.showMessage('Ready')

    def is_ready(self):
        """ Check if gui is ready for (longer) calculations """

        if (self.create_labels_state or self.import_data_state or self.export_data_state or self.train_state
                or self.eval_state or self.infer_state):
            return False
        else:
            return True

    def label_creation_finished(self):
        """ Set label creation state to False after label creation """
        self.create_labels_state = False
        if self.is_ready():
            self.status_bar.showMessage('Ready')

    def label_creation_started(self):
        """ Set label creation state to True """
        self.create_labels_state = True
        self.status_bar.showMessage('Busy')

    def message_box(self, title, text):
        """ Open message box

        :param title: Title of the window
        :type title: str
        :param text: Message
        :type text: str
        :return: None
        """
        box = QMessageBox()
        box.setWindowTitle(title)
        box.setText(text)
        box.setPalette(self.get_dark_palette())
        box.exec()

    def omero_connect_button_clicked(self):
        """ Check login data """
        # Check if server is available
        conn_status = self.connect()
        # Check if username and password are correct
        if conn_status:
            self.output_edit.append('\nConnected to OMERO server.')
            self.group_id = self.conn.getGroupFromContext().getId()
            self.disconnect()
            self.omero_login_dialog.close()
        else:
            self.conn = None

    def omero_project_button_clicked(self):
        """ Select available OMERO projects """
        # Clear some lists
        self.omero_project_list.clear()
        self.omero_dataset_list.clear()
        self.datasets = []
        self.omero_dataset_list.clearSelection()
        self.omero_dataset_edit.clear()
        self.files = []
        self.omero_file_list.clear()
        self.omero_file_list.clearSelection()
        self.omero_file_text_edit.clear()

        # Connect to server and fill QListWidget with available projects
        conn_status = self.connect()
        if not conn_status:
            return
        idx = 0

        project_list = []
        for project in self.conn.getObjects("Project", opts={'group': self.conn.getEventContext().groupId}):
            if project.getName() == 'training sets':  # Project containing the training data and no experiments
                continue
            project_list.append([project.getName(), project])

        # Projects are not sorted alphabetically but with order of creation date
        project_list = sorted(project_list, key=lambda x: x[0])

        for _, project in project_list:
            self.omero_project_list.addItem(project.getName())
            if len(self.projects) > 0 and project.getName() in self.projects:  # preselect already selected items
                self.omero_project_list.item(idx).setSelected(True)
            idx += 1
        self.disconnect()

        # Run project selection menu
        self.omero_project_selection.exec()

        # Check which projects have been selected
        self.projects = [list_item.text() for list_item in self.omero_project_list.selectedItems()]
        self.omero_project_edit.clear()
        if len(self.projects) > 0:
            projects = str(self.projects[0])
            for project in self.projects[1:]:
                projects += ' | {}'.format(project)
            self.omero_project_edit.append(projects)

    def omero_project_selection_select_all_button_clicked(self):
        """ Select all available OMERO projects """
        self.omero_project_list.selectAll()

    def omero_project_selection_deselect_all_button_clicked(self):
        """ Clear OMERO project selection """
        self.omero_project_list.clearSelection()

    def omero_project_selection_button_clicked(self):
        """ Close project selection menu """
        self.omero_project_selection.close()

    def omero_dataset_button_clicked(self):
        """ Select available OMERO datasets (depending on selected projects) """
        # Clear some lists
        self.omero_dataset_list.clear()
        self.files = []
        self.omero_file_list.clear()
        self.omero_file_list.clearSelection()
        self.omero_file_text_edit.clear()

        # Connect to server and fill QListWidget with available datasets
        conn_status = self.connect()
        if not conn_status:
            return
        idx, omero_dataset_project_list = 0, []

        project_list = []
        for project in self.conn.getObjects("Project", opts={'group': self.conn.getEventContext().groupId}):

            if project.getName() == 'training sets':
                continue

            if len(self.projects) > 0 and project.getName() not in self.projects:
                continue

            project_list.append([project.getName(), project])

        # Projects are not sorted alphabetically but with order of creation date
        project_list = sorted(project_list, key=lambda x: x[0])

        for _, project in project_list:

            for dataset in project.listChildren():
                omero_dataset_project_list.append({'dataset': dataset.getName(),
                                                   'dataset_id': dataset.getId(),
                                                   'project': project.getName(),
                                                   'project_id': project.getId()})
                self.omero_dataset_list.addItem('({}) {}'.format(project.getName(), dataset.getName()))
                # Preselect already selected items
                if len(self.datasets) > 0 and {'dataset': dataset.getName(),
                                               'dataset_id': dataset.getId(),
                                               'project': project.getName(),
                                               'project_id': project.getId()} in self.datasets:
                    self.omero_dataset_list.item(idx).setSelected(True)
                idx += 1
        self.disconnect()

        # Run dataset selection menu
        self.omero_dataset_selection.exec()

        # Check which datasets have been selected
        self.datasets = [{'dataset': omero_dataset_project_list[list_idx.row()]['dataset'],
                          'dataset_id': omero_dataset_project_list[list_idx.row()]['dataset_id'],
                          'project': omero_dataset_project_list[list_idx.row()]['project'],
                          'project_id': omero_dataset_project_list[list_idx.row()]['project_id']}
                         for list_idx in self.omero_dataset_list.selectedIndexes()]
        self.omero_dataset_edit.clear()
        if len(self.datasets) > 0:
            if len(self.datasets[0]['dataset']) < 20 or len(self.datasets) == 1:
                datasets = '{}'.format(self.datasets[0]['dataset'])
            else:
                datasets = '{}...{}'.format(self.datasets[0]['dataset'][0:8], self.datasets[0]['dataset'][-8:])
            for dataset in self.datasets[1:]:
                if len(dataset['dataset']) < 20:
                    datasets += ' | {}'.format(dataset['dataset'])
                else:
                    datasets += ' | {}...{}'.format(dataset['dataset'][0:8], dataset['dataset'][-8:])
            self.omero_dataset_edit.append(datasets)

    def omero_dataset_selection_select_all_button_clicked(self):
        """ Select all available OMERO datasets """
        self.omero_dataset_list.selectAll()

    def omero_dataset_selection_deselect_all_button_clicked(self):
        """ Clear OMERO dataset selection """
        self.omero_dataset_list.clearSelection()

    def omero_dataset_selection_button_clicked(self):
        """ Close project selection menu """
        self.omero_dataset_selection.close()

    def omero_file_button_clicked(self):
        """ Select available OMERO files (depends on selected datasets and projects """

        # Clear dataset list and selection
        self.omero_file_list.clear()
        self.omero_file_list.clearSelection()

        # Connect to server and fill QListWidget with available files
        conn_status = self.connect()
        if not conn_status:
            return
        if len(self.datasets) > 0:
            idx, omero_file_list, filename_list = 0, [], []
            for dataset_dict in self.datasets:
                dataset = self.conn.getObject("Dataset", oid=dataset_dict['dataset_id'])
                for file in dataset.listChildren():
                    omero_file_list.append(file.getId())
                    filename_list.append(file.getName())
                    self.omero_file_list.addItem('({}, {}) {}'.format(dataset_dict['project'],
                                                                      dataset.getName(),
                                                                      file.getName()))
                    if len(self.files) > 0 and file.getId() in self.file_ids:
                        self.omero_file_list.item(idx).setSelected(True)
                    idx += 1
        else:  # go through all projects
            idx, omero_file_list, filename_list = 0, [], []
            project_list = []
            for project in self.conn.getObjects("Project", opts={'group': self.conn.getEventContext().groupId}):
                if project.getName() == 'training sets':
                    continue
                if len(self.projects) > 0 and project.getName() not in self.projects:
                    continue

                project_list.append([project.getName(), project])

            # Projects are not sorted alphabetically but with order of creation date
            project_list = sorted(project_list, key=lambda x: x[0])

            for _, project in project_list:
                for dataset in project.listChildren():
                    for file in dataset.listChildren():
                        omero_file_list.append(file.getId())
                        filename_list.append(file.getName())
                        self.omero_file_list.addItem('({}, {}) {}'.format(project.getName(),
                                                                          dataset.getName(),
                                                                          file.getName()))
                        if len(self.files) > 0 and file.getId() in self.file_ids:
                            self.omero_file_list.item(idx).setSelected(True)
                        idx += 1
        self.disconnect()

        # Run file selection menu
        self.omero_file_selection.exec()

        # Check which files have been selected
        self.file_ids = [omero_file_list[list_idx.row()] for list_idx in self.omero_file_list.selectedIndexes()]
        self.files = [filename_list[list_idx.row()] for list_idx in self.omero_file_list.selectedIndexes()]
        self.omero_file_text_edit.clear()
        if len(self.files) > 0:
            if len(self.files[0]) < 20 or len(self.files) == 1:
                filename_string = '{}'.format(self.files[0])
            else:
                filename_string = '{}...{}'.format(self.files[0][0:8], self.files[0][-8:])
            for filename in self.files[1:]:
                if len(filename) < 20:
                    filename_string += ' | {}'.format(filename)
                else:
                    filename_string += ' | {}...{}'.format(filename[0:8], filename[-8:])
            self.omero_file_text_edit.append(filename_string)

    def omero_file_selection_select_all_button_clicked(self):
        """ Select all available OMERO files """
        self.omero_file_list.selectAll()

    def omero_file_selection_deselect_all_button_clicked(self):
        """ Clear OMERO file selection"""
        self.omero_file_list.clearSelection()

    def omero_file_selection_button_clicked(self):
        """ Close file selection menu """
        self.omero_file_selection.close()

    def omero_trainset_button_clicked(self):
        """ Select available OMERO training datasets"""

        # Clear training set list
        self.omero_trainset_list.clear()

        # Connect to server and get or create train set project
        conn_status = self.connect()
        if not conn_status:
            return
        if not self.trainset_project_id:
            for project in self.conn.getObjects("Project", opts={'group': self.conn.getEventContext().groupId}):
                if project.getName() == 'training sets':
                    if not self.conn.canWrite(project):
                        continue
                    else:
                        self.trainset_project_id = project.getId()
            if not self.trainset_project_id:
                trainset_project = ProjectWrapper(self.conn, omero.model.ProjectI())
                trainset_project.setName('training sets')
                trainset_project.save()
                self.trainset_project_id = trainset_project.getId()
        trainset_project = self.conn.getObject("Project", oid=self.trainset_project_id)

        # Fill QListWidget with available training datasets
        for idx, dataset in enumerate(trainset_project.listChildren()):
            self.omero_trainset_list.addItem(dataset.getName())
            if dataset.getName() == self.trainset:  # preselect already selected items
                self.omero_trainset_list.item(idx).setSelected(True)
        self.disconnect()

        # Run trainset selection menu
        self.omero_trainset_selection.exec()

        # Check which dataset has been selected and connect to server to get dataset id
        if len(self.omero_trainset_list.selectedItems()) > 0:
            # other inference model needs to be preselected if evaluation exists
            if self.omero_trainset_list.selectedItems()[0].text() != self.trainset:
                self.inference_model = None
                self.inference_model_ths = None
            self.trainset = self.omero_trainset_list.selectedItems()[0].text()
            self.omero_trainset_text_edit.setText(self.trainset)
            conn_status = self.connect()
            if not conn_status:
                return
            for dataset in self.conn.getObject('Project', oid=self.trainset_project_id).listChildren():
                if dataset.getName() == self.trainset:
                    self.trainset_id = dataset.getId()
            self.disconnect()

    def omero_new_trainset_add_button_clicked(self):
        """ Create new (empty) training set with selected crop size """

        # Check if dataset name has been inserted
        if len(self.omero_new_trainset_name_edit.text()) == 0:
            self.message_box(title='Name Error', text='Insert a (unique) name first!')
            return

        # Check for special characters (windows)
        if any(sc in self.omero_new_trainset_name_edit.text() for sc in r'\/:*?"<>|'):
            self.message_box(title='Name Error', text=r'Do not use special characters: \ / : * ? " < > |')
            return

        # Add crop size
        self.get_crop_size()
        trainset_name = self.omero_new_trainset_name_edit.text()
        if str(self.crop_size) not in trainset_name:
            trainset_name = "{} ({})".format(self.omero_new_trainset_name_edit.text(), self.crop_size)

        # Check if dataset with given name already exists
        for i in range(self.omero_trainset_list.count()):
            if trainset_name == self.omero_trainset_list.item(i).text():
                self.message_box(title='Name Error', text='Insert a unique name!')
                return

        # Connect to server and add new dataset
        conn_status = self.connect()
        if not conn_status:
            return
        # Check if write access is given
        if not self.conn.canWrite(self.conn.getObject("Project", oid=self.trainset_project_id)):
            self.message_box(title="Permission Error", text="Seems like you do not own the existing project 'training "
                                                            "projects' in your group and that you have no write access."
                                                            " Either change the group permissions to 'Read-Write' or "
                                                            "add a new group for which you have write access.")
            return
        new_dataset = DatasetWrapper(self.conn, omero.model.DatasetI())
        new_dataset.setName(trainset_name)
        new_dataset.save()

        # Link new dateset to the 'training sets' project
        link = omero.model.ProjectDatasetLinkI()
        link.setChild(omero.model.DatasetI(new_dataset.getId(), False))
        link.setParent(omero.model.ProjectI(self.trainset_project_id, False))
        self.conn.getUpdateService().saveObject(link)

        # Add crop size to data set as key-value-pair
        key_value_data = [["crop_size", str(self.crop_size)]]
        map_ann = MapAnnotationWrapper(self.conn)
        map_ann.setNs(metadata.NSCLIENTMAPANNOTATION)  # Use 'client' namespace to allow editing in Insight & web
        map_ann.setValue(key_value_data)
        map_ann.save()
        new_dataset.linkAnnotation(map_ann)

        # Add new dataset also to QListWidget
        self.omero_trainset_list.addItem(new_dataset.getName())
        self.omero_trainset_list.item(self.omero_trainset_list.count()-1).setSelected(True)
        self.omero_trainset_list.scrollToBottom()
        self.disconnect()

    def omero_trainset_selection_button_clicked(self):
        """ Close training dataset selection menu """
        self.omero_trainset_selection.close()

    def open_annotation_tool(self, mode='annotate'):
        """ Open the cell annotation tool"""
        if self.annotation_tool_url[-1] != '/':
            self.annotation_tool_url += '/'
        if mode == 'annotate':
            annotation_tool_url = "{}login?u={}&p=&r=/omero-dataset;dataset={}".format(self.annotation_tool_url,
                                                                                       self.omero_username_edit.text(),
                                                                                       self.trainset_id)
        else:
            annotation_tool_url = f"{self.annotation_tool_url}login?u={self.omero_username_edit.text()}&p=&r=" \
                                  f"{urllib.parse.quote(f'omero-dashboard?group={self.group_id}')}"
        webbrowser.open(annotation_tool_url)

    def prelabel_checkbox_clicked(self):
        """ Show and hide pre-labeling buttons """
        self.select_crops_overlay_checkbox.setVisible(self.train_data_prelabel_checkbox.isChecked())
        self.select_crops_partially_checkbox.setVisible(self.train_data_prelabel_checkbox.isChecked())

        if self.train_data_prelabel_checkbox.isChecked():
            self.select_crops_checked_checkbox.setText('upload image and segmentation')
            self.select_crops_img_left_checkbox.setTristate(True)
            self.select_crops_img_center_checkbox.setTristate(True)
            self.select_crops_img_right_checkbox.setTristate(True)
            self.train_data_prelabel_model_edit.setVisible(True)
            self.train_data_prelabel_model_button.setVisible(True)
        else:
            self.select_crops_checked_checkbox.setText('upload image')
            self.select_crops_img_left_checkbox.setTristate(False)
            self.select_crops_img_center_checkbox.setTristate(False)
            self.select_crops_img_right_checkbox.setTristate(False)
            self.train_data_prelabel_model_edit.setVisible(False)
            self.train_data_prelabel_model_button.setVisible(False)

        QApplication.processEvents()
        self.train_data.adjustSize()

    def prelabel_model_selection_button_clicked(self):
        """ Close model selection menu """
        self.prelabel_model_selection.close()

    def train_button_clicked(self):
        """ Open training menu """

        # Check if training and test set has been selected:
        if not self.trainset_id:
            self.message_box(title='Training Set Error', text='Select a training set first')
            return

        # Batch size depends on numbers of gpus
        if self.device_multi_gpu_checkbox.isChecked():
            self.train_settings_batchsize_edit.setText(str(torch.cuda.device_count() * 4))

        # Show train settings menu
        self.train_settings.show()

    def train_data_annotate_button_clicked(self):
        """ Open annotation tool"""

        conn_status = self.connect()
        if not conn_status:
            return
        if (self.train_path / 'split_info.json').exists():
            os.remove(str(self.train_path / 'split_info.json'))
        self.split_info = {}
        for ann in self.conn.getObject("Dataset", self.trainset_id).listAnnotations(ns='split.info.namespace'):
            with open(self.train_path / 'split_info.json', 'wb') as outfile:
                for chunk in ann.getFileInChunks():
                    outfile.write(chunk)
            with open(self.train_path / 'split_info.json', 'r') as infile:
                self.split_info = json.load(infile)
        if not self.split_info or len(self.split_info['used']) == '0':
            self.message_box(title='Training Data Error', text='No created crops found!')
            return
        self.disconnect()
        self.open_annotation_tool()

    def train_data_button_clicked(self):
        """ Open training data creation menu """
        # Check if training and test set has been selected:
        if not self.trainset_id:
            self.message_box(title='Training Set Error', text='Select a training set first!')
            return
        # Find color channel
        self.get_color_channel()

        # Check if training set is selected for showing evaluation scores
        if not self.trainset_id:
            self.scores_df = pd.DataFrame(columns=['model'])
        else:
            if (self.eval_path / '{}.csv'.format(self.trainset)).is_file():
                self.scores_df = pd.read_csv(self.eval_path / '{}.csv'.format(self.trainset))
            else:
                self.scores_df = pd.DataFrame(columns=['model'])

        # Get trained models (and scores)
        self.prelabel_model = None
        self.prelabel_models = self.get_trained_models(scores_df=self.scores_df)

        # Fill QListWidget
        self.prelabel_model_list.clear()
        self.train_data_prelabel_model_edit.setText('')
        for prelabel_model in self.prelabel_models:
            self.prelabel_model_list.addItem(prelabel_model[0])
        # Preselect best model (only if evaluated)
        if self.prelabel_models:
            if self.prelabel_models[0][-1] > 0:
                self.prelabel_model_list.item(0).setSelected(True)
                self.prelabel_model = self.prelabel_models[0][1]
                self.train_data_prelabel_model_edit.setText("{}: {}".format(self.prelabel_models[0][1].parent.stem,
                                                                            self.prelabel_models[0][1].stem))

        if self.prelabel_model:
            model_row = self.scores_df.loc[self.scores_df['model'] == "{}: {}".format(
                self.prelabel_model.parent.stem,
                self.prelabel_model.stem)]
            self.prelabel_model_ths = [model_row['th_cell'].values[0], model_row['th_seed'].values[0]]

        if len(self.prelabel_models) > 0:
            self.train_data_prelabel_checkbox.setEnabled(True)
        else:
            self.train_data_prelabel_checkbox.setEnabled(False)
            self.train_data_prelabel_checkbox.setChecked(False)
            self.select_crops_overlay_checkbox.setChecked(False)
            self.select_crops_overlay_checkbox.setVisible(False)

        # Open menu
        self.train_data.show()

    def train_data_crop_button_clicked(self):
        """ Open crop selection menu """

        if self.train_data_prelabel_checkbox.isChecked() and not self.is_ready():
            self.message_box(title='GUI busy', text='Try again when current calculation is finished or without'
                                                    'pre-labeling.')
            return

        if self.train_data_prelabel_checkbox.isChecked() and not self.prelabel_model:
            self.message_box(title="Pre-label model selection error",
                             text="Please select a pre-label model or disable the pre-labeling!")
            return

        p_train, p_val, p_test = self.get_crop_train_split()
        if p_train + p_val + p_test == 0:
            self.message_box(title="Set selection error", text="Please select at least one of the following sets: "
                                                               "train/val/test!")
            return

        self.select_crops_img_left_checkbox.setChecked(False)
        self.select_crops_img_center_checkbox.setChecked(False)
        self.select_crops_img_right_checkbox.setChecked(False)

        if self.train_data_all_frames_checkbox.isChecked():
            p_max_frames = 1
            n_max_frames = 1e4
        else:
            p_max_frames = 0.4  # use maximum 40% of the frames of an experiment
            n_max_frames = 30  # use maximum 30 frames of an experiment

        # Check if files have been selected
        if not self.file_ids:
            self.message_box(title='File Error', text='Select some files!')
            return

        # Initialize list with files and frames to crop
        crop_list = []

        # Connect to OMERO server
        conn_status = self.connect()
        if not conn_status:
            return

        if not self.conn.canWrite(self.conn.getObject("Dataset", oid=self.trainset_id)):
            self.message_box(title="Permission Error", text="Seems like you do not own the selected training set "
                                                            "in your group and that you have no write access."
                                                            " Either change the group permissions to 'Read-Write' or "
                                                            "add a new group for which you have write access.")
            self.disconnect()
            return

        # Get crop size used in the selected training set
        for ann in self.conn.getObject("Dataset", self.trainset_id).listAnnotations():
            if ann.OMERO_TYPE == omero.model.MapAnnotationI:
                keys_values = ann.getValue()
                for key, value in keys_values:
                    if key == 'crop_size':
                        self.crop_size = int(value)
        if self.crop_size == 0:
            self.message_box(title='Key Value Error',
                             text='No crop size of the selected training dataset found in the metadata. '
                                  'Check in omero insight or the webviewer if the crop size is set. If not add '
                                  'a key value pair (crop_size, value) or create a new training set with microbeSEG.')
            self.disconnect()
            return

        # Get available info about used crops
        self.split_info = []
        for ann in self.conn.getObject("Dataset", self.trainset_id).listAnnotations(ns='split.info.namespace'):
            with open(self.train_path / 'split_info.json', 'wb') as outfile:
                for chunk in ann.getFileInChunks():
                    outfile.write(chunk)
            with open(self.train_path / 'split_info.json', 'r') as infile:
                self.split_info = json.load(infile)
        if not self.split_info:
            self.split_info = {'used': [],      # used/denied experiment-frames
                               'num_acc': 0,    # number of accepted crops (without imported)
                               'num_ext:': 0}   # needed for filenaming of imported (external) crops
        if 'num_ext' not in self.split_info:
            self.split_info['num_ext'] = 0

        # Get number of crops in dataset
        self.trainset_length = len(list(self.conn.getObject("Dataset", self.trainset_id).listChildren()))
        self.select_crops_counter_label.setText("{}".format(self.trainset_length))
        if 'num_acc' not in self.split_info:
            self.split_info['num_acc'] = self.trainset_length - self.split_info['num_ext']

        # Set main window to busy
        self.status_bar.showMessage('Busy')
        self.output_edit.append('\nGet data for crop creation ...')
        QApplication.processEvents()

        # Initialize progress bar
        self.stop_crop_creation = False
        crop_list_progressbar = QProgressDialog('Progress', None, 0, len(self.files)-1, self)
        crop_list_progressbar.setPalette(self.get_dark_palette())
        crop_list_progressbar.setMinimumWidth(400)
        crop_list_progressbar.setWindowTitle("Get information from OMERO server")
        crop_list_progressbar.setWindowModality(Qt.ApplicationModal)
        crop_list_progressbar.setWindowFlags(self.windowFlags() | Qt.CustomizeWindowHint)
        crop_list_progressbar.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)
        self.train_data.close()
        crop_list_progressbar.show()
        crop_list_progressbar.setValue(0)
        QShortcut(QKeySequence("Ctrl+C"), crop_list_progressbar, self.stop_crop_creation_process)

        # Go through selected files, get frames of interest and append to list
        for i, file_id in enumerate(self.file_ids):
            img = self.conn.getObject("Image", file_id)
            name, dataset = img.getName(), img.getParent()
            project = dataset.getParent().getName()
            size_t = img.getSizeT()

            # Update progress bar
            crop_list_progressbar.setValue(i)
            QApplication.processEvents()

            if self.stop_crop_creation:
                self.disconnect()
                crop_list_progressbar.close()
                self.output_edit.append('Crop creation stopped due to user interaction.')
                self.train_data.show()
                if self.is_ready():
                    self.status_bar.showMessage('Ready')
                return

            # Get frames to extract crops from (avoid too use too much frames from the same experiment
            if p_max_frames * size_t > n_max_frames:
                frames = list(range(0, size_t, int(np.ceil(size_t // n_max_frames))))
            else:
                frames = list(range(0, size_t, int(1 / p_max_frames)))

            # Skip z stacks and too small images (user should use appropriate crop size)
            if img.getSizeZ() > 1 or img.getSizeX() < 0.9 * self.crop_size or img.getSizeY() < 0.9 * self.crop_size:
                continue

            # Less color channels available than required --> skip
            if img.getSizeC() < 3:
                continue
            # if self.color_channel > 0 and img.getSizeC() == 1:
            #     continue

            # Go through selected frames and check if already been used
            for frame in frames:

                # Check if frame has been used already
                if [file_id, frame, self.color_channel] in self.split_info['used']:
                    continue

                # Append frame to crop list
                crop_list.append({'id': file_id,
                                  'name': name,
                                  'project': project,
                                  'dataset': dataset.getName(),
                                  'frame': frame,
                                  'channel': self.color_channel})
        if len(crop_list) == 0:
            self.output_edit.append('No supported files found. Check color channel or if z-stacks were selected.'
                                    'The selected files may also be too small for the selected crop size. It may also '
                                    'be the case that from each selected file/frame already crops were extracted.')

            if self.is_ready():
                self.status_bar.showMessage('Ready')

            return

        shuffle(crop_list)
        self.crop_list = crop_list
        crop_list_progressbar.close()

        self.output_edit.append('Crop creation ready.')
        QApplication.processEvents()

        # Worker for cropping
        conn_status = self.connect()
        if not conn_status:
            return
        self.crop_thread = QThread(parent=self)
        self.crop_worker = DataCropWorker(self.crop_list,
                                          self.crop_size,
                                          self.trainset_id,
                                          self.train_path,
                                          self.omero_username_edit.text(),
                                          self.omero_password_edit.text(),
                                          self.omero_host_edit.text(),
                                          self.omero_port_edit.text(),
                                          self.group_id,
                                          self.train_data_prelabel_checkbox.isChecked(),
                                          self.prelabel_model,
                                          self.get_device(),
                                          self.get_num_gpus(),
                                          self.prelabel_model_ths)
        self.crop_worker.moveToThread(self.crop_thread)
        self.crop_thread.started.connect(self.crop_creation_started)
        self.crop_thread.started.connect(self.select_crops.exec)
        self.crop_thread.started.connect(self.crop_worker.next_crop)  # First crop needs to be initialized
        self.crop_thread.started.connect(self.crop_worker.get_crop)
        self.crop_worker.text_output.connect(self.get_worker_information)
        self.crop_worker.crops.connect(self.show_crop)
        self.get_crop_signal.connect(self.crop_worker.get_crop)
        self.stop_cropping_signal.connect(self.crop_worker.crop_creation_finished)
        self.select_crops.crop_selection_closed_signal.connect(self.crop_worker.crop_creation_finished)
        self.select_crops.crop_selection_closed_signal.connect(self.train_data.show)
        self.select_crops.crop_selection_closed_signal.connect(self.disconnect)
        self.crop_worker.finished.connect(self.crop_creation_finished)
        self.crop_worker.finished.connect(self.crop_thread.quit)
        self.crop_worker.finished.connect(self.crop_worker.deleteLater)
        self.crop_thread.finished.connect(self.crop_thread.deleteLater)
        self.crop_thread.start()

    def train_data_prelabel_model_button_clicked(self):
        """ Open model selection menu """
        self.prelabel_model_selection.exec()
        for list_item in self.prelabel_model_list.selectedIndexes():
            self.prelabel_model = self.prelabel_models[list_item.row()][1]
            if f"{self.prelabel_model.parent.stem}: {self.prelabel_model.stem}" in self.scores_df.model.values:
                model_row = self.scores_df.loc[self.scores_df['model'] == "{}: {}".format(
                    self.prelabel_model.parent.stem,
                    self.prelabel_model.stem)]
                self.prelabel_model_ths = [model_row['th_cell'].values[0], model_row['th_seed'].values[0]]
            else:
                self.prelabel_model_ths = [0.10, 0.45]  # standard thresholds for distance method
            self.train_data_prelabel_model_edit.setText("{}: {}".format(
                self.prelabel_models[list_item.row()][1].parent.stem,
                self.prelabel_models[list_item.row()][1].stem))

    def select_crops_overlay_checkbox_clicked(self):
        """ Add / remove pre-label overlay"""

        if self.select_crops_overlay_checkbox.isChecked():
            for i in range(len(self.crops)):
                if i == 0:
                    self.select_crops_img_left.setPixmap(self.prelabel_pixmaps[i][1])
                elif i == 1:
                    self.select_crops_img_center.setPixmap(self.prelabel_pixmaps[i][1])
                elif i == 2:
                    self.select_crops_img_right.setPixmap(self.prelabel_pixmaps[i][1])
        else:
            for i in range(len(self.crops)):
                if i == 0:
                    self.select_crops_img_left.setPixmap(self.prelabel_pixmaps[i][0])
                elif i == 1:
                    self.select_crops_img_center.setPixmap(self.prelabel_pixmaps[i][0])
                elif i == 2:
                    self.select_crops_img_right.setPixmap(self.prelabel_pixmaps[i][0])
        QApplication.processEvents()  # Needed for upscaling only
        self.select_crops.adjustSize()

    def show_crop(self, crops):
        """ Show crop sent from crop worker """

        self.crops = crops
        self.prelabel_pixmaps = []

        max_width = crops[0]['img_show'].shape[1]

        for i in range(len(crops)):

            qimage = QImage(crops[i]['img_show'],
                            crops[i]['img_show'].shape[1],
                            crops[i]['img_show'].shape[0],
                            QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            # Scaling directly to available space produces for some image sizes artifacts for the roi overlay
            if (self.screen_height - self.crop_size) < 250:
                if self.screen_height - 768 < 250:
                    max_width = 512
                else:
                    max_width = 768
                pixmap = pixmap.scaledToHeight(max_width)
            if self.screen_width - len(crops) * crops[i]['img_show'].shape[1] < 50:
                if self.screen_width - len(crops) * 512 < 50:
                    max_width = 320
                else:
                    max_width = 512

            pixmap = pixmap.scaledToWidth(max_width)

            if self.train_data_prelabel_checkbox.isChecked():
                roi_show = crops[i]['roi_show']
                # Scaling directly to available space produces for some image sizes artifacts for the roi overlay
                if (self.screen_height - self.crop_size) < 250:
                    if self.screen_height - 768 < 250:
                        max_width = 512
                    else:
                        max_width = 768
                    roi_show = resize(roi_show, (max_width, max_width, 3), order=3, anti_aliasing=True,
                                      preserve_range=True).astype(roi_show.dtype)
                if self.screen_width - len(crops) * roi_show.shape[1] < 50:
                    if self.screen_width - len(crops) * 512 < 50:
                        max_width = 320
                    else:
                        max_width = 512
                    roi_show = resize(roi_show, (max_width, max_width, 3), order=3, anti_aliasing=True,
                                      preserve_range=True).astype(roi_show.dtype)

                qimage_polygon = QImage(roi_show, roi_show.shape[1], roi_show.shape[0], QImage.Format_RGB888)
                pixmap_polygon = QPixmap.fromImage(qimage_polygon)
                self.prelabel_pixmaps.append([pixmap, pixmap_polygon])

            if i == 0:  # Left/center image
                if self.train_data_prelabel_checkbox.isChecked() and self.select_crops_overlay_checkbox.isChecked():
                    self.select_crops_img_left.setPixmap(pixmap_polygon)
                else:
                    self.select_crops_img_left.setPixmap(pixmap)
            elif i == 1:  # Center/right image
                if self.train_data_prelabel_checkbox.isChecked() and self.select_crops_overlay_checkbox.isChecked():
                    self.select_crops_img_center.setPixmap(pixmap_polygon)
                else:
                    self.select_crops_img_center.setPixmap(pixmap)
            elif i == 2:  # right image
                if self.train_data_prelabel_checkbox.isChecked() and self.select_crops_overlay_checkbox.isChecked():
                    self.select_crops_img_right.setPixmap(pixmap_polygon)
                else:
                    self.select_crops_img_right.setPixmap(pixmap)

        if len(crops) == 3:
            self.select_crops_img_left.setVisible(True)
            self.select_crops_img_center.setVisible(True)
            self.select_crops_img_right.setVisible(True)
            self.select_crops_img_left_checkbox.setVisible(True)
            self.select_crops_img_center_checkbox.setVisible(True)
            self.select_crops_img_right_checkbox.setVisible(True)

        elif len(crops) == 2:
            self.select_crops_img_left.setVisible(True)
            self.select_crops_img_center.setVisible(True)
            self.select_crops_img_right.setVisible(False)
            self.select_crops_img_left_checkbox.setVisible(True)
            self.select_crops_img_center_checkbox.setVisible(True)
            self.select_crops_img_right_checkbox.setVisible(False)
        elif len(crops) == 1:
            self.select_crops_img_left.setVisible(True)
            self.select_crops_img_center.setVisible(False)
            self.select_crops_img_right.setVisible(False)
            self.select_crops_img_left_checkbox.setVisible(True)
            self.select_crops_img_center_checkbox.setVisible(False)
            self.select_crops_img_right_checkbox.setVisible(False)

        self.select_crops.setMinimumWidth(max_width * len(crops) + 50)
        self.select_crops.setMinimumHeight(max_width + 100)

        self.select_crops_filename_label.setText("{}: {}\n{} - Frame: {}".format(self.crops[0]['project'],
                                                                                 self.crops[0]['dataset'],
                                                                                 self.crops[0]['image'],
                                                                                 self.crops[0]['frame']))
        QApplication.processEvents()
        self.select_crops.adjustSize()

    def show_overlay_sc_pressed(self):
        """ Add/remove pre-label overlay """
        if self.train_data_prelabel_checkbox.isChecked():
            self.select_crops_overlay_checkbox.click()

    def train_data_export_button_clicked(self):
        """ Import annotated data to omero """

        # Avoid importing multiple dataset at once
        if not self.is_ready():
            self.message_box(title='GUI busy', text='Try again when current calculation is finished.')
            return

        # Check if training and test set has been selected:
        if not self.trainset_id:
            self.message_box(title='Training Set Error', text='Select a training set first')
            return

        self.set_up_train_val_test_dirs()

        # Show progress bar
        self.export_data_progress_bar.show()

        # Get worker thread for data import
        self.export_thread = QThread(parent=self)
        self.export_worker = DataExportWorker()
        self.export_worker.moveToThread(self.export_thread)
        self.export_thread.started.connect(self.data_export_started)
        self.export_thread.started.connect(partial(self.export_worker.export_data,
                                                   self.trainset_id,
                                                   self.train_path,
                                                   self.omero_username_edit.text(),
                                                   self.omero_password_edit.text(),
                                                   self.omero_host_edit.text(),
                                                   self.omero_port_edit.text(),
                                                   self.group_id))
        self.stop_import_export_sc.activated.connect(self.export_worker.stop_export_process)
        self.export_worker.text_output.connect(self.get_worker_information)
        self.export_worker.progress.connect(self.export_data_progress_bar.setValue)
        self.export_worker.finished.connect(self.data_export_finished)
        self.export_worker.finished.connect(self.export_thread.quit)
        self.export_worker.finished.connect(self.export_worker.deleteLater)
        self.export_thread.finished.connect(self.export_thread.deleteLater)
        self.export_thread.start()

    def train_data_import_button_clicked(self):
        """ Import annotated data to omero """

        if not self.is_ready():
            self.message_box(title='GUI busy', text='Try again when current calculation is finished.')
            return

        # Get crop size from metadata of the selected training set
        conn_status = self.connect()
        if not conn_status:
            return

        if not self.conn.canWrite(self.conn.getObject("Dataset", oid=self.trainset_id)):
            self.message_box(title="Permission Error", text="Seems like you do not own the selected training set "
                                                            "in your group and that you have no write access."
                                                            " Either change the group permissions to 'Read-Write' or "
                                                            "add a new group for which you have write access.")
            self.disconnect()
            return

        for ann in self.conn.getObject("Dataset", self.trainset_id).listAnnotations():
            if ann.OMERO_TYPE == omero.model.MapAnnotationI:
                keys_values = ann.getValue()
                for key, value in keys_values:
                    if key == 'crop_size':
                        self.crop_size = int(value)
        self.disconnect()

        # Get probability of a crop to belong to train/val/test set
        p_train, p_val, p_test = self.get_import_train_split()
        if p_train == 0 and p_val == 0 and p_test == 0:
            return  # No set chosen

        # Get path of annotated data set
        file_dialog = QFileDialog()
        file_dialog.setPalette(self.get_dark_palette())
        file_dialog.setWindowTitle("Select Directory")
        file_dialog.setFileMode(QFileDialog.DirectoryOnly)
        if file_dialog.exec() == QFileDialog.Accepted:
            img_dir_path = Path(file_dialog.selectedFiles()[0])
        else:
            return

        # Get image ids and check if images are available
        img_ids = sorted(img_dir_path.glob('img*'))
        if len(img_ids) == 0:
            return

        # Get worker thread for data import
        self.import_thread = QThread(parent=self)
        self.import_worker = DataImportWorker()
        self.import_worker.moveToThread(self.import_thread)
        self.import_thread.started.connect(self.data_import_started)
        self.import_thread.started.connect(partial(self.import_worker.import_data,
                                                   img_ids,
                                                   self.import_data_normalization_checkbox.isChecked(),
                                                   self.crop_size,
                                                   self.trainset_id,
                                                   self.train_path,
                                                   self.omero_username_edit.text(),
                                                   self.omero_password_edit.text(),
                                                   self.omero_host_edit.text(),
                                                   self.omero_port_edit.text(),
                                                   self.group_id,
                                                   p_train,
                                                   p_val,
                                                   p_test))
        self.import_worker.progress.connect(self.import_data_progress_bar.setValue)
        self.stop_import_export_sc.activated.connect(self.import_worker.stop_import_process)
        self.import_worker.finished.connect(self.data_import_finished)
        self.import_worker.finished.connect(self.import_thread.quit)
        self.import_worker.finished.connect(self.import_worker.deleteLater)
        self.import_thread.finished.connect(self.import_thread.deleteLater)
        self.import_worker.text_output.connect(self.get_worker_information)
        self.import_thread.start()

    def training_finished(self):
        """ Set training state to False after training """
        self.train_state = False
        if self.is_ready():
            self.status_bar.showMessage('Ready')

    def training_started(self):
        """ Set train state to True """
        self.train_state = True
        self.status_bar.showMessage('Busy')

    def train_settings_train_button_clicked(self):
        """ Start training process """

        # Avoid training during data set import ...
        if not self.is_ready():
            self.message_box(title='GUI busy', text='Try again when current calculation is finished.')
            return

        # Get paths
        trainset_path = self.set_up_train_val_test_dirs()
        model_path = self.model_path / trainset_path.stem
        model_path.mkdir(exist_ok=True)

        # Get segmentation method
        label_type = self.get_segmentation_method()

        # Reset progress bars
        self.train_export_data_progress_bar.setValue(0), self.train_export_data_progress_bar.show()
        self.train_create_labels_progress_bar.setValue(0), self.train_create_labels_progress_bar.show()
        self.train_training_progress_bar.setValue(0), self.train_training_progress_bar.show()

        # Get worker threads and workers for data export, label creation and training
        self.export_thread, self.export_worker = QThread(parent=self), DataExportWorker()
        self.label_thread, self.label_worker = QThread(parent=self), CreateLabelsWorker()
        self.train_thread, self.train_worker = QThread(parent=self), TrainWorker()
        self.export_worker.moveToThread(self.export_thread)
        self.label_worker.moveToThread(self.label_thread)
        self.train_worker.moveToThread(self.train_thread)

        # Connect data export signals
        self.export_thread.started.connect(self.data_export_started)
        self.export_thread.started.connect(partial(self.export_worker.export_data,
                                                   self.trainset_id,
                                                   self.train_path,
                                                   self.omero_username_edit.text(),
                                                   self.omero_password_edit.text(),
                                                   self.omero_host_edit.text(),
                                                   self.omero_port_edit.text(),
                                                   self.group_id))
        self.stop_training_sc.activated.connect(self.export_worker.stop_export_process)
        self.export_worker.text_output.connect(self.get_worker_information)
        self.export_worker.progress.connect(self.train_export_data_progress_bar.setValue)
        self.export_worker.finished.connect(self.data_export_finished)
        self.export_worker.finished.connect(self.export_thread.quit)
        self.export_worker.finished.connect(self.export_worker.deleteLater)
        self.export_thread.finished.connect(self.label_thread.start)
        self.export_thread.finished.connect(self.export_thread.deleteLater)

        # Connect label creation signals
        self.label_thread.started.connect(self.label_creation_started)
        self.label_thread.started.connect(partial(self.label_worker.create_labels, trainset_path, label_type))
        self.stop_training_sc.activated.connect(self.label_worker.stop_label_creation_process)
        self.label_worker.text_output.connect(self.get_worker_information)
        self.label_worker.progress.connect(self.train_create_labels_progress_bar.setValue)
        self.label_worker.finished.connect(self.label_creation_finished)
        self.label_worker.finished.connect(self.label_thread.quit)
        self.label_worker.finished.connect(self.label_worker.deleteLater)
        self.label_thread.finished.connect(self.train_thread.start)
        self.label_thread.finished.connect(self.label_thread.deleteLater)

        # Connect training signals
        self.train_thread.started.connect(self.train_progress_edit.show)
        self.train_thread.started.connect(self.training_started)
        self.train_thread.started.connect(partial(self.train_worker.start_training,
                                                  trainset_path,
                                                  model_path,
                                                  label_type,
                                                  int(self.train_settings_iterations_line_edit.text()),
                                                  self.get_optimizer(),
                                                  int(self.train_settings_batchsize_edit.text()),
                                                  self.get_device(),
                                                  self.get_num_gpus()))
        self.stop_training_sc.activated.connect(self.train_worker.stop_training_process)
        self.train_worker.text_output_main_gui.connect(self.get_worker_information)
        self.train_worker.text_output.connect(self.get_training_information)
        self.train_worker.progress.connect(self.train_training_progress_bar.setValue)
        self.train_worker.finished.connect(self.training_finished)
        self.train_worker.finished.connect(self.train_thread.quit)
        self.train_worker.finished.connect(self.train_worker.deleteLater)
        self.train_thread.finished.connect(self.train_thread.deleteLater)

        # Start the export thread (which starts the label creation thread when finished, which starts training thread)
        self.export_thread.start()

    @pyqtSlot()
    def select_crops_accept_button_clicked(self):
        """ Accept proposed crop """

        if not (self.select_crops_img_left_checkbox.checkState() > 0
                or self.select_crops_img_center_checkbox.checkState() > 0
                or self.select_crops_img_right_checkbox.checkState() > 0):
            # Emit signal to update gui and calculate next crops
            if self.create_crops:  # Worker may be finished and closed since no more crops are available
                self.get_crop_signal.emit()
            else:
                self.select_crops.close()
            return

        p_train, p_val, p_test = self.get_crop_train_split()

        for i, crop_dict in enumerate(self.crops):

            random_number = random()

            if p_test > 0 and p_val > 0 and p_train > 0:
                subset = self.split_assignment(self.split_info['num_acc'])
            else:
                if random_number < p_test:
                    subset = 'test'
                elif random_number < p_test + p_val:
                    subset = 'val'
                else:
                    subset = 'train'

            # Only proceed if corresponding checkbox is checked
            if i == 0 and not self.select_crops_img_left_checkbox.checkState() > 0:
                continue
            elif i == 1 and not self.select_crops_img_center_checkbox.checkState() > 0:
                continue
            elif i == 2 and not self.select_crops_img_right_checkbox.checkState() > 0:
                continue

            try:
                omero_img = self.conn.createImageFromNumpySeq(plane_gen_rgb(crop_dict['img']),
                                                              "img_{:03d}.tif".format(self.split_info['num_acc']),
                                                              1, 3, 1,
                                                              description='training image crop',
                                                              dataset=self.conn.getObject("Dataset", self.trainset_id))
            except Exception as e:  # probably timeout  --> reconnect and try again
                self.disconnect()
                conn_status = self.connect()
                if not conn_status:
                    self.select_crops.close()
                    return
                omero_img = self.conn.createImageFromNumpySeq(plane_gen_rgb(crop_dict['img']),
                                                              "img_{:03d}.tif".format(self.split_info['num_acc']),
                                                              1, 3, 1,
                                                              description='training image crop',
                                                              dataset=self.conn.getObject("Dataset", self.trainset_id))

            self.trainset_length += 1
            self.split_info['num_acc'] += 1
            self.split_info['used'].append((crop_dict['image_id'], crop_dict['frame'], crop_dict['channel']))

            # Upload metadata
            crop_dict['set'] = subset
            if self.train_data_prelabel_checkbox.isChecked():
                if i == 0 and self.select_crops_img_left_checkbox.checkState() == 2:
                    crop_dict['pre_labeled'] = "{}_{}".format(self.prelabel_model.parent.stem, self.prelabel_model.stem)
                elif i == 1 and self.select_crops_img_center_checkbox.checkState() == 2:
                    crop_dict['pre_labeled'] = "{}_{}".format(self.prelabel_model.parent.stem, self.prelabel_model.stem)
                elif i == 2 and self.select_crops_img_right_checkbox.checkState() == 2:
                    crop_dict['pre_labeled'] = "{}_{}".format(self.prelabel_model.parent.stem, self.prelabel_model.stem)
            key_value_data = self.get_crop_key_value_data(crop_dict)
            map_ann = MapAnnotationWrapper(self.conn)
            map_ann.setNs(metadata.NSCLIENTMAPANNOTATION)  # Use 'client' namespace to allow editing in Insight & web
            map_ann.setValue(key_value_data)
            map_ann.save()
            omero_img.linkAnnotation(map_ann)

            # Upload segmentation
            if self.train_data_prelabel_checkbox.isChecked():
                if i == 0 and not self.select_crops_img_left_checkbox.checkState() == 2:
                    continue
                elif i == 1 and not self.select_crops_img_center_checkbox.checkState() == 2:
                    continue
                elif i == 2 and not self.select_crops_img_right_checkbox.checkState() == 2:
                    continue

                update_service = self.conn.getUpdateService()
                omero_img = self.conn.getObject("Image", omero_img.getId())  # Reload needed
                mask_polygon = omero.model.PolygonI()
                mask_polygon.theZ, mask_polygon.theT, mask_polygon.fillColor = rint(0), rint(0), rint(0)  # ToDo: add to first channels? what does obiwan-microbi?
                mask_polygon.strokeColor = rint(int.from_bytes([255, 255, 0, 255], byteorder='big', signed=True))
                mask_polygon_list = []
                for roi in crop_dict['roi']:
                    mask_polygon.points = roi
                    mask_polygon_list.append(copy(mask_polygon))
                update_service.saveAndReturnObject(create_roi(omero_img, mask_polygon_list))

        # Write json file
        with open(self.train_path / 'split_info.json', 'w', encoding='utf-8') as outfile:
            json.dump(self.split_info, outfile, ensure_ascii=False, indent=2)
        namespace = 'split.info.namespace'
        file_ann = self.conn.createFileAnnfromLocalFile(str(self.train_path / 'split_info.json'),
                                                        mimetype='application/json',
                                                        ns=namespace,
                                                        desc='info about train/test split and already used frames')
        dataset = self.conn.getObject("Dataset", self.trainset_id)
        to_delete = []
        for ann in dataset.listAnnotations(ns=namespace):
            to_delete.append(ann.id)
        if to_delete:
            self.conn.deleteObjects('Annotation', to_delete, wait=True)
        self.conn.getObject("Dataset", self.trainset_id).linkAnnotation(file_ann)

        # Increase counter
        self.select_crops_counter_label.setText("{}".format(self.trainset_length))
        self.select_crops_counter_label.repaint()

        # Uncheck checkboxes for crop selection
        self.select_crops_img_left_checkbox.setChecked(False)
        self.select_crops_img_center_checkbox.setChecked(False)
        self.select_crops_img_right_checkbox.setChecked(False)

        # Emit signal to update gui and calculate next crops
        if self.create_crops:  # Worker may be finished and closed since no more crops are available
            self.get_crop_signal.emit()
        else:
            self.select_crops.close()

    def select_crops_img_left_clicked(self, event):
        """ Check checkbox related to the image"""
        self.select_crops_img_left_checkbox.click()
        
    def select_crops_img_center_clicked(self, event):
        """ Check checkbox related to the image"""
        self.select_crops_img_center_checkbox.click()
        
    def select_crops_img_right_clicked(self, event):
        """ Check checkbox related to the image"""
        self.select_crops_img_right_checkbox.click()
        
    def select_crops_partially_checkbox_clicked(self):
        """ Avoid that user changes state """
        self.select_crops_partially_checkbox.setCheckState(Qt.PartiallyChecked)

    def select_crops_checked_checkbox_clicked(self):
        """ Avoid that user changes state """
        self.select_crops_checked_checkbox.setChecked(True)
        
    def select_crops_unchecked_checkbox_clicked(self):
        """ Avoid that user changes state """
        self.select_crops_unchecked_checkbox.setChecked(False)

    def set_up_train_val_test_dirs(self):
        """ Clean existing / create train, val and test directories"""
        trainset_path = self.train_path / self.trainset
        if trainset_path.is_dir():
            rmtree(str(trainset_path))
        trainset_path.mkdir(exist_ok=True)
        (trainset_path / 'train').mkdir(exist_ok=True)
        (trainset_path / 'val').mkdir(exist_ok=True)
        (trainset_path / 'test').mkdir(exist_ok=True)
        return trainset_path

    @staticmethod
    def split_assignment(num_crops):
        """ The first crops should be assigned in a fixed way to it's subset (since some validation and test images
        are needed).

        :param num_crops:
        :return:
        """
        if num_crops < 8:
            assignment_list = ['train', 'train', 'val', 'test',
                               'train', 'train', 'val', 'test']  # after 8 crops: 4/2/2
            subset = assignment_list[num_crops]
        else:
            # modulo
            assignment_list = ['train', 'train', 'val',
                               'train', 'train', 'test']  # 14 crops: 8/3/3, 20: 12/4/4, 26: 16/5/5, 32: 20/6/6
            subset = assignment_list[(num_crops - 8) % 6]
        return subset

    def stop_crop_creation_process(self):
        """ Set flag for stopping crop creation process """
        self.stop_crop_creation = True


def run_gui(model_path, training_data_path, eval_results_path, inference_results_path, omero_settings, gpu, multi_gpu):
    """ Open graphical user interface """
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MicrobeSegMainWindow(model_path=model_path,
                                  train_path=training_data_path,
                                  eval_path=eval_results_path,
                                  results_path=inference_results_path,
                                  gpu=gpu,
                                  multi_gpu=multi_gpu,
                                  omero_settings=omero_settings)
    window.show()
    sys.exit(app.exec())
