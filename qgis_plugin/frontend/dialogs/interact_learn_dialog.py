# -*- coding: utf-8 -*-

import os

from qgis.PyQt import uic
from qgis.PyQt import QtWidgets
from .layers_dialog import LayersDialog
from qgis.core import (
    QgsProject,
)
from qgis.gui import QgsPalettedRendererWidget
from PyQt5.QtCore import (
    QTimer,
)
from PyQt5.QtWidgets import (
    QDialogButtonBox,
)
from PyQt5.QtGui import (
    QColor,
)

# This loads your .ui file so that PyQt can populate your plugin with the elements from Qt Designer
FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'ui/interact_learn.ui'))

class InteractLearnDialog(QtWidgets.QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        """Constructor."""
        super(InteractLearnDialog, self).__init__(parent)
        self.setModal(True)
        self.layerData = None
        self.layerLabel = None
        self.paletted_renderer_widget = None
        # Set up the user interface from Designer through FORM_CLASS.
        # After self.setupUi() you can access any designer object by doing
        # self.<objectname>, and you can use autoconnect slots - see
        # http://qt-project.org/doc/qt-4.8/designer-using-a-ui-file.html
        # #widgets-and-dialogs-with-auto-connect
        self.setupUi(self)
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)
        self.toolButton_selectData.clicked.connect(self.selectData)
        self.toolButton_selectLabel.clicked.connect(self.selectLabel)

    def selectData(self):
        sub_dlg = LayersDialog(2)
        sub_dlg.show()
        result = sub_dlg.exec_()
        if result:
            item = sub_dlg.listWidget.currentItem()
            if item:
                self.label_data_name.setText(item.text())
                self.layerData = QgsProject.instance().mapLayersByName(item.text())[0]
                band_count = self.layerData.bandCount()
                used_bands = self.layerData.renderer().usesBands()
                self.spinBox_R.setValue(used_bands[0])
                self.spinBox_R.setMaximum(band_count)
                self.spinBox_G.setValue(used_bands[1])
                self.spinBox_G.setMaximum(band_count)
                self.spinBox_B.setValue(used_bands[2])
                self.spinBox_B.setMaximum(band_count)
                self.layerData.nameChanged.connect(self.dataNameChanged)
                self.layerData.willBeDeleted.connect(self.dataLayerRemoved)
                if self.layerLabel != None:
                    self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(True)

    def dataNameChanged(self):
        self.label_data_name.setText(self.layerData.name())

    def dataLayerRemoved(self):
        self.layerData = None
        self.label_data_name.setText('none')
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)

    def selectLabel(self):
        sub_dlg = LayersDialog(1)
        sub_dlg.show()
        result = sub_dlg.exec_()
        if result:
            item = sub_dlg.listWidget.currentItem()
            if item:
                self.label_label_name.setText(item.text())
                self.layerLabel = QgsProject.instance().mapLayersByName(item.text())[0]
                self.layerLabel.nameChanged.connect(self.labelNameChanged)
                self.layerLabel.willBeDeleted.connect(self.labelLayerRemoved)
                if not self.paletted_renderer_widget:
                    self.paletted_renderer_widget = QgsPalettedRendererWidget(self.layerLabel)
                    self.paletted_renderer_widget.classify()
                    self.verticalLayout_Wclasse.addWidget(self.paletted_renderer_widget)
                else:
                    self.paletted_renderer_widget.classify()
                if self.layerData != None:
                    self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(True)

    def labelLayerRemoved(self):
        self.layerLabel = None
        self.label_label_name.setText('none')
        for i in reversed(range(self.verticalLayout_Wclasse.count())): 
            self.verticalLayout_Wclasse.takeAt(i).widget().deleteLater()
        self.paletted_renderer_widget = None
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)
        QTimer.singleShot(0, self.adjustSize)

    def labelNameChanged(self):
        self.label_label_name.setText(self.layerLabel.name())


    def get_config(self):
        assert(self.layerData != None and self.layerLabel != None), "cannot get dataset"

        dataset_param = {}
        dataset_param['img_pth'] = self.layerData.dataProvider().dataSourceUri()
        dataset_param['gt_pth'] = {'train' : {1 : self.layerLabel.dataProvider().dataSourceUri()}}

        label_values, palette = zip(*self.layerLabel.renderer().legendSymbologyItems())
        label_values, palette = list(label_values), list(map(QColor.getRgb, palette))
        palette = [(p[0],p[1],p[2]) for p in palette]

        dataset_param['palette'] = palette
        dataset_param['label_values'] = label_values

        dataset_param['ignored_labels'] = [0]
        dataset_param['n_bands'] = self.layerData.bandCount()
        dataset_param['img_shape'] = self.layerData.width(), self.layerData.height()
        dataset_param['rgb_bands'] = (self.spinBox_R.value(), self.spinBox_G.value(), self.spinBox_B.value())

        config = {}
        config['query'] = self.comboBox_query.currentText()
        config['n_px'] = self.spinBox_n_px.value()
        config['epochs'] = self.spinBox_epochs.value()
        config['batch_size'] = self.spinBox_batch_size.value()
        config['lr'] = self.doubleSpinBox_lr.value()
        config['weight_decay'] = self.doubleSpinBox_weight_decay.value()
        config['device'] = self.comboBox_device.currentText().lower()

        config['dataset'] = "Houston"
        config['run'] = [1]
        config['remove'] = []

        return config, dataset_param        
