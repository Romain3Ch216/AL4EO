# -*- coding: utf-8 -*-

import os
from .layers_dialog import LayersDialog
from qgis.PyQt import QtWidgets, uic
from qgis.core import QgsProject
from qgis.PyQt.QtWidgets import QDialogButtonBox, QFileDialog

# This loads your .ui file so that PyQt can populate your plugin with the elements from Qt Designer
FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'ui/annotation_selection.ui'))

class AnnotSelectDialog(QtWidgets.QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        """Constructor."""
        super(AnnotSelectDialog, self).__init__(parent)
        self.setModal(True)
        self.layerLabel = None
        self.history_pth = None
        self.setupUi(self)
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)
        self.toolButton_selectLabel.clicked.connect(self.selectLabel)
        self.toolButton_selectHistory.clicked.connect(self.selectHistory)

    #Select label layer from LayersDialog
    def selectLabel(self):
        sub_dlg = LayersDialog(1)
        sub_dlg.show()
        result = sub_dlg.exec_()

        if result:
            item = sub_dlg.listWidget.currentItem()

            if item:
                self.label_label_name.setText(item.text())
                self.layerLabel = QgsProject.instance().mapLayersByName(item.text())[0]

                if self.history_pth != None:
                    self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(True)

    #Select history path from QFileDialog
    def selectHistory(self):
        self.history_pth, check = QFileDialog.getOpenFileName(None, "", "Pickle Files (*.pkl)")

        if check:
            self.label_history_name.setText(os.path.basename(self.history_pth))

            if self.layerLabel != None:
                self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(True)
        