import os
from qgis.PyQt import QtWidgets, uic
from ..utils import get_layers

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'ui/layers.ui'))

class LayersDialog(QtWidgets.QDialog, FORM_CLASS):
    """Window to select layers based on the current active layers in Qgis (get_layers)"""
    def __init__(self, type):
        super(LayersDialog, self).__init__()
        self.setModal(True)
        self.setupUi(self)
        if type == 1:
            self.setWindowTitle("Single band layers")
        if type == 2:
            self.setWindowTitle("Multi bands layers")
        _, layer_list = get_layers(type)
        self.listWidget.addItems(layer_list)