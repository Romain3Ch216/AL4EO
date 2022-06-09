from numpy import double
from qgis.core import (
    QgsRasterBandStats,
    QgsContrastEnhancement,
    QgsLinearMinMaxEnhancement,
)

HOST = 'localhost'
PORT = 65432

import numpy as np

from .core import core_plugin
from .dialogs import *
from .utils import WarnQgs

import socket
import pickle

class InteractLearn(core_plugin):
    def __init__(self, iface):
        super().__init__(iface)
        self.iface = iface

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = ':/icons/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'Dataset_config'),
            callback=self.runIlearnDialog,
            parent=self.iface.mainWindow())

        # will be set False in run()
        self.first_start = True

    def runIlearnDialog(self):
        # Create the dialog with elements (after translation) and keep reference
        # Only create GUI ONCE in callback, so that it will only load when the plugin is started
        if self.first_start == True:
            self.first_start = False
            self.dlg = InteractLearnDialog()

        # show the dialog
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result:
            # set rgb for data layer
            layer = self.dlg.layerData
            renderer = layer.renderer()
            renderer.setRedBand(self.dlg.spinBox_R.value())
            renderer.setGreenBand(self.dlg.spinBox_G.value())
            renderer.setBlueBand(self.dlg.spinBox_B.value())
            layer.setDefaultContrastEnhancement()
            self.dlg.layerData.triggerRepaint()

            #set palette for label layer
            layer = self.dlg.layerLabel
            layer.setRenderer(self.dlg.paletted_renderer_widget.renderer())
            layer.setOpacity(0.6)
            layer.triggerRepaint()

            config, dataset_param = self.dlg.get_config()
            param = {'config' : config, 'dataset_param' : dataset_param}
            
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((HOST, PORT))
                param_pkl = pickle.dumps(param)
                s.send(param_pkl)
                s.close()

            