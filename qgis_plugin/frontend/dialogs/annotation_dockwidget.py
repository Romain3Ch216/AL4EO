# -*- coding: utf-8 -*-
"""
/***************************************************************************
 lolDockWidget
                                 A QGIS plugin
 lol
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                             -------------------
        begin                : 2022-06-15
        git sha              : $Format:%H$
        copyright            : (C) 2022 by lol
        email                : olol
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

import pickle
import os
from qgis.PyQt import QtWidgets, uic
from qgis.PyQt.QtCore import pyqtSignal
from qgis.core import QgsPalettedRasterRenderer
from ..utils import createHistoryLayer
from ..mapTool import MapTool

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'ui/annotation_dockwidget.ui'))

class annotationDockWidget(QtWidgets.QDockWidget, FORM_CLASS):

    closingPlugin = pyqtSignal()

    def __init__(self, iface, parent=None):
        """Constructor."""
        super(annotationDockWidget, self).__init__(parent)
        # Set up the user interface from Designer.
        # After setupUI you can access any designer object by doing
        # self.<objectname>, and you can use autoconnect slots - see
        # http://doc.qt.io/qt-5/designer-using-a-ui-file.html
        # #widgets-and-dialogs-with-auto-connect
        self.iface = iface
        self.history = None
        self.config = None
        self.annot_layer = None
        self.setupUi(self)


    #init dockWidget requirement 
    def initSession(self, history_path, vector_layer, raster_path):
        #load history
        with open(history_path, 'rb') as f:
            self.history, _, self.config = pickle.load(f)

        #create history layer 
        createHistoryLayer(os.path.basename(history_path)[:-4], self.history['coordinates'], raster_path)
