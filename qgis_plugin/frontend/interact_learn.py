HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)
HEADER = 64
FORMAT = 'utf-8'

from qgis.PyQt.QtCore import Qt

from qgis.core import (
    QgsPalettedRasterRenderer,
    QgsApplication,
    QgsTask,
    Qgis,
)

from .core import core_plugin
from .dialogs import *
from .utils import *

from processing.gui.RectangleMapTool import RectangleMapTool

import socket
import pickle

class InteractLearn(core_plugin):
    def __init__(self, iface):
        super().__init__(iface)
        self.param = None
        self.layerLabel = None
        self.history_path = None
        self.dockwidget = None
        self.rectangle = None

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        self.add_action(
            ':/icons/letter-q.png',
            text=self.tr(u'Query'),
            callback=self.runIlearnDialog,
            parent=self.iface.mainWindow()
            )
        self.add_action(
            ':/icons/letter-a.png',
            text=self.tr(u'Annotation'),
            callback=self.runAnnotationDialog,
            parent=self.iface.mainWindow()
            )
        self.add_action(
            ':/icons/S-icon.png',
            text=self.tr(u'Subset'),
            callback=self.selectSubset,
            parent=self.iface.mainWindow()
            )

        # will be set False in run()
        self.first_start = True

    #run Annotation dockWidget for annot pixels
    def runAnnotationDockWidget(self, history_path, annot_layer):
        # Create the dockwidget (after translation) and keep reference
        self.dockwidget = annotationDockWidget(self.iface)
        # connect to provide cleanup on closing of dockwidget
        self.dockwidget.closingPlugin.connect(self.onClosePlugin)
        # show the dockwidget
        # TODO: fix to allow choice of dock location
        self.dockwidget.initSession(history_path, annot_layer)
        self.iface.addDockWidget(Qt.LeftDockWidgetArea, self.dockwidget)
        self.dockwidget.show()

    #Run annotationDialog to select history path and label layer
    def runAnnotationDialog(self):
        dlg = AnnotSelectDialog()
        dlg.show()
        result = dlg.exec_()

        if result:

            self.runAnnotationDockWidget(dlg.history_pth, dlg.layerLabel)


    def selectSubset(self):
        if self.rectangle is not None:
            self.rectangle.reset()
        else:
            self.rectangle = RectangleMapTool(self.iface.mapCanvas())
        self.iface.mapCanvas().setMapTool(self.rectangle)

    
    #run interactive learning dialog for selecting data layer, label layer and query config
    def runIlearnDialog(self):
        # Create the dialog with elements (after translation) and keep reference
        # Only create GUI ONCE in callback, so that it will only load when the plugin is started
        if self.first_start == True:
            self.first_start = False
            self.dlg = InteractLearnDialog(self.iface)

        # show the dialog
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result:

            ymax = self.dlg.layerLabel.extent().yMaximum()
            xmin = self.dlg.layerLabel.extent().xMinimum()
            xsize = self.dlg.layerLabel.rasterUnitsPerPixelX()
            ysize = self.dlg.layerLabel.rasterUnitsPerPixelY()

            if self.rectangle is None:
                bounding_box = None 
            else:
                # start_point = int(((self.rectangle.startPoint.x() - xmin) / xsize)), int(((ymax - self.rectangle.startPoint.y() / ysize)))
                # end_point = int(((self.rectangle.endPoint.x() - xmin) / xsize)), int(((ymax - self.rectangle.endPoint.y() / ysize)))

                start_point = int(((self.rectangle.startPoint.x() - xmin) / xsize)), int(((ymax - self.rectangle.startPoint.y() / ysize)))
                end_point = int(((self.rectangle.endPoint.x() - xmin) / xsize)), int(((ymax - self.rectangle.endPoint.y() / ysize)))

                bounding_box = (start_point, end_point)

            #get config and dataset parameters from dialog
            config, dataset_param = self.dlg.get_config()
            config['bounding_box'] = bounding_box
            self.param = {'name': 'query', 'config' : config, 'dataset_param' : dataset_param}

            #set data layer RGB bands from dialog values
            setLayerRGB(self.dlg.layerData, self.dlg.spinBox_R.value(), self.dlg.spinBox_G.value(), self.dlg.spinBox_B.value())

            #change opacity of layer label 
            self.layerLabel = self.dlg.layerLabel
            classes = self.layerLabel.renderer().classes()
            classes[0].color.setAlpha(0)
            renderer = QgsPalettedRasterRenderer(self.layerLabel.dataProvider(), 1, classes)
            renderer.setOpacity(0.7)

            #reproject label layer to data layer size
            formatAnnotationRaster(dataset_param['img_pth'], dataset_param['gt_pth'])
            name = self.layerLabel.name()
            QgsProject.instance().removeMapLayer(self.layerLabel.id())
            self.layerLabel = self.iface.addRasterLayer(dataset_param['gt_pth'], name)
            self.layerLabel.setRenderer(renderer) 
            self.layerLabel.triggerRepaint()
            
            #communicate query config and params to serveur in QgsTask thread
            task = QgsTask.fromFunction(
                "Ilearn Query",
                self.send_and_recv_Serveur,
                on_finished=self._completed,
            )
            QgsApplication.taskManager().addTask(task)
            self.task = (
                task
            )

    #run when QgsTask is complete, run annotationDockWidget if query has reached the end
    def _completed(self, exception, result=None):
        if exception is None:
            if result != None:
                self.runAnnotationDockWidget(result, self.layerLabel)
            else:
                self.iface.messageBar().pushMessage("Can't run annotation because Query don't finish", level=Qgis.Warning)
        else:
            self.iface.messageBar().pushMessage(str(exception), level=Qgis.Warning)

    #QgsTask for sending config and param to serveur and receive history path   
    def send_and_recv_Serveur(self, task):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((HOST, PORT))
            except Exception as e:
                raise e

            param_pkl = pickle.dumps(self.param)

            size = len(param_pkl)
            send_size = str(size).encode(FORMAT) 
            send_size += b' ' * (HEADER - len(send_size))
            s.send(send_size)
            s.send(param_pkl)

            recv_size = s.recv(HEADER).decode(FORMAT)
            history_path_pkl = None
            if recv_size:
                history_path_pkl = s.recv(int(recv_size))

            s.close()

        if history_path_pkl:
            return pickle.loads(history_path_pkl)
