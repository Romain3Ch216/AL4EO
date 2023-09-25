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

from osgeo import gdal
import socket
import pickle
import subprocess


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
            ':/icons/S-icon.png',
            text=self.tr(u'Subset'),
            callback=self.selectSubset,
            parent=self.iface.mainWindow()
            )

        # will be set False in run()
        self.first_start = True

    def runAnnotationDockWidget(self, history_path, vector_layer, raster_path):
        # Create the dockwidget (after translation) and keep reference
        self.dockwidget = annotationDockWidget(self.iface)
        # connect to provide cleanup on closing of dockwidget
        self.dockwidget.closingPlugin.connect(self.onClosePlugin)
        # show the dockwidget
        # TODO: fix to allow choice of dock location
        self.dockwidget.initSession(history_path, vector_layer, raster_path)
        self.dockwidget.show()

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

            gt_path = self.dlg.layerLabel.dataProvider().dataSourceUri()
            out_path = gt_path[:-3] + 'tif'
            width = self.dlg.layerData.extent().xMaximum() - self.dlg.layerData.extent().xMinimum()
            height = self.dlg.layerData.extent().yMaximum() - self.dlg.layerData.extent().yMinimum()
            extent = self.dlg.layerData.extent()
            extent = "%.17f %.17f %.17f %.17f" % (extent.xMinimum(), extent.yMinimum(), extent.xMaximum(), extent.yMaximum())
            print(extent)
            query = f"gdal_rasterize -a Material -ts {width} {height} -init 0.0 -te {extent} -ot UInt16 -of GTiff {gt_path} {out_path}" 
            subprocess.call(query, shell=True)
            self.dlg.gt_raster_path = out_path
            gt_raster = gdal.Open(out_path, gdal.GA_ReadOnly)
            label_values = np.unique(gt_raster.ReadAsArray())

            geoTransform = gt_raster.GetGeoTransform()
            xmin = geoTransform[0]
            ymax = geoTransform[3]
            xsize = gt_raster.RasterXSize
            ysize = gt_raster.RasterYSize
            xmax = xmin + geoTransform[1] * xsize
            ymin = ymax + geoTransform[5] * ysize

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
            dataset_param['label_values'] = label_values
            self.param = {'name': 'query', 'config' : config, 'dataset_param' : dataset_param}

            #set data layer RGB bands from dialog values
            # setLayerRGB(self.dlg.layerData, self.dlg.spinBox_R.value(), self.dlg.spinBox_G.value(), self.dlg.spinBox_B.value())

            #change opacity of layer label 
            # self.layerLabel = self.dlg.layerLabel
            
            # name = self.layerLabel.name()
            # QgsProject.instance().removeMapLayer(self.layerLabel.id())
            # self.layerLabel = self.iface.addRasterLayer(dataset_param['gt_pth'], name)
            # self.layerLabel.setRenderer(renderer) 
            # self.layerLabel.triggerRepaint()
            
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
            

    def _completed(self, exception, result=None):
        print("Completed")
        if exception is None:
            print("Exception is None")
            if result != None:
                print("Result is not None")
                self.runAnnotationDockWidget(result, self.dlg.layerLabel, self.dlg.gt_raster_path)
            else:
                print("Result is None")
                self.iface.messageBar().pushMessage("Can't run annotation because Query don't finish", level=Qgis.Warning)
        else:
            print("Exception")
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
