HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)
HEADER = 64
FORMAT = 'utf-8'

from qgis.PyQt.QtCore import Qt

from qgis.core import (
    QgsApplication,
    QgsTask,
    Qgis,
)

from .core import core_plugin
from .dialogs import *
from .utils import *

import socket
import pickle

class InteractLearn(core_plugin):
    def __init__(self, iface):
        super().__init__(iface)
        self.param = None
        self.history_path = None
        self.dockwidget = None

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

    def runAnnotationDockWidget(self, history_path, img_pth, gt_path):
        if self.dockwidget == None:
            # Create the dockwidget (after translation) and keep reference
            self.dockwidget = annotationDockWidget(self.iface)
            # connect to provide cleanup on closing of dockwidget
            self.dockwidget.closingPlugin.connect(self.onClosePlugin)
            # show the dockwidget
            # TODO: fix to allow choice of dock location
            self.dockwidget.initSession(history_path, img_pth, gt_path)
            self.iface.addDockWidget(Qt.LeftDockWidgetArea, self.dockwidget)
            self.dockwidget.show()
        else:
            self.dockwidget.initSession(history_path, img_pth, gt_path)

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

            setLayerRGB(self.dlg.layerData, self.dlg.spinBox_R.value(), self.dlg.spinBox_G.value(), self.dlg.spinBox_B.value())

            setPaletteRenderer(self.dlg.layerLabel, self.dlg.paletted_renderer_widget.renderer())

            config, dataset_param = self.dlg.get_config()
            self.param = {'config' : config, 'dataset_param' : dataset_param}

            #self.runAnnotationDockWidget('/home/clement/code/TER/AL4EO/Results/ActiveLearning/Houston/gt1/breaking_tie/history_22_06_23_16_37.pkl', dataset_param['img_pth'], dataset_param['gt_pth']['train'][1])
            
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
        if exception is None:
            dataset_param = self.param['dataset_param']
            if result != None:
                self.runAnnotationDockWidget(result, dataset_param['img_pth'], dataset_param['gt_pth']['train'][1])
            else:
                self.iface.messageBar().pushMessage("Can't run annotation because Query don't finish", level=Qgis.Warning)
        else:
            self.iface.messageBar().pushMessage(str(exception), level=Qgis.Warning)

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
            

            