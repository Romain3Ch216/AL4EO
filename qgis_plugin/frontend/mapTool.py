from qgis.gui import QgsMapToolEmitPoint
import rasterio as rio

class MapTool(QgsMapToolEmitPoint):
    def __init__(self, iface, layer, classeSelected):
        QgsMapToolEmitPoint.__init__(self, iface.mapCanvas())

        self.iface = iface
        self.layer = layer
        self.classeSelected = classeSelected
        with rio.open(layer.dataProvider().dataSourceUri(), 'r') as src:
            self.layerArray = src.read(1)
            self.profile = src.profile.copy()
        self.canvasClicked.connect(self.onClick)

    def onClick(self, point, button):
        
        with rio.open(self.layer.dataProvider().dataSourceUri(), 'w', **self.profile) as src:
            row, col = rio.transform.rowcol(src.transform, point.x(), point.y())
            if 0 < row and row < src.height and 0 < col and col < src.width: 
                self.layerArray[row, col] = self.classeSelected
                src.write(self.layerArray, 1)

        self.layer.setAutoRefreshEnabled(False)
        self.iface.mapCanvas().refreshAllLayers()


    
