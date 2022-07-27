from qgis.gui import QgsMapToolEmitPoint
from qgis.core import QgsRasterBlock

class MapTool(QgsMapToolEmitPoint):
    def __init__(self, mapCanvas, layer, classeSelected):
        QgsMapToolEmitPoint.__init__(self, mapCanvas)
        #Get all layer info for annotate pixels
        self.layer = layer
        self.provider = self.layer.dataProvider()
        self.block = QgsRasterBlock(self.provider.dataType(1), 1, 1)
        self.width = self.layer.width()
        self.height = self.layer.height()
        self.xsize = self.layer.rasterUnitsPerPixelX()
        self.ysize = self.layer.rasterUnitsPerPixelY()
        extent = self.layer.extent()
        self.ymax = extent.yMaximum()
        self.xmin = extent.xMinimum()
        self.classeSelected = classeSelected
        self.canvasClicked.connect(self.onClick)

    def onClick(self, point, button):

        #row in pixel coordinates
        row = int(((self.ymax - point.y()) / self.ysize))

        #row in pixel coordinates
        column = int(((point.x() - self.xmin) / self.xsize))

        if row <= 0 or column <=0 or row > self.height or column > self.width:
            #if pixel not in layer
            row = "out of extent"
            column = "out of extent"
        else:
            #if pixel in layer, change pixel value
            self.block.setData(self.classeSelected.to_bytes(2, 'little'), 0)

            self.provider.setEditable(True)
            self.provider.writeBlock(self.block, 1, column, row)
            self.provider.setEditable(False)

            self.layer.triggerRepaint()


    
