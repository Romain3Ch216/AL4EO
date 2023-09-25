from qgis.core import (
    QgsProject,
    QgsRasterLayer,
    QgsVectorLayer, 
    QgsFeature,
    QgsGeometry, 
    QgsPointXY, 
    QgsMapLayer,
    QgsMarkerSymbol,
    QgsCoordinateReferenceSystem,
)

from osgeo import gdal 
import numpy as np  

class WarnQgs:
    """Warning class used to pass Exceptions to Qgis without interrupting the daemon"""

    def __init__(self, msg, iface=None):
        self.msg = msg
        self.iface = iface

    def __str__(self):
        return "\033[91m" + self.msg + "\033[0m"

    def __call__(self, level):
        """ Print self.msg on qgis interface.
        ----------
        Parameters:
            level: int Qgis level, eg Qgis.Critical, Qgis.Info, Qgis.Warning, ..."""
        if self.iface:
            self.iface.messageBar().pushMessage(str(self.msg), level=level)
        else:
            raise TypeError("QgsWarn is not callable until iface has not been set")

def get_layers(type):
    """
    type: 0 = all, 1 = single band, 2 = Multiband
    Get the different layers currently opened in Qgis. Rename them based on their occurence to manage the case where
    different layers have the same name in Qgis.
    """
    layers = []
    for tree_layer in QgsProject.instance().layerTreeRoot().findLayers():
        layer = tree_layer.layer()
        if layer.type() == QgsMapLayer.RasterLayer or layer.type() == QgsMapLayer.VectorLayer:
            # if (type == 0 or (type == 1 and tree_layer.layer().rasterType() in [0,1]) 
            # or (type == 2 and tree_layer.layer().rasterType() == 2)):
            layers.append(layer)

    layer_list = [i.name() for i in layers]
    count = {i: layer_list.count(i) for i in layer_list if layer_list.count(i) > 1}
    for i, layer in enumerate(layer_list):
        if layer in count.keys() and isinstance(layers[i], QgsRasterLayer):
                bands, height, width = (
                    layers[i].bandCount(),
                    layers[i].height(),
                    layers[i].width(),
                )
                layer_list[
                    i
                ] += f" ({count[layer] - 1}); {bands} bands; H*W: {height}*{width}"
                count[layer] -= 1
    return layers, layer_list

def setLayerRGB(layer, R, G, B):
    # set RGB for multiband layer
    renderer = layer.renderer()
    renderer.setRedBand(R)
    renderer.setGreenBand(G)
    renderer.setBlueBand(B)
    layer.setDefaultContrastEnhancement()
    layer.triggerRepaint()


def createHistoryLayer(name, coordinates, gt_pth):
    #Create a Vector layer with points which are at the coordinates in the history
    print(gt_path)
    src = gdal.Open(gt_pth)
    crs = QgsCoordinateReferenceSystem()
    crs.createFromWkt(src.GetProjection())
    transform = src.GetGeoTransform()
    
    vl = QgsVectorLayer("MultiPoint", name, "memory", crs=crs)
    pr = vl.dataProvider()
    # Enter editing mode
    vl.startEditing()
  
    for point in coordinates:
        point = point.astype(np.float64)
        pointXY = gdal.ApplyGeoTransform(transform, point[1]+0.5, point[0]+0.5)
        fet = QgsFeature()
        fet.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(pointXY[0], pointXY[1])))
        pr.addFeatures([ fet ])

    # Commit changes
    vl.commitChanges()

    symbol = QgsMarkerSymbol.createSimple({'name': 'circle', 'color': 'red'})
    vl.renderer().setSymbol(symbol)
        
    # Show in project
    root = QgsProject.instance().layerTreeRoot()
    QgsProject.instance().addMapLayer(vl, False)
    root.insertLayer(0, vl)
    vl.triggerRepaint()


    