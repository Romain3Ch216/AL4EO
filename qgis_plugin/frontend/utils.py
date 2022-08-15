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
import rasterio as rio
from rasterio.warp import reproject, Resampling

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
        if layer.type() == QgsMapLayer.RasterLayer:
            if (type == 0 or (type == 1 and tree_layer.layer().rasterType() in [0,1]) 
            or (type == 2 and tree_layer.layer().rasterType() == 2)):
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

def formatAnnotationRaster(img_pth, gt_pth):
    #reproject gt_pth to img pth shape
    with rio.open(img_pth) as src:
        #get data image transform crs shape metadata
        dst_transform = src.transform
        dst_crs = src.crs
        dst_shape = src.height, src.width
    
    with rio.open(gt_pth) as src:
        #update metadata of label image with image metadata
        kwargs = src.profile.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': dst_transform,
            'width': dst_shape[1],
            'height': dst_shape[0]
        })
        #get label header tags 
        dst_tags = src.tags(ns=src.driver)

        with rio.open(gt_pth, 'w', **kwargs) as dst:
            #reproject label image and write it 
            reproject(
                source=rio.band(src, 1),
                destination=rio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)
            #write classes info header tags
            dst.update_tags(ns=dst.driver, classes=dst_tags['classes'])
            dst.update_tags(ns=dst.driver, class_lookup=dst_tags['class_lookup'])
            dst.update_tags(ns=dst.driver, class_name=dst_tags['class_names'])
    correctTagNameIssue(gt_pth)

def getClasseNameColor(gt_path):
    #get classes names and colors from header file
    with rio.open(gt_path) as src:
        src_tags = src.tags(ns=src.driver)
        class_names = src_tags['class_names'].replace(' ', '').replace('{', '').replace('}', '').split(',')
        class_lookup = src_tags['class_lookup'].replace(' ', '').replace('{', '').replace('}', '').split(',')

    class_color = [(int(class_lookup[i]), int(class_lookup[i+1]), int(class_lookup[i+2])) for i in range(0, len(class_lookup), 3)]

    return class_names, class_color

def updateClassNameColor(class_names, class_color, gt_pth):
    #update header file classes names and colors 
    class_names_string = '{'
    for i in range(len(class_names)-1):
        class_names_string += class_names[i] + ', '
    class_names_string += class_names[len(class_names)-1] + '}'
    class_lookup_string = str(class_color).replace('[', '{').replace(']', '}').replace('(','').replace(')','')

    with rio.open(gt_pth) as src:
        kwargs = src.profile.copy()
        with rio.open(gt_pth, 'w', **kwargs) as dst:
            dst.write(src.read(1),1)
            dst.update_tags(ns=dst.driver, classes=len(class_names))
            dst.update_tags(ns=dst.driver, class_lookup=class_lookup_string)
            dst.update_tags(ns=dst.driver, class_name=class_names_string)
    correctTagNameIssue(gt_pth)

def correctTagNameIssue(gt_pth):
    """Change the tag 'class name' to 'class names' 
    because rasterio write 'class name', but QGIS reconize only 'class names' tag"""
    gt_hdr_pth = gt_pth[:-4] + "hdr"
    file_content = ""
    with open(gt_hdr_pth, "r") as f:
        file_content = f.read()
    file_content = file_content.replace("class name =","class names =")
    with open(gt_hdr_pth, "w") as f:
        f.write(file_content)

def createHistoryLayer(name, coordinates, gt_pth):
    #Create a Vector layer with points which are at the coordinates in the history
    with rio.open(gt_pth, 'r') as src:
        transform = src.transform
        crs = QgsCoordinateReferenceSystem()
        crs.createFromString(src.crs.to_string())
    
    vl = QgsVectorLayer("MultiPoint", name, "memory", crs=crs)
    pr = vl.dataProvider()
    # Enter editing mode
    vl.startEditing()

    # add a feature
    # To just create the layer and add features later, delete the four lines from here until Commit changes
    for point in coordinates:
        pointXY = rio.transform.xy(transform, point[0], point[1])
        fet = QgsFeature()
        fet.setGeometry( QgsGeometry.fromPointXY(QgsPointXY(pointXY[0], pointXY[1])) )
        pr.addFeatures( [ fet ] )

    # Commit changes
    vl.commitChanges()

    symbol = QgsMarkerSymbol.createSimple({'name': 'circle', 'color': 'red'})
    vl.renderer().setSymbol(symbol)
        
    # Show in project
    root = QgsProject.instance().layerTreeRoot()
    QgsProject.instance().addMapLayer(vl, False)
    root.insertLayer(0, vl)
    vl.triggerRepaint()


    