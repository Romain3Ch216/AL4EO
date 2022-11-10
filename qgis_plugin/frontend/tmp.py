class PolygonMapTool:
	def __init__(self):
		self.points = []
	def select_point(self, point, mouse_button): 
		geo_pt = QgsGeometry.fromPoint(QgsPoint(point.x(), point.y()))
		self.points.append(geo_pt)



layer = iface.activeLayer()
canvas = iface.mapCanvas() 
pointTool = QgsMapToolEmitPoint(canvas)
polygon_tool = PolygonMapTool()
pointTool.canvasClicked.connect(polygon_tool.select_point)


from qgis.gui import QgsMapToolEmitPoint

class PolygonMapTool(QgsMapToolEmitPoint):
	def __init__(self, canvas, crs):
		self.canvas = canvas
		self.points = []
		self.layer = QgsVectorLayer("Polygon?crs=" + crs, 'poly' , "memory")
		self.pr = layer.dataProvider() 
		QgsMapToolEmitPoint.__init__(self, self.canvas)
	def canvasPressEvent(self, e):
		point = self.toMapCoordinates(self.canvas.mouseLastXY())
		point = QgsPointXY(point[0], point[1])
		pt = QgsFeature()
		pt.setGeometry(QgsGeometry.fromPointXY(point))
		# geo_pt = QgsGeometry.fromPoint(QgsPoint(point[0], point[1]))
		# geo_pt = QgsGeometry.fromPoint(QgsPoint(point[0], point[1]))
		self.points.append(pt)
		print('({:.4f}, {:.4f})'.format(point[0], point[1]))
		if len(self.points) == 3:
			poly = QgsFeature()
			poly.setGeometry(QgsGeometry.fromPolygonXY(self.points))
			self.pr.addFeatures([poly])
			self.layer.updateExtents()
			QgsProject.instance().addMapLayers([self.layer])


crs = qgis.utils.iface.activeLayer().crs().authid()
polygon_tool = PolygonMapTool(iface.mapCanvas(), crs)
iface.mapCanvas().setMapTool(polygon_tool)



points = [point1,QgsPoint(50,150),point2,QgsPoint(100,50)]
# or points = [QgsPoint(50,50),QgsPoint(50,150),QgsPoint(100,150),QgsPoint(100,50)] 

pr.addFeatures([poly])
layer.updateExtents()
QgsMapLayerRegistry.instance().addMapLayers([layer])