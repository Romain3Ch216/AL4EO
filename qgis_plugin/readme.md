# Presentation
This repository contains the code of the QGIS plug-in part of AL4EO. This is a
[QGIS](https://www.qgis.org/en/site/) plugin designed for interactive semantic 
segmentation of geo-referenced images using deep-learning. 

# Set up

## Install the plugin in QGIS
Compress this repository (`qgis_plugin`) into a zip file and install the plugin with the Qgis plugin manager.

# Use plug-in
The plug-in use the PyQGIS python package version 3.22.4.

The plug-in also use rasterio python package version 1.3b2.

You can download rasterio version 1.3b2 with the command:  
```pip install rasterio==1.3b2```

This project was developed on linux so it is preferable to use this plugin on a linux system.

## Lunch an active learning step
You must launch the python script "server.py" which is in the AL4EO folder.

Click on the "Q" button in QGIS. 
In the dialog select your data layer and your label layer and adjust your settings.

<strong>WARNING</strong>: Your image data and label raster must be in [ENVI](https://www.l3harrisgeospatial.com/docs/enviimagefiles.html#:~:text=The%20ENVI%20image%20format%20is,an%20accompanying%20ASCII%20header%20file.) format
and their file extensions must be ".tiff".  
Your raster label image must have "classes", "class names" and "class lookup" specified.

Then click on the "Ok" button to run the active learning step 

<strong>WARNING</strong>: Your label raster image is going to be reproject in the same size as your data raster image
without copy ! 

You can see the progress of the request in the terminal of the "server.py" script.

When the request is finished, a window appears at the bottom left of QGIS and a layer with the pixels to annotate is created.  
In this window you can choose the class you want to annotate and click on the mouse icon to select the annotation tool.    
Once the annotation tool is selected, you can click on the image label raster to annotate the image.    
You can also add classes to your image which will be written in the header file.

## Annotate directly from a history

You can annotate directly from a history that was already generated by AL4EO.  
Just click on the "A" button and select your label raster image and your "history".  
Then click "Ok", a window appears like after an active learning query.
  





