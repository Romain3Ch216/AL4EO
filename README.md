<div align="center">
  <img src="https://github.com/Romain3Ch216/AL4EO/blob/qgis-plugin/imgs/logo_onera.png" alt="drawing" width="200"/>
  <img src="https://github.com/Romain3Ch216/AL4EO/blob/qgis-plugin/imgs/logo_magellium.png" alt="drawing" width="150" />
</div>

<br />

<p align="center">
  <img alt="GitHub" src="https://img.shields.io/github/license/Romain3Ch216/AL4EO?color=brightgreen">
  <img alt="Github" src="https://img.shields.io/badge/version-beta-9cf">
</p>

# AL4EO

AL4EO/qgis-plugin is a <a href="https://www.qgis.org/fr/site/"> QGIS </a> plug-in to run Active Learning techniques on remote sensing data. <br/>  
The goal of Active Learning is to interactively build optimal training data sets for supervised learning.
For further details on Active Learning for Earth observation, you can read our comparative review <a href="https://ieeexplore.ieee.org/document/9774342">here</a>.

To run benchmark experiments, see the branch `benchmark` of this repository.

## Requirements

### QGIS

We suggest to install QGIS on a [conda virtual environment](https://docs.python.org/3/tutorial/venv.html) with a 1.3b2 [rasterio](https://rasterio.readthedocs.io/en/latest/) version.

To install the plug-in, compress the ```qgis_plugin``` folder into a zip file and use the QGIS plug-in manager. 

### Python dependencies 

AL4EO is compatible with Python 3.6+ and PyTorch 1.10.0+.

The easiest way to install the backend code is to create a AL4EO [conda virtual environment](https://docs.python.org/3/tutorial/venv.html) and to install dependencies using:
`pip install -r requirements.txt`

(on Windows you should use `pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html`)

## Usage 

### Active Learning Framework

The two major building blocks of the Active Learning Framework are: 
 * the **acquisition function**, that tells how informative the unlabeled pixels are,
 * the **oracle**, that labels the queried pixels (the pixels that maximise the acquisition function).

Active Learning techniques iteratively query pixels as follows:

<img src="https://github.com/Romain3Ch216/AL4EO/blob/qgis-plugin/imgs/al_algo.png" alt="active_learning_flowchart" width="700" />

At this time, the qgis plug-in handles two acquisition functions:
 * <a href="https://www.jmlr.org/papers/volume6/luo05a/luo05a.pdf">Breaking Tie<a/>, an inter-class uncertainty heuristic,
 * <a href="https://arxiv.org/abs/1112.5745">BALD<a/>, an epistemic uncertainty heuristic.

See more acquisition functions in AL4EO/benchmark.

### Models
  
Acquisition functions are often based on machine learning classifiers that are defined in `AL4EO/learning/models.py`. 
  
### Data 

AL4EO plug-in handles data in the [ENVI](https://www.l3harrisgeospatial.com/docs/enviimagefiles.html#:~:text=The%20ENVI%20image%20format%20is,an%20accompanying%20ASCII%20header%20file.) format. Your data folder should be organized as follows:

```
Data
├── image.tiff
├── image.hdr
├── labels.tiff
├── labels.hdr
```

The `image.tiff` file is the image to annotate. Its `image.hdr` header should contain a `bbl` attribute that lists the bad band multiplier values of each band in the image, typically 0 for bad bands and 1 for good bands.

The `labels.tiff` file is the ground truth, encoded in 8-bits, that contains the initial annotations before the first Active Learning step.
Its `labels.hdr` header should contain a `classes`, a `class lookup` and a `class names` attributes that specify the number of classes, the RGB colors of the classes and the names of the classes, respectively. 

```
classes = 2
class lookup = {255 0 0 0 255 0}
class names = {Vegetation Buildings}
```

## How to start
  
Launch the backend in your AL4EO conda virtual environment with the following command line:
  `python -m server` 
  
In QGIS, 
  * Click on the `Q` button to start a query,
  * Select your acquisition function and the various hyperparameters,
  * When the query is completed (see progress in the terminal), queried pixels are showed by red points,
  * In the annotation window, select or add a class, enable the annotation by clicking on the cell below and label the pixels with the mouse left click,
  * Start over.

## Reference

This toolbox was used for our review paper in IEEE Geoscience and Remote Sensing Magazine:

> R. Thoreau, V. ACHARD, L. Risser, B. Berthelot and X. BRIOTTET, "Active Learning for Hyperspectral Image Classification: A Comparative Review," in IEEE Geoscience and Remote Sensing Magazine, doi: 10.1109/MGRS.2022.3169947.
