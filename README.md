<div align="center">
  <img src="https://github.com/Romain3Ch216/AL4EO/blob/qgis-plugin/imgs/logo_onera.png" alt="drawing" width="200"/>
  <img src="https://github.com/Romain3Ch216/AL4EO/blob/qgis-plugin/imgs/logo_magellium.png" alt="drawing" width="150" />
</div>

<br />

<p align="center">
  <img alt="GitHub" src="https://img.shields.io/github/license/Romain3Ch216/AL4EO?color=brightgreen">
  <img alt="Github" src="https://img.shields.io/badge/version-0.1-9cf">
</p>

# AL4EO

AL4EO/qgis-plugin is a <a href="https://www.qgis.org/fr/site/"> QGIS </a> plug-in to run Active Learning techniques on Earth observation data. <br/>  

The goal of Active Learning is to interactively and iteratively build optimal training data sets for supervised learning. <br/>
For further details on Active Learning for remote sensing data, see our comparative review <a href="https://ieeexplore.ieee.org/document/9774342" target="_blank">here</a> for which we used AL4EO.

If you use AL4EO as part of a published research project, please see the Reference section below to cite our paper.

To run benchmark experiments, see the [benchmark](https://github.com/Romain3Ch216/AL4EO/tree/benchmark) branch of this repository. 

*Despite our constant efforts to improve the user experience, AL4EO can still be tricky to use. Feel free to open an issue on Github or directly send us an email at <a href="mailto:romain.thoreau@onera.fr;" target="_blank">romain.thoreau@onera.fr</a>.*

## Requirements

### QGIS

We suggest to install QGIS on a [conda virtual environment](https://docs.python.org/3/tutorial/venv.html) with the latest [matplotlib](https://pypi.org/project/matplotlib/) version.

To install the plug-in, compress the ```qgis_plugin``` folder into a zip file and use the QGIS plug-in manager. 

To plot pixels spectra, we strongly recommend to install the [QGIS Temporal/Spectral Profile Tool](https://plugins.qgis.org/plugins/temporalprofiletool/).

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

At this time, the qgis plug-in handles three acquisition functions:
 * <a href="https://www.jmlr.org/papers/volume6/luo05a/luo05a.pdf">Breaking Tie<a/>, an inter-class uncertainty heuristic,
 * <a href="https://arxiv.org/abs/1112.5745">BALD<a/>, an epistemic uncertainty heuristic,
 * Core-Set, a representativeness heuristic.

See more acquisition functions in [AL4EO/benchmark](https://github.com/Romain3Ch216/AL4EO/tree/benchmark).

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
Its `labels.hdr` header should contain a `classes`, a `class lookup` and a `class names` attributes that specify the number of classes, the RGB colors of the classes and the names of the classes, respectively. For instance:

```
classes = 2
class lookup = {255 0 0 0 255 0}
class names = {Vegetation Buildings}
```

## How to start
  
Launch the backend in your AL4EO conda virtual environment with the following command line:
  ```python -m server```
  
<p align="center">
  <img alt="demo" src="https://github.com/Romain3Ch216/AL4EO/blob/qgis-plugin/imgs/demo.png" width="800">
</p>


In QGIS, 
  * the `Q` button within block 1 in the overhead picture opens a window where you can select:<br>
     * the layers on which you run the query, the acquisition function, various hyperparameters and whether to use the preprocessing step introduced <a href="https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLIII-B3-2022/435/2022/" target="_blank">here</a>.
  * the red rectangle within block 1 in the overhead picture allows to select a specific geographic zone. Only pixels from this zone will be queried.
  * Once the query is completed (see progress in the terminal), a history layer pops up (see the top of block 2 in the overhead picture). Red points indicate pixels to be labeled.
  * Block 3 pops up after the query is completed: 
    * Select a class with the dropdown menu "A" (the "Untitled" class allows to remove wrong labels),
    * Select a selection option with the dropdown menu "B": 
       * select pixels one by one,
       * or select polygons (one click saves one edge while double click closes the polygon),
  * **click** on the "Annotation" button "C" below the dropdown menus to start labeling.
  * To add a class, fill the empty box "D" with the class name, choose a color with the menu "E" and click on the button "F" to confirm
  
## Reference

This toolbox was used for our review paper in IEEE Geoscience and Remote Sensing Magazine:

> R. Thoreau, V. Achard, L. Risser, B. Berthelot and X. Briottet, "Active Learning for Hyperspectral Image Classification: A Comparative Review," in IEEE Geoscience and Remote Sensing Magazine, doi: 10.1109/MGRS.2022.3169947.
  
```
@ARTICLE{9774342,  
author={Thoreau, Romain and Achard, Veronique and Risser, Laurent and Berthelot, Beatrice and Briottet, Xavier},  
journal={IEEE Geoscience and Remote Sensing Magazine},   
title={Active Learning for Hyperspectral Image Classification: A Comparative Review},   
year={2022},    
pages={2-24},  
doi={10.1109/MGRS.2022.3169947}}
```
  
If you use the preprocessing method implemented in AL4EO as part of your work, please cite the following paper: 

> Thoreau, R., Achard, V., Risser, L., Berthelot, B., and Briottet, X.: ACTIVE LEARNING ON LARGE HYPERSPECTRAL DATASETS: A PREPROCESSING METHOD, Int. Arch. Photogramm. Remote Sens. Spatial Inf. Sci., XLIII-B3-2022, 435–442, https://doi.org/10.5194/isprs-archives-XLIII-B3-2022-435-2022, 2022

```
@Article{isprs-archives-XLIII-B3-2022-435-2022,
AUTHOR = {Thoreau, R. and Achard, V. and Risser, L. and Berthelot, B. and Briottet, X.},
TITLE = {ACTIVE LEARNING ON LARGE HYPERSPECTRAL DATASETS: A PREPROCESSING METHOD},
JOURNAL = {The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
VOLUME = {XLIII-B3-2022},
YEAR = {2022},
PAGES = {435--442},
URL = {https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLIII-B3-2022/435/2022/},
DOI = {10.5194/isprs-archives-XLIII-B3-2022-435-2022}
}
```
