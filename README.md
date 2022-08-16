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

AL4EO/benchmark is an Active Learning toolbox to run benchmark experiments on Earth Observation data sets. 

This toolbox was used for our <a href="https://ieeexplore.ieee.org/document/9774342">comparative review</a> in IEEE GRSM.

To run Active Learning techniques in an operational context, see the [qgis-plugin](https://github.com/Romain3Ch216/AL4EO/tree/qgis-plugin) branch of this repository. 

## How to cite 

If you used AL4EO in one of your project, we would greatly appreciate if you cite our paper:

> R. Thoreau, V. Achard, L. Risser, B. Berthelot and X. Briottet, "*Active Learning for Hyperspectral Image Classification: A Comparative Review*," in IEEE Geoscience and Remote Sensing Magazine, doi: 10.1109/MGRS.2022.3169947.

```
@ARTICLE{9774342,  
author={Thoreau, Romain and ACHARD, Veronique and Risser, Laurent and Berthelot, Beatrice and BRIOTTET, Xavier},  
journal={IEEE Geoscience and Remote Sensing Magazine},   
title={Active Learning for Hyperspectral Image Classification: A Comparative Review},   
year={2022},    
pages={2-24},  
doi={10.1109/MGRS.2022.3169947}}
```

## Requirements

AL4EO is compatible with Python 3.6+ and PyTorch 1.10.0+.

The easiest way to install the code is to create a AL4EO [conda virtual environment](https://docs.python.org/3/tutorial/venv.html) and to install dependencies using:

`pip install -r requirements.txt`

(on Windows you should use `pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html`)

## Usage 

### Active Learning Framework

The two major building blocks of the Active Learning Framework are: 
 * the **acquisition function**, that tells how informative the unlabeled pixels are,
 * the **oracle**, that labels the queried pixels (the pixels that maximise the acquisition function).

Active Learning techniques iteratively query pixels as follows:

<img src="https://github.com/Romain3Ch216/AL4EO/blob/qgis-plugin/imgs/al_algo.png" alt="active_learning_flowchart" width="700" />

Seven acquisition functions are implemented in AL4EO:
 * <a href="https://www.jmlr.org/papers/volume6/luo05a/luo05a.pdf">Breaking Tie<a/>, an inter-class uncertainty heuristic,
 * <a href="https://arxiv.org/abs/1112.5745">BALD<a/>, an epistemic uncertainty heuristic,
 * <a href="https://proceedings.neurips.cc/paper/2019/hash/95323660ed2124450caaac2c46b5ed90-Abstract.html">Batch-BALD<a/>, an epistemic uncertainty heuristic,
 * <a href="https://arxiv.org/abs/1708.00489">Core-Set<a/>, a representativeness heuristic,
 * <a href="https://openaccess.thecvf.com/content_ICCV_2019/html/Sinha_Variational_Adversarial_Active_Learning_ICCV_2019_paper.html">VAAL<a/>, a representativeness heuristic,
 * <a href="https://dl.acm.org/doi/pdf/10.1145/1390156.1390183?casa_token=YyGsiIMaR6EAAAAA:uKtcjncNn2TuL4g-Q0aQi2UHULHcQSDSUm0lxTtfzH-_kYZ02_tXW8Kvh8c_OsuWWnevm0muXQ">Hierarchical Sampling<a/>, a representativeness heuristic,
 * <a href="https://proceedings.neurips.cc/paper/2017/hash/8ca8da41fe1ebc8d3ca31dc14f5fc56c-Abstract.html">Learning Active Learning<a/>, a performance heuristic.

### Models
  
Acquisition functions are often based on machine learning classifiers that are defined in `AL4EO/learning/models.py`. 
  
### Data

You should write the path to your folders in the `path.py` file and organize your dataset folder as follows:

```
Datasets
├── PaviaU
│   ├── paviau_img.npy
|   ├── gt1
|       ├── initial_gt.npy
|       ├── pool.npy
|       ├── test_gt.npy
|   ├── gt2
|       ├── initial_gt.npy
|       ├── pool.npy
|       ├── test_gt.npy
├── IndianPines
│   ├── indianpines_img.npy
|   ├── gt1
|       ├── initial_gt.npy
|       ├── pool.npy
|       ├── test_gt.npy

```

Several public hyperspectral datasets are available on the [UPV/EHU](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes) wiki.


### Adding a new data set

Adding a custom dataset can be done by modifying the `data/datasets.py` file and the `data/sensors.py` file.

## How to start

To run benchmark experiments on already labeled datasets, run the script `main.py`. The mandatory arguments are:
 * `--dataset` to specify which dataset to use (e.g. Houston, Indian Pines, PaviaU, Mauzac...)
 * `--query` to specify which active learning query system to use (e.g. breaking_tie, coreset, bald...)
 * `--steps` to specify the number of steps
 * `--n_px` to specify the number of pixels to select at each step
 * `--run` to specify which dataset split to use (it should be in the following format: 'myrun-1')

Pixels queried by the AL algorithm will be stored in a `history_YY_MM_DD_HH_mm.pkl` file.
To compute the metrics (as we did in our review paper), run `python results.py history_YY_MM_DD_HH_mm.pkl`.
Results will be saved in a `results_YY_MM_DD_HH_mm.pkl` file. Finally, to plot figures, run `python plot.py results_YY_MM_DD_HH_mm.pkl`.

### Operational context

For an operational context, we suggest to use the [qgis-plugin](https://github.com/Romain3Ch216/AL4EO/tree/qgis-plugin). However, you could also proceed as follows.

To run an active learning step before labeling the pixels yourself, you should add the `--op` argument.
It will save the results of the query in a `history_YY_MM_DD_HH_mm_step_x.pkl` file.
To label the queried pixels (visualize and annotate on a local webpage), run the following:
`python oracle.py history_YY_MM_DD_HH_mm_step_x.pkl`. 
It will save the results in a `oracle_YY_MM_DD_HH_mm_step_x.pkl` file.
To run the second step, run the script `main.py` with the extra argument `--restore oracle_YY_MM_DD_HH_mm_step_x.pkl`
