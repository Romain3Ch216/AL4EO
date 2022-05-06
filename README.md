# AL4EO

AL4EO is an Active Learning Toolbox for Earth Observation. QGIS plug-in to come soon!

## Reference

This toolbox was used for our review paper in IEEE Geoscience and Remote Sensing Magazine:

> R. Thoreau, V. Achard, L. Risser, B. Berthelot and X. Briottet, "*Active Learning for Hyperspectral Image Classification: a Comparative Review*", in IEEE Geoscience and Remote Sensing Magazine, to be published soon.

## Requirements

This tool is compatible with Python 3.6+.

The easiest way to install this code is to create a [Python virtual environment](https://docs.python.org/3/tutorial/venv.html) and to install dependencies using:
`pip install -r requirements.txt`

(on Windows you should use `pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html`)

## Setup

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


### Adding a new dataset

Adding a custom dataset can be done by modifying the `data/datasets.py` file and the `data/sensors.py` file.

## Usage

To run benchmark experiments on already labeled datasets, run the script `main.py`. The mandatory arguments are:
 * `--dataset` to specify which dataset to use (e.g. Houston, Indian Pines, PaviaU, Mauzac...)
 * `--query` to specify which active learning query system to use (e.g. breaking_tie, coreset, bald...)
 * `--steps` to specify the number of steps
 * `--n_px` to specify the number of pixels to select at each step
 * `--run` to specify which dataset split to use (it should be in the following format: 'myrun-1')

Pixels queried by the AL algorithm will be stored in a `history_YY_MM_DD_HH_mm.pkl` file.
To compute the metrics (as we did in our review paper), run `python results.py history_YY_MM_DD_HH_mm.pkl`.
Results will be saved in a `results_YY_MM_DD_HH_mm.pkl`. Finally, to plot figures, run `python plot.py results_YY_MM_DD_HH_mm.pkl`.

In an operational context, to run an active learning step before labeling the pixels yourself, you should add the `--op` argument.
It will save the results of the query in a `history_YY_MM_DD_HH_mm_step_x.pkl` file.
To label the queried pixels, run the following:
`python oracle.py history_YY_MM_DD_HH_mm_step_x.pkl`
It will save the results in a `oracle_YY_MM_DD_HH_mm_step_x.pkl` file.
To run the second step, run the script `main.py` with the extra argument `--restore oracle_YY_MM_DD_HH_mm_step_x.pkl`
