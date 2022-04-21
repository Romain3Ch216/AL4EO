# AL4EO

AL4EO is an Active Learning Toolbox for Earth Observation. 

## Reference 

This toolbox was used for our review paper in IEEE Geoscience and Remote Sensing Magazine:

> R. Thoreau, V. Achard, L. Risser, B. Berthelot and X. Briottet, "*Active Learning for Hyperspectral Image Classification: a Comparative Review*," in IEEE Geoscience and Remote Sensing Magazine, in review.

## Usage 

To run benchmark experiments on already labeled datasets, run the script 'main.py'. The mandatory arguments are:
 * '--dataset' to specify which dataset to use (e.g. Houston, Indian Pines, PaviaU, Mauzac...)
 * '--query' to specify which active learning query system to use (e.g. breaking_tie, coreset, bald...)
 * '--steps' to specify the number of steps 
 * '--n_px' to specify the number of pixels to select at each step
 * '--run' to specify which dataset split to use (it should be in the following format: 'myrun-1')

In an operational context, to run an active learning step before labeling the pixels yourself, you should add the '--op' argument.
It will save the results of the query in a 'history_YY_MM_DD_HH_mm_step_x.pkl' file. 
To label the queried pixels, run the following:
`python oracle.py history_YY_MM_DD_HH_mm_step_x.pkl`
It will save the results in a 'oracle_YY_MM_DD_HH_mm_step_x.pkl' file.
To run the second step, run the script 'main.py' with the extra argument '--restore oracle_YY_MM_DD_HH_mm_step_x.pkl'

