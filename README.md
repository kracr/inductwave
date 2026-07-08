# InductWave

The repository contains implementation details for InductWave. 
We have included a test case for 175% ratio in this code.

Prepare the environment similar to InductiveQE - https://github.com/DeepGraphLearning/InductiveQE

install python library 'cupy-cuda11x' additionally

Copy files -
rspmm.cu
rspmm.cpp
rspmm.h
engine.py
from here to the location where they are installed in torchdrug library

The config folder contains the hyperparameters to run the code.

The data folder will contain the data for querying.

The inductwave folder contains files to run model.
	- utillocal.py contains code to generate wavelet embeddings, you can generate them by uncommenting the command in gnn.py.

The utils folder contains files to generate queries.

The script folder contains the script to run the model.

Train queries - https://drive.google.com/file/d/1s4hCvJg4p8ScD3-lcMzaFFvGWBd4vhVZ/view?usp=sharing
Download train queries from above link and save them in data folder with name 175

Wave emb - https://drive.google.com/file/d/1z97FsNqeH3TULsXzYGb9KlemWkLNkiQW/view?usp=sharing
Download wavelet embeddings from the above link and update the path in inductwave/gnn.py

run runnr.py to train the model.

Some part of the code is raken from InductiveQE and pyTorch Geometric

All the baselines are taken from InductiveQE - https://github.com/DeepGraphLearning/InductiveQE
