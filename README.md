# Comparison of Attention Mechanisms in Machine Learning Models for Vehicle Routing Problems

### This repository contains the implementation of our project titled 'Comparison of Attention Mechanisms in Machine Learning Models for Vehicle Routing Problems' (TensorFlow2). 

### <a href="https://github.com/mvkrishna2001/">Vamsi Krishna Munjuluri</a>, <a href="https://github.com/SanathKumar123/">Sanath Kumar Parimi</a>

This work was done as a final year project under the guidance of Dr. Georg Gutjahr".
 
<hr style="width:50%;text-align:left;margin-left:0">

# File Description

 0) **AM-D for VRP Report.ipynb** - demo report notebook
 1) **enviroment.py** - enviroment for VRP RL Agent
 2) **layers.py** - MHA layers for encoder
 3) **attention_graph_encoder.py** - Graph Attention Encoder
 4) **reinforce_baseline.py** - class for REINFORCE baseline
 5) **attention_dynamic_model_x-head(s).py** - main model and decoder with x number of head(s)
 6) **train.py** - defines training loop which we use in train_model.ipynb
 7) **train_model.ipynb** - from this file one can start training or continue training from chechpoint
 8) **utils.py** and **utils_demo.py** - various auxiliary functions for data creation, saving and visualisation
 9) **lkh3_baseline** folder - everything for running LKH algorithm + logs.
 10) results folder: folder name is ADM_VRP_{graph_size}_{batch_size}. There are training logs, learning curves and saved models in each folder 
 
 # Training procedure:
  1) Open  **train_model.ipynb** and choose training parameters and choose the required number of heads by editing the file name at 'from attention_dynamic_model' .
  2) All outputs would be saved in current directory.

Majority of this code has been fetched from a publicly available repository of <a href="https://github.com/d-eremeev/ADM-VRP/"> The Dynamic Attention Model </a> 
