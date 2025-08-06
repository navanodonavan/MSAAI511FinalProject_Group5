# Project Title : Music Genre and Composer Classification Using Deep Learning
This project is a part of the AAI-511 course in the Applied Artificial Intelligence Program at the University of San Diego (USD). 

### Project Status: [Completed]

## Installation

Launch Jupyter notebook and open the `MSAAI511_CNN_LSTM_Model_Group5_Final.ipynb` file from this repository. 

## Required libraries to be installed including:

    import pretty_midi
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import LabelEncoder
    import sys
    import os
    from pathlib import Path
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

**For PyTorch**

Import the following (in addition to the above)

    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import torch.optim as optim

Please note, to utilize CUDA, the appropriate PyTorch CUDA version needs to be installed in your environment via pip or conda.
  
## Project Intro/Objective

The primary objective of this project is to develop a deep learning model that can predict the composer of a given musical score accurately. The project aims to accomplish this objective by using two deep learning techniques: Long Short-Term Memory (LSTM) and Convolutional Neural Network (CNN).

### Partner(s)/Contributor(s)

•	Donavan Trigg

•	Michael Skirvin

•	Matthew Ongcapin


### Methods Used

•	Classification

•	Machine Learning

•	Neural Networks

•	Deep Learning


### Technologies

•	Python

•	Jupyter Notebook

•	PyTorch


### Project Description

Leveraging CNNs and LSTMS to train and classify music genre and composers from MIDI files.
