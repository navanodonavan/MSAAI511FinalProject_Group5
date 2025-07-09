# Project Title : Music Genre and Composer Classification Using Deep Learning
This project is a part of the AAI-511 course in the Applied Artificial Intelligence Program at the University of San Diego (USD). 

### Project Status: [In Progress]

## Installation

Launch Jupyter notebook and open the `TBD` file from this repository. 

## Required libraries to be installed including:

    import math
    import os
    import librosa
    import librosa.display
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from collections import Counter

**For HuBERT Model**

Import the following (in addition to the above)

    from transformers import Wav2Vec2Processor, HubertModel, Wav2Vec2Model
    from transformers import Wav2Vec2FeatureExtractor
    from torch.utils.data import DataLoader, TensorDataset, Dataset
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

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

Leveraging CNNs and LSTMS to train and classify music genre and composers from audio samples.
