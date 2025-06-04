import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn

''' LOAD DATA '''

#mat = scipy.io.loadmat('./data/Burgers.mat')

path = "./data"
files = os.listdir(path)
files = [f for f in files if f[-3:] == 'csv']

# location on beam
x = []
# distributed load magnitude
w = []
# corresponding displacement
y = []

for i, file in enumerate(files):
    df = pd.read_csv(path + "/" + file)
    if i==0:
        x = df["Node location (m)"].to_numpy()
    load = float(file.split(".csv")[0])
    w.append(load)
    disp = df["Nodal transverse displacement (m)"].to_numpy()
    y.append(disp)

x = np.array(x) # n_points
w = np.array(w) # n_files
y = np.array(y) # n_files x n_points


''' BEAM PARAMETERS '''
L = 10.0 # meters
E = 210000000000 # Pa
I = 0.0005 # m4
# simply supported