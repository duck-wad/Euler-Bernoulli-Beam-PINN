import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

# configure the device for GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


''' BEAM PARAMETERS '''

L = 10.0 # meters
E = 210000000000 # Pa
I = 0.0005 # m4
# simply supported


''' DEFINE PINN CLASS '''

class BeamPINN(nn.Module):
    def __init__(self, num_input, num_output, num_neurons, num_layers):
        super().__init__()
        activation = nn.Tanh
        # define the input layer
        self.input_layer = nn.Sequential(nn.Linear(num_input, num_neurons), activation())
        # define the hidden layer as sequence of num_layers
        self.hidden_layer = nn.Sequential(*[nn.Sequential(nn.Linear(num_neurons, num_neurons), activation()) for _ in range(num_layers-1)])
        # define the output layer with no activation
        self.output_layer = nn.Linear(num_neurons, num_output)
    def forward(self, x, w):
        # torch.cat expects arrays of (N, 1), unsqueeze to get in that form
        temp = torch.cat([x.unsqueeze(1),w[3].unsqueeze(1)],dim=1)
        temp = self.input_layer(temp)
        temp = self.hidden_layer(temp)
        temp = self.output_layer(temp)
        return temp
    def PDE_loss():
        pass

    def boundary_loss():
        pass

    def data_loss():
        pass

    
''' DEFINE NETWORK '''

# 2 inputs, x and w
num_input = 2
# 1 output, y
num_output = 1
# default neurons and layers from NVIDIA PhysicsNemo config
num_neurons = 512
num_layers = 6
# define the hyperparameters
learning_rate = 1e-3
epochs = 1000
lambda_1 = 1e-1 # balance term for boundary condition
lambda_2 = 1e-4

# define network
pinn = BeamPINN(num_input, num_output, num_neurons, num_layers).to(device)
# define optimizer
optimizer = torch.optim.Adam(pinn.parameters(), lr=learning_rate)


''' PREPARE DATA '''

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

    x_temp = df["Node location (m)"].to_numpy()
    x.append(x_temp)

    y_temp = df["Nodal transverse displacement (m)"].to_numpy()
    y.append(y_temp)

    load = float(file.split(".csv")[0])
    w.append(load)

x = np.array(x) # n_files x n_points 
y = np.array(y) # n_files x n_points
# w is not sorted because of sorting standard in file explorer, sort by ascending
w = np.sort(np.array(w)) # n_files
# put w into same shape as x and y
w = w[:,np.newaxis]
w = np.repeat(w, len(x[0]), axis=1)

# stack x and w into one input 
X_all = np.stack((x,w), axis=2)
# make y 3D array
Y_all = y[...,np.newaxis]

# pull random sample of 20% for testing
holdout_size = round(len(X_all)*0.2)
test_index = np.random.choice(len(X_all), size=holdout_size, replace=False)
all_index = np.arange(len(X_all))
# get the remaining indices that aren't test_index
train_index = np.setdiff1d(all_index, test_index)
X_test = X_all[test_index]
Y_test = Y_all[test_index]
# strip out the test samples from X_all 
X_all = X_all[train_index]
Y_all = Y_all[train_index]

# kfold cross-validation with 5 splits
kf = KFold(n_splits=5, shuffle=True, random_state=69)


''' DEFINE TRAINING LOOP '''

for fold, (train_index, valid_index) in enumerate(kf.split(X_all)):
    print(f"Fold {fold + 1}")
    # flatten the training and validation data into a flat list of points 
    # instead of shape (n_beams, n_points, n_features) it become (n_beams x n_points, n_features)
    X_train = (X_all[train_index]).reshape(-1,2)
    Y_train = (Y_all[train_index]).reshape(-1,1)
    X_valid = (X_all[valid_index]).reshape(-1,2)
    Y_valid = (Y_all[valid_index]).reshape(-1,1)
    # convert to torch tensors
    X_train = torch.tensor(X_train, requires_grad=True).to(device)
    Y_train = torch.tesnor(Y_train, requires_grad=True).to(device)
    X_valid = torch.tensor(X_valid, requires_grad=True).to(device)
    Y_valid = torch.tensor(Y_valid, requires_grad=True).to(device)
    
