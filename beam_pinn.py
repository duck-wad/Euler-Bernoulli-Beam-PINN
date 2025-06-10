import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold

# configure the device for GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


''' ----------------------- BEAM PARAMETERS ----------------------- '''

L = 10.0 # meters
E = 210000000000 # Pa
I = 0.0005 # m4
tol = 1e-5
# simply supported


''' ----------------------- NETWORK PARAMETERS ----------------------- '''

# 2 inputs, x and w
num_input = 2
# 1 output, y
num_output = 1
# default neurons and layers from NVIDIA PhysicsNemo config
num_neurons = 512
num_layers = 6
# define the hyperparameters
learning_rate = 1e-3
epochs = 50
lambda_1 = 1e-2 # balance term for boundary condition
lambda_2 = 1e-5 # balance term for PDE
lambda_3 = 1e-1 # balance term for data loss
batch_size = 1024


''' ----------------------- PREPARE DATA ----------------------- '''

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

# normalize x by beam length to better generalization
# but need to accomodate for normalization in PDE
# x = np.array(x) / L # n_files x n_points 
x = np.array(x)
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


''' ----------------------- PINN CLASS ----------------------- '''

class BeamPINN(nn.Module):
    def __init__(self, num_input, num_output, num_neurons, num_layers):
        super().__init__()
        activation = nn.Tanh
        # define the input layer
        self.input_layer = nn.Sequential(nn.Linear(num_input, num_neurons), activation())
        # define the hidden layer as sequence of num_layers
        self.hidden_layer = nn.Sequential(*[nn.Sequential(nn.Linear(num_neurons, num_neurons), 
                                                          activation()) for _ in range(num_layers-1)])
        # define the output layer with no activation
        self.output_layer = nn.Linear(num_neurons, num_output)

    def forward(self, X_batch):
        # torch.cat expects arrays of (N, 1), unsqueeze to get in that form
        #temp = torch.cat([x.unsqueeze(1),w[3].unsqueeze(1)],dim=1)
        temp = self.input_layer(X_batch)
        temp = self.hidden_layer(temp)
        temp = self.output_layer(temp)
        return temp
    
    def PDE_loss(self, Y_domain, X_domain):

        # need to do derivatives wrt the original X_domain tensor rather than splicing X_domain to get the 
        # x position tensor, that creates a new tensor and grad won't work
        dy = torch.autograd.grad(Y_domain, X_domain, torch.ones_like(Y_domain), create_graph=True)[0]
        d2y = torch.autograd.grad(dy, X_domain, torch.ones_like(dy), create_graph=True)[0]
        d3y = torch.autograd.grad(d2y, X_domain, torch.ones_like(d2y), create_graph=True)[0]
        d4y = torch.autograd.grad(d3y, X_domain, torch.ones_like(d3y), create_graph=True)[0]

        d4y_dx4 = d4y[:,0].unsqueeze(1)

        # loss is mean squared error
        # multiply RHS by L^4 to account for the length normalization (x is scaled by 1/L)
        # return torch.mean((E*I*d4y_dx4 - X_domain[:,1].unsqueeze(1)*L**4)**2)
        return torch.mean((E*I*d4y_dx4 - X_domain[:,1].unsqueeze(1))**2)

    def boundary_loss(self, Y_bc, X_bc):
        # for simply supported beam, y=0 and d2ydx2=0
        dy = torch.autograd.grad(Y_bc, X_bc, torch.ones_like(Y_bc), create_graph=True)[0]
        d2y = torch.autograd.grad(dy, X_bc, torch.ones_like(dy), create_graph=True)[0]

        d2y_dx2 = d2y[:,0].unsqueeze(1)

        # y=0
        bc_loss1 = torch.mean((Y_bc - 0.)**2)
        # d2ydx2=0
        bc_loss2 = torch.mean((d2y_dx2 - 0)**2)
        return bc_loss1, bc_loss2
        

    def data_loss(self, Y_domain, Y_bc, Y_domain_true, Y_bc_true):
        Y_all_true = torch.cat([Y_domain_true, Y_bc_true], dim=0)
        Y_all_pred = torch.cat([Y_domain, Y_bc], dim=0)
        return torch.mean((Y_all_pred - Y_all_true)**2)

    def compute_loss(self, Y_domain, Y_bc, X_domain, X_bc, Y_domain_true, Y_bc_true):
        loss_PDE = self.PDE_loss(Y_domain, X_domain)
        loss_BC_1, loss_BC_2 = self.boundary_loss(Y_bc, X_bc)
        loss_data = self.data_loss(Y_domain, Y_bc, Y_domain_true, Y_bc_true)
        loss = lambda_1 * loss_BC_1 + lambda_1 * loss_BC_2 + lambda_2 * loss_PDE + lambda_3 * loss_data
        #loss = lambda_1 * loss_BC_1 + lambda_1 * loss_BC_2 + lambda_3 * loss_data
        #loss = loss_data
        return loss
        


''' ----------------------- TRAINING LOOP ----------------------- '''

# for each of k folds, iterate for {epoch} epochs
# in each epoch

for fold, (train_index, valid_index) in enumerate(kf.split(X_all)):
    print(f"Fold {fold + 1}")
    # flatten the training and validation data into a flat list of points 
    # instead of shape (n_beams, n_points, n_features) it become (n_beams x n_points, n_features)
    X_train = (X_all[train_index]).reshape(-1,2)
    Y_train = (Y_all[train_index]).reshape(-1,1)
    X_valid = (X_all[valid_index]).reshape(-1,2)
    Y_valid = (Y_all[valid_index]).reshape(-1,1)
    # convert to torch tensors
    X_train = torch.tensor(X_train, requires_grad=True, dtype=torch.float32).to(device)
    Y_train = torch.tensor(Y_train, requires_grad=True, dtype=torch.float32).to(device)
    X_valid = torch.tensor(X_valid, requires_grad=True, dtype=torch.float32).to(device)
    Y_valid = torch.tensor(Y_valid, requires_grad=True, dtype=torch.float32).to(device)

    # batch the training data into batches of {batch_size}
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = TensorDataset(X_valid, Y_valid)
    valid_dataset = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    
    # define network
    model = BeamPINN(num_input, num_output, num_neurons, num_layers).to(device)
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # find the indices of boundary points where x==0 or x==L
            x_vals = X_batch[:,0].unsqueeze(1)
            is_left = torch.abs(x_vals - 0.0) < tol
            is_right =  torch.abs(x_vals - L) < tol
            is_bc = is_left | is_right
            is_domain = ~is_bc
            # split boundary and domain points

            X_domain = X_batch[is_domain.squeeze()].detach().clone().requires_grad_()
            X_bc = X_batch[is_bc.squeeze()].detach().clone().requires_grad_()

            Y_domain_true = Y_batch[is_domain.squeeze()].detach().clone().requires_grad_()
            Y_bc_true = Y_batch[is_bc.squeeze()].detach().clone().requires_grad_()

            Y_domain = model(X_domain)
            Y_bc = model(X_bc)

            # FILL IN
            loss = model.compute_loss(Y_domain, Y_bc, X_domain, X_bc, Y_domain_true, Y_bc_true)
            print(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # COMPUTE VALIDATION LOSS

            


