import numpy as np
import pandas as pd
import os
import zipfile
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from tqdm import tqdm

from pinn_class import BeamPINN
from helper import *

# configure the device for GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":

    ''' ----------------------- BEAM PARAMETERS ----------------------- '''

    L = 1.0
    E = 1.0
    I = 1.0

    ''' ----------------------- NETWORK PARAMETERS ----------------------- '''

    # x and w(x) inputs (currently 11 discretized points on beam)
    num_input = 12
    # 1 output, y
    num_output = 1
    epochs = 500

    # list of different sets of hyperparameters to tune the model
    # [num_neurons, num_layers, learning_rate, w_decay, lambda_PDE, lambda_BC]
    hyperparameters = [[32, 2, 1e-4, 1e-4, 1e-2, 1e-1, 1e-2, 1e-1],
                       [32, 2, 1e-5, 1e-4, 1e-2, 1e-1, 1e-2, 1e-1]]

    num_neurons = 32
    num_layers = 2
    # hyperparameters
    learning_rate = 1e-4
    w_decay = 1e-4
    epochs = 1000

    for hyperparameter in hyperparameters:

        num_neurons = hyperparameter[0]
        num_layers = hyperparameter[1]
        learning_rate = hyperparameter[2]
        w_decay = hyperparameter[3]
        lambda_PDE = hyperparameter[4]
        lambda_BC = hyperparameter[5]

        model_name = '_'.join([str(num_neurons),str(num_layers),str(learning_rate),str(w_decay),
                              str(lambda_PDE),str(lambda_BC)])

        ''' ----------------------- IMPORT DATA ----------------------- '''

        X_train, Y_train, DYDX_train, X_validation, Y_validation, DYDX_validation, X_test, Y_test, DYDX_test \
            = prepare_data(device)

        # flatten the training tensors, but maintain the structure of validation and testing because we 
        # need to plot them later on a beam i.e) keep all points associated with a "beam" together
        X_train = X_train.reshape(-1, num_input)
        Y_train = Y_train.reshape(-1, num_output)
        DYDX_train = DYDX_train.reshape(-1, num_output)

        # split training data into BC and domain points
        X_domain_train, X_bc_train, Y_domain_train_true, Y_bc_train_true = separate_domain_bc(X_train, Y_train, L)

        ''' ----------------------- INITIALIZE THE MODEL ----------------------- '''

        # define network
        model = BeamPINN(num_input, num_output, num_neurons, num_layers).to(device)
        # define optimizer
        #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=w_decay)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=w_decay)

        # arrays to store training data
        train_loss_PDE = []
        train_loss_BC1 = []
        train_loss_BC2 = []
        train_loss_data_displacement = []
        train_loss_data_slope = []
        train_loss = []

        ''' ----------------------- TRAINING LOOP ----------------------- '''

        for epoch in tqdm(range(epochs)):

            # train the model
            Y_domain_train_pred = model(X_domain_train)
            Y_bc_train_pred = model(X_bc_train)
            Y_full_train_pred = model(X_train)

            # compute the loss
            loss_PDE_train = model.PDE_loss(Y_domain_train_pred, X_domain_train, E, I)
            loss_BC1_train, loss_BC2_train = model.boundary_loss(Y_bc_train_pred, X_bc_train, E, I)
            loss_data_displacement_train = model.displacement_data_loss(Y_full_train_pred, Y_train)
            loss_data_slope_train = model.slope_data_loss(Y_full_train_pred, X_train, DYDX_train)
            
            loss_train = loss_BC1_train + lambda_BC * loss_BC2_train + lambda_PDE * loss_PDE_train + \
                loss_data_displacement_train + loss_data_slope_train

            # track loss during training
            train_loss_PDE.append(loss_PDE_train.item())
            train_loss_BC1.append(loss_BC1_train.item())
            train_loss_BC2.append(loss_BC2_train.item())
            train_loss_data_displacement.append(loss_data_displacement_train.item())
            train_loss_data_slope.append(loss_data_slope_train.item())
            train_loss.append(loss_train.item())

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss_train.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            
            if epoch % 100 == 0:
                tqdm.write(f'Epoch: {epoch}, Training Loss: {round(loss_train.item(), 5)}')

        ''' ----------------------- SAVE TRAINING RESULTS ----------------------- '''

        num_epochs = list(range(len(train_loss)))

        plot_training_loss(train_loss_PDE, train_loss_BC1, train_loss_BC2, train_loss_data_displacement,
            train_loss_data_slope, train_loss, num_epochs, model_name)
        
        # save model 
        os.makedirs('./models', exist_ok=True)
        torch.save(model.state_dict(), f'./folds/{model_name}.pt')

        ''' ----------------------- VALIDATE MODEL ----------------------- '''

        for i in range(len(X_validation)):

            # run model on the validation set and plot against analytical solution
            X_beam = X_validation[i]
            Y_validation_pred = model(X_beam)
            # get the derivatives
            DY_validation_pred, D2Y_validation_pred, D3Y_validation_pred, D4Y_validation_pred \
            = model.return_derivatives(Y_validation_pred, X_beam)
            # plot the results
            plot_beam_results(Y_validation_pred, DY_validation_pred, D2Y_validation_pred, D3Y_validation_pred, \
                D4Y_validation_pred)