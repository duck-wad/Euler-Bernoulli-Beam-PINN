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
from prepare_data import prepare_data, separate_domain_bc

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
    num_neurons = 32
    num_layers = 2
    num_folds = 5 # for k fold
    # hyperparameters
    learning_rate = 1e-4
    w_decay = 1e-4
    epochs = 10000
    lambda_PDE = 1e-2
    lambda_BC = 1e-1

    ''' ----------------------- IMPORT DATA ----------------------- '''

    X_all, Y_all, X_test, Y_test = prepare_data()
    X_all = torch.tensor(X_all, requires_grad=True, dtype=torch.float32).to(device)
    Y_all = torch.tensor(Y_all, requires_grad=True, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, requires_grad=True, dtype=torch.float32).to(device)
    Y_test = torch.tensor(Y_test, requires_grad=True, dtype=torch.float32).to(device)

    X_test = X_test.reshape(-1,num_input)
    Y_test = Y_test.reshape(-1,num_output)
    # split testing data into BC and domain points
    X_domain_test, X_bc_test, Y_domain_true_test, Y_bc_true_test = separate_domain_bc(X_test, Y_test, L)

    # kfold cross-validation with 5 splits
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    ''' ----------------------- TRAINING LOOP ----------------------- '''

    for fold, (train_index, valid_index) in enumerate(kf.split(X_all)):
        print(f"Fold {fold + 1}")

        ''' ----------------------- PREPARE THE FOLD DATA ----------------------- '''

        # flatten the training and validation data into a flat list of points 
        # instead of shape (n_beams, n_points, n_features) it become (n_beams x n_points, n_features)
        X_train = (X_all[train_index]).reshape(-1,num_input)
        Y_train = (Y_all[train_index]).reshape(-1,num_output)
        X_valid = (X_all[valid_index]).reshape(-1,num_input)
        Y_valid = (Y_all[valid_index]).reshape(-1,num_output)

        # split dataset into domain points and BC points
        X_domain, X_bc, Y_domain_true, Y_bc_true = separate_domain_bc(X_train, Y_train, L)
        X_domain_valid, X_bc_valid, Y_domain_true_valid, Y_bc_true_valid = separate_domain_bc(X_valid, Y_valid, L)

        ''' ----------------------- INITIALIZE MODEL ----------------------- '''
    
        # define network
        model = BeamPINN(num_input, num_output, num_neurons, num_layers).to(device)
        # define optimizer
        #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=w_decay)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=w_decay)

        # arrays to store training data
        fold_loss_PDE = []
        fold_loss_BC1 = []
        fold_loss_BC2 = []
        fold_loss_data = []
        fold_loss = []
        validation_loss = []
        validation_epochs = []

        ''' ----------------------- TRAIN THE MODEL ----------------------- '''

        for epoch in tqdm(range(epochs)):

            # train the model
            Y_domain = model(X_domain)
            Y_bc = model(X_bc)

            Y_true = torch.cat([Y_domain_true, Y_bc_true], dim=0)
            Y_pred = torch.cat([Y_domain, Y_bc], dim=0)

            # compute the loss
            loss_PDE = model.PDE_loss(Y_domain, X_domain, E, I)
            loss_BC1, loss_BC2 = model.boundary_loss(Y_bc, X_bc, E, I)
            loss_data = model.data_loss(Y_pred, Y_true)
            
            loss = loss_BC1 + lambda_BC * loss_BC2 + lambda_PDE * loss_PDE + loss_data

            # track loss during training
            fold_loss_PDE.append(loss_PDE.item())
            fold_loss_BC1.append(loss_BC1.item())
            fold_loss_BC2.append(loss_BC2.item())
            fold_loss_data.append(loss_data.item())
            fold_loss.append(loss.item())

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            
            # compute validation loss 
            if epoch % 100 == 0:
                #with torch.no_grad():
                Y_domain_valid = model(X_domain_valid)
                Y_bc_valid = model(X_bc_valid)
                Y_true_valid = torch.cat([Y_domain_true_valid, Y_bc_true_valid])
                Y_pred_valid = torch.cat([Y_domain_valid, Y_bc_valid])

                loss_PDE_valid = model.PDE_loss(Y_domain_valid, X_domain_valid, E, I)
                loss_BC1_valid, loss_BC2_valid = model.boundary_loss(Y_bc_valid, X_bc_valid, E, I)
                loss_data_valid = model.data_loss(Y_pred_valid, Y_true_valid)

                loss_valid = loss_BC1_valid + lambda_BC * loss_BC2_valid + lambda_PDE * loss_PDE_valid + loss_data_valid

                validation_loss.append(loss_valid.item())
                validation_epochs.append(epoch)
                tqdm.write(f'Epoch: {epoch}, Total Validation Loss: {round(loss_valid.item(), 5)}')

        ''' ----------------------- TEST MODEL ----------------------- '''
        
        # run model on test data to compute testing loss
        Y_domain_test = model(X_domain_test)
        Y_bc_test = model(X_bc_test)
        Y_true_test = torch.cat([Y_domain_true_test, Y_bc_true_test])
        Y_pred_test = torch.cat([Y_domain_test, Y_bc_test])

        loss_PDE_test = model.PDE_loss(Y_domain_test, X_domain_test, E, I)
        loss_BC1_test, loss_BC2_test = model.boundary_loss(Y_bc_test, X_bc_test, E, I)
        loss_data_test = model.data_loss(Y_pred_test, Y_true_test)
        loss_test = loss_BC1_test + lambda_BC * loss_BC2_test + lambda_PDE * loss_PDE_test + loss_data_test

        tqdm.write(f'Fold: {fold + 1}, Total Testing Loss: {round(loss_test.item(), 5)}')

        ''' ----------------------- PLOT RESULTS ----------------------- '''

        # plot the training loss and save plot
        num_epochs = list(range(len(fold_loss)))
        plt.figure(figsize=(10,6))

        plt.plot(num_epochs, fold_loss_PDE, label=r'Training PDE Loss ($EI\frac{d^4y}{dx^4} = w$)')
        plt.plot(num_epochs, fold_loss_BC1, label=r'Training BC1 Loss ($y=0$ for $x=0,L$)')
        plt.plot(num_epochs, fold_loss_BC2, label=r'Training BC2 Loss ($EI\frac{d^2y}{dx^2}=0$ for $x=0,L$)')
        plt.plot(num_epochs, fold_loss_data, label='Training Data Loss')
        plt.plot(num_epochs, fold_loss, label='Total Training Loss', linewidth=2, color='black')
        plt.plot(validation_epochs, validation_loss, label='Total Validation Loss')
        plt.plot([], [], '', label=f'Total Testing Loss = {round(loss_test.item(), 5)}', linewidth=0, color='white')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.title('Loss vs Epochs')
        plt.legend(loc='upper right')
        plt.tight_layout()
        
        # save plot to loss folder
        os.makedirs('./loss', exist_ok=True)
        plt.savefig(f'./loss/fold_{fold+1}_training_loss.pdf', format='pdf')

        # save model 
        os.makedirs('./folds', exist_ok=True)
        torch.save(model.state_dict(), f'./folds/fold_{fold+1}.pt')