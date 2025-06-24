import numpy as np
import pandas as pd
import os
import zipfile
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

from pinn_class import BeamPINN
from prepare_data import prepare_data

# configure the device for GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":

    ''' ----------------------- BEAM PARAMETERS ----------------------- '''

    L = 1.0
    E = 1.0
    I = 1.0
    tol = 1e-5

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
    epochs = 40000

    ''' ----------------------- IMPORT DATA ----------------------- '''

    X_all, Y_all, X_test, Y_test = prepare_data()
    X_all = torch.tensor(X_all, requires_grad=True, dtype=torch.float32).to(device)
    Y_all = torch.tensor(Y_all, requires_grad=True, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, requires_grad=True, dtype=torch.float32).to(device)
    Y_test = torch.tensor(Y_test, requires_grad=True, dtype=torch.float32).to(device)


    # kfold cross-validation with 5 splits
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=69)

    ''' ----------------------- TRAINING LOOP ----------------------- '''

    # for each of k folds, iterate for {epoch} epochs
    # in each epoch
    validation_loss = []
    testing_loss = []

    for fold, (train_index, valid_index) in enumerate(kf.split(X_all)):
        print(f"Fold {fold + 1}")
        # flatten the training and validation data into a flat list of points 
        # instead of shape (n_beams, n_points, n_features) it become (n_beams x n_points, n_features)
        X_train = (X_all[train_index]).reshape(-1,num_input)
        Y_train = (Y_all[train_index]).reshape(-1,num_output)
        X_valid = (X_all[valid_index]).reshape(-1,num_input)
        Y_valid = (Y_all[valid_index]).reshape(-1,num_output)

        # batch the training data into batches of {batch_size}
        train_dataset = TensorDataset(X_train, Y_train)
        valid_dataset = TensorDataset(X_valid, Y_valid)
        
        # define network
        model = BeamPINN(num_input, num_output, num_neurons, num_layers).to(device)
        # define optimizer
        #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=w_decay)

        fold_loss_PDE = []
        fold_loss_BC1 = []
        fold_loss_BC2 = []
        fold_loss_data = []
        fold_loss = []

        for epoch in range(epochs):
            # find the indices of boundary points where x==0 or x==L
            x_vals = X_train[:,0].unsqueeze(1)
            is_left = torch.abs(x_vals - 0.0) < tol
            is_right =  torch.abs(x_vals - L) < tol
            is_bc = is_left | is_right
            is_domain = ~is_bc

            # split boundary and domain points
            X_domain = X_train[is_domain.squeeze()].detach().clone().requires_grad_()
            X_bc = X_train[is_bc.squeeze()].detach().clone().requires_grad_()

            Y_domain_true = Y_train[is_domain.squeeze()].detach().clone().requires_grad_()
            Y_bc_true = Y_train[is_bc.squeeze()].detach().clone().requires_grad_()

            Y_domain = model(X_domain)
            Y_bc = model(X_bc)

            Y_true = torch.cat([Y_domain_true, Y_bc_true], dim=0)
            Y_pred = torch.cat([Y_domain, Y_bc], dim=0)

            # compute the loss
            loss_PDE = model.PDE_loss(Y_domain, X_domain, E, I)
            loss_BC1, loss_BC2 = model.boundary_loss(Y_bc, X_bc, E, I)
            loss_data = model.data_loss(Y_pred, Y_true)
            
            loss = loss_BC1 + loss_BC2 + loss_PDE + loss_data
            #print(loss)

            # track loss during training
            fold_loss_PDE.append(loss_PDE.item())
            fold_loss_BC1.append(loss_BC1.item())
            fold_loss_BC2.append(loss_BC2.item())
            fold_loss_data.append(loss_data.item())
            fold_loss.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # compute validation loss 
            if epoch % 50 == 0:
                with torch.no_grad():
                    Y_valid_prediction = model(X_valid)
                    data_loss_validation = model.data_loss(Y_valid_prediction, Y_valid)
                    print(f'Epoch: {epoch}, Data Loss: {data_loss_validation} ')
        
        # run model on test data to compute testing loss
        with torch.no_grad():
            Y_test_prediction = model(X_test)
            data_loss_test = model.data_loss(Y_test_prediction, Y_test)
            print(f'Fold: {fold + 1}, Testing Data Loss: {data_loss_test} ')
            testing_loss.append([fold+1, data_loss_test.item()])

        # plot the training loss and save plot
        num_epochs = list(range(len(fold_loss)))
        plt.figure(figsize=(10,6))

        plt.plot(num_epochs, fold_loss_PDE, label=r'PDE Loss ($EI\frac{d^4y}{dx^4} = w$)')
        plt.plot(num_epochs, fold_loss_BC1, label=r'BC1 Loss ($y=0$ for $x=0,L$)')
        plt.plot(num_epochs, fold_loss_BC2, label=r'BC2 Loss ($EI\frac{d^2y}{dx^2}=0$)')
        plt.plot(num_epochs, fold_loss_data, label='Data Loss')
        plt.plot(num_epochs, fold_loss, label='Total Loss', linewidth=2, color='black')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.title('Loss vs Epochs')
        plt.legend()
        plt.tight_layout()
        # save plot to loss folder
        os.makedirs('./loss', exist_ok=True)
        plt.savefig(f'./loss/fold_{fold+1}_training_loss.pdf', format='pdf')

        # save model 
        os.makedirs('./folds', exist_ok=True)
        torch.save(model.state_dict(), f'./folds/fold_{fold+1}.pt')
        
    # save testing and validation loss for each model to csv
    os.makedirs('./loss', exist_ok=True)
    df_test = pd.DataFrame(testing_loss, columns=['Fold', 'Test_Loss'])
    df_test.to_csv('./loss/testing_loss.csv', index=False)