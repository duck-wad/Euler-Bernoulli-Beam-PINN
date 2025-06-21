import numpy as np
import pandas as pd
import os
import zipfile
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold

from pinn_class import BeamPINN

# configure the device for GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":

    ''' ----------------------- BEAM PARAMETERS ----------------------- '''

    #L = 10.0 # meters
    #E = 210000000000 # Pa
    #I = 0.0005 # m4
    L = 1.0
    E = 1.0
    I = 1.0
    tol = 1e-5
    # simply supported


    ''' ----------------------- NETWORK PARAMETERS ----------------------- '''

    # 2 inputs, x and w
    num_input = 2
    # 1 output, y
    num_output = 1
    num_neurons = 128
    num_layers = 3
    num_folds = 5 # for k fold
    # define the hyperparameters
    learning_rate = 1e-4
    w_decay = 1e-4
    momentum = 0.9
    epochs = 10000
    lambda_1 = 1e-2 # balance term for boundary condition
    lambda_2 = 1e-3 # balance term for PDE
    lambda_3 = 1e-2 # balance term for data loss
    batch_size = 2048


    ''' ----------------------- PREPARE DATA ----------------------- '''

    with zipfile.ZipFile('./compressed_data.zip', 'r') as zip_ref:
        zip_ref.extractall('./data')
    path = './data'
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
    X_test = torch.tensor(X_test, requires_grad=True, dtype=torch.float32).to(device)
    Y_test = torch.tensor(Y_test, requires_grad=True, dtype=torch.float32).to(device)
    # strip out the test samples from X_all 
    X_all = X_all[train_index]
    Y_all = Y_all[train_index]

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
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
        
        # define network
        model = BeamPINN(num_input, num_output, num_neurons, num_layers).to(device)
        # define optimizer
        #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=w_decay)
        #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

        fold_validation_loss = []

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
                #print(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            # compute validation loss 
            if epoch % 50 == 0:
                with torch.no_grad():
                    Y_valid_prediction = model(X_valid)
                    data_loss_validation = model.data_loss(Y_valid_prediction, Y_valid)
                    print(f'Epoch: {epoch}, Data Loss: {data_loss_validation} ')
                    fold_validation_loss.append(data_loss_validation.item())
        
        validation_loss.append(fold_validation_loss)

        # run model on test data to compute testing loss
        with torch.no_grad():
            Y_test_prediction = model(X_test)
            data_loss_test = model.data_loss(Y_test_prediction, Y_test)
            print(f'Fold: {fold + 1}, Testing Data Loss: {data_loss_test} ')
            testing_loss.append([fold+1, data_loss_test.item()])

        os.makedirs('./folds', exist_ok=True)
        torch.save(model.state_dict(), f'./folds/fold_{fold+1}.pt')

    # save testing and validation loss for each model to csv
    os.makedirs('./loss', exist_ok=True)
    df_test = pd.DataFrame(testing_loss, columns=['Fold', 'Test_Loss'])
    df_test.to_csv('./loss/testing_loss.csv', index=False)

    df_validation = pd.DataFrame(validation_loss).T
    df_validation.columns = [f'Fold_{i+1}' for i in range(df_validation.shape[1])]
    df_validation.to_csv('./loss/validation_loss.csv', index=False)