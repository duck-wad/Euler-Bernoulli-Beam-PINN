import numpy as np
import pandas as pd
import zipfile
import os



# if beam is discretized into n points, the network will expect n+1 inputs
# 1 input is the x location on the beam
# n inputs is the w(x) load at each discretized point on the beam
# so the NN will learn to output a y(x), given a location on the beam x under the load w(x)

def prepare_data():
    
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

        # in the future if w(x) is not a UDL, will need to handle this differently
        load = float(file.split(".csv")[0])
        w.append(load)

    x = np.array(x) #n_files x n_points
    y = np.array(y) # n_files x n_points
    # w is not sorted because of sorting standard in file explorer, sort by ascending
    w = np.sort(np.array(w)) # n_files
    # put w into n_files x n_points x n_points
    # for each x point, it corresponds with the full w(x) over the beam
    w = w[:,np.newaxis]
    w = np.repeat(w, len(x[0]), axis=1)
    w = w[:,:,np.newaxis]
    w = np.repeat(w, len(x[0]), axis=2)

    # normalize data so everything is between 0 to 1
    # for x, scale everything by the max x value (the length of the beam)
    # for w and y, since they vary by orders of magnitude over the entire dataset, 
    # perform lognormal operation and then scale to be 0 to 1
    x = x / np.max(x)
    w = np.log(w)
    w = w / np.max(w)
    # y values could be negative. take the log of the absolute value, then multiply by the sign
    
    print(y[0])

    y = np.log(y)
    print(y[0])
    exit()
    y = y / np.max(y)

    print(w.shape)
    print(x.shape)
    exit()

    # stack x and w into one input 
    X_all = np.stack((x,w[:,:,0]), axis=2)
    print(X_all)
    exit()
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

prepare_data()