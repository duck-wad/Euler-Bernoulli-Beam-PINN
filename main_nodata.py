import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import time
import shutil

from pinn_class import BeamPINN
from helper import *

# configure the device for GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    total_start_time = time.time()

    """----------------------- BEAM PARAMETERS -----------------------"""

    L = 1.0
    E = 1.0
    I = 1.0
    max_load = 100
    num_points = 101

    """ ----------------------- NETWORK PARAMETERS ----------------------- """

    # x and w inputs (constant w, one input)
    num_input = 2
    # 1 output, y
    num_output = 1
    # once this loss is reached, stop training the model
    loss_threshold = 0.0005

    # list of different sets of hyperparameters to tune the model
    # [num_neurons, num_layers, learning_rate, w_decay, lambda_PDE, lambda_BC, max_norm, max_epochs]
    hyperparameters = [
        [64, 3, 1e-3, 1e-4, 1e-2, 1e-0, 1.0, 20000],
        [64, 3, 9e-4, 1e-4, 1e-2, 1e-0, 1.0, 20000],
        [64, 3, 8e-4, 1e-4, 1e-2, 1e-0, 1.0, 20000],
        [64, 3, 7e-4, 1e-4, 1e-2, 1e-0, 1.0, 20000],
        [64, 3, 6e-4, 1e-4, 1e-2, 1e-0, 1.0, 20000],
        [64, 3, 5e-4, 1e-4, 1e-2, 1e-0, 1.0, 20000],
        [64, 3, 4e-4, 1e-4, 1e-2, 1e-0, 1.0, 20000],
        [64, 3, 3e-4, 1e-4, 1e-2, 1e-0, 1.0, 20000],
        [64, 3, 2e-4, 1e-4, 1e-2, 1e-0, 1.0, 20000],
        [64, 3, 1e-4, 1e-4, 1e-2, 1e-0, 1.0, 20000],
    ]

    os.makedirs("./training results", exist_ok=True)
    runtime_file = "./training results/runtime.txt"
    with open(runtime_file, "w") as f:
        f.write("Model Training Runtimes\n")
        f.write("-----------------------\n")

    os.makedirs("./models", exist_ok=True)

    """ ----------------------- GENERATE INPUTS ----------------------- """

    for hyperparameter in hyperparameters:

        min_loss = 1000

        hyper_start_time = time.time()

        num_neurons = hyperparameter[0]
        num_layers = hyperparameter[1]
        learning_rate = hyperparameter[2]
        w_decay = hyperparameter[3]
        lambda_PDE = hyperparameter[4]
        lambda_BC = hyperparameter[5]
        max_norm = hyperparameter[6]
        max_epochs = hyperparameter[7]

        model_name = "_".join(
            [
                str(num_neurons),
                str(num_layers),
                str(learning_rate),
                str(w_decay),
                str(lambda_PDE),
                str(lambda_BC),
                str(max_norm),
                str(max_epochs),
            ]
        )

        """ ----------------------- IMPORT DATA ----------------------- """

        X_train, X_validation, X_test = prepare_linspace(
            L, num_points, max_load, device
        )

        # flatten the training tensors, but maintain the structure of validation and testing because we
        # need to plot them later on a beam i.e) keep all points associated with a "beam" together
        X_train = X_train.reshape(-1, num_input)

        # split training data into BC and domain points
        X_domain_train, X_bc_train, _, _ = separate_domain_bc(
            X_train, [], L, nodata=True
        )

        """ ----------------------- INITIALIZE THE MODEL ----------------------- """

        # define network
        model = BeamPINN(num_input, num_output, num_neurons, num_layers).to(device)
        # define optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=w_decay
        )

        # arrays to store training data
        train_loss_PDE = []
        train_loss_BC1 = []
        train_loss_BC2 = []
        train_loss = []

        """ ----------------------- TRAINING LOOP ----------------------- """

        for epoch in tqdm(range(max_epochs)):

            # train the model
            Y_domain_train_pred = model(X_domain_train)
            Y_bc_train_pred = model(X_bc_train)

            # compute the loss
            loss_PDE_train = model.PDE_loss(Y_domain_train_pred, X_domain_train, E, I)
            loss_BC1_train, loss_BC2_train = model.boundary_loss(
                Y_bc_train_pred, X_bc_train, E, I
            )

            loss_train = (
                loss_BC1_train
                + lambda_BC * loss_BC2_train
                + lambda_PDE * loss_PDE_train
            )

            # track loss during training
            train_loss_PDE.append(loss_PDE_train.item())
            train_loss_BC1.append(loss_BC1_train.item())
            train_loss_BC2.append(loss_BC2_train.item())
            train_loss.append(loss_train.item())

            if loss_train.item() < min_loss:
                min_loss = loss_train.item()
            if min_loss < loss_threshold:
                tqdm.write(f"Minimum loss threshold reached. Loss: {min_loss}")
                break

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            loss_train.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()

            if epoch % 100 == 0:
                tqdm.write(
                    f"Epoch: {epoch}, Training Loss: {round(loss_train.item(), 5)}\n"
                    f"Minimum loss: {min_loss}"
                )

        """ ----------------------- SAVE TRAINING RESULTS AND VALIDATE ----------------------- """

        # save model
        torch.save(model.state_dict(), f"./models/{model_name}.pt")

        num_epochs = list(range(len(train_loss)))

        with PdfPages(f"./training results/{model_name}_results.pdf") as pdf:

            fig_loss = plot_training_loss(
                train_loss_PDE,
                train_loss_BC1,
                train_loss_BC2,
                [],
                [],
                train_loss,
                num_epochs,
                model_name,
                nodata=True,
            )
            pdf.savefig(fig_loss)
            plt.close(fig_loss)

            for i in range(len(X_validation)):

                # run model on the validation set and plot against analytical solution
                X_beam = X_validation[i]
                Y_validation_pred = model(X_beam)

                # get the derivatives
                (
                    DY_validation_pred,
                    D2Y_validation_pred,
                    D3Y_validation_pred,
                    D4Y_validation_pred,
                ) = model.return_derivatives(Y_validation_pred, X_beam)
                # convert to numpy arrays
                Y_validation_pred_np = (
                    Y_validation_pred.detach().to("cpu").numpy()[:, 0]
                )
                DY_validation_pred_np = (
                    DY_validation_pred.detach().to("cpu").numpy()[:, 0]
                )
                D2Y_validation_pred_np = (
                    D2Y_validation_pred.detach().to("cpu").numpy()[:, 0]
                )
                D3Y_validation_pred_np = (
                    D3Y_validation_pred.detach().to("cpu").numpy()[:, 0]
                )
                D4Y_validation_pred_np = (
                    D4Y_validation_pred.detach().to("cpu").numpy()[:, 0]
                )
                X_beam_np = X_beam.detach().to("cpu").numpy()[:, 0]
                load = X_beam.detach().to("cpu").numpy()[:, 1][0]  # assuming a UDL

                # plot the results
                fig = plot_beam_results(
                    Y_validation_pred_np,
                    DY_validation_pred_np,
                    D2Y_validation_pred_np,
                    D3Y_validation_pred_np,
                    D4Y_validation_pred_np,
                    X_beam_np,
                    E * I,
                    L,
                    load,
                )
                pdf.savefig(fig)
                plt.close(fig)

        hyper_end_time = time.time()
        hyper_runtime = hyper_end_time - hyper_start_time
        hyper_minutes = hyper_runtime / 60
        with open(runtime_file, "a") as f:
            f.write(f"Model {model_name} runtime (min): {hyper_minutes}\n")

    total_end_time = time.time()
    total_runtime = total_end_time - total_start_time
    total_minutes = total_runtime / 60
    with open(runtime_file, "a") as f:
        f.write(f"Total runtime (min): {total_minutes}\n")
