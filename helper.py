import numpy as np
import pandas as pd
import zipfile
import os
import torch
import matplotlib.pyplot as plt

# if beam is discretized into n points, the network will expect n+1 inputs
# 1 input is the x location on the beam
# n inputs is the w(x) load at each discretized point on the beam
# so the NN will learn to output a y(x), given a location on the beam x under the load w(x)


def prepare_data(device):

    with zipfile.ZipFile("./data.zip", "r") as zip_ref:
        zip_ref.extractall("./data")
    path = "./data"
    files = os.listdir(path)
    files = [f for f in files if f[-3:] == "csv"]

    # location on beam
    x = []
    # distributed load magnitude
    w = []
    # corresponding displacement
    y = []
    # corresponding rotation
    theta = []

    for i, file in enumerate(files):
        df = pd.read_csv(path + "/" + file)

        x_temp = df["Node location (m)"].to_numpy()
        x.append(x_temp)

        y_temp = df["Nodal transverse displacement (m)"].to_numpy()
        y.append(y_temp)

        theta_temp = df["Nodal rotation (rad)"].to_numpy()
        theta.append(theta_temp)

        # in the future if w(x) is not a UDL, will need to handle this differently
        load = float(file.split(".csv")[0])
        w.append(load)

    x = np.array(x)  # n_files x n_points
    x = np.expand_dims(x, axis=2)
    y = np.array(y)  # n_files x n_points
    theta = np.array(theta)
    # w is not sorted because of sorting standard in file explorer, sort by ascending
    w = np.sort(np.array(w))  # n_files
    # put w into n_files x n_points x n_points
    # for each x point, it corresponds with the full w(x) over the beam
    w = w[:, np.newaxis]
    w = np.repeat(w, len(x[0]), axis=1)
    w = w[:, :, np.newaxis]
    w = np.repeat(w, len(x[0]), axis=2)
    # multiply by -1 to put w(x) in the downward direction
    w = w * -1

    # stack x and w into one input
    X_all = np.concatenate((x, w), axis=2)

    # make y 3D array
    Y_all = y[..., np.newaxis]
    DYDX_all = theta[..., np.newaxis]

    # pull random sample of 10% for testing and 20% for validation
    np.random.seed(42)
    num_samples = len(X_all)
    # shuffle indices and slice through the shuffled list to get the indices for testing, validation, training
    all_indices = np.random.permutation(num_samples)
    num_test = round(num_samples * 0.1)
    num_validation = round(num_samples * 0.2)
    num_train = num_samples - num_test - num_validation
    test_index = all_indices[:num_test]
    validation_index = all_indices[num_test : num_test + num_validation]
    train_index = all_indices[num_test + num_validation :]

    X_train = X_all[train_index]
    Y_train = Y_all[train_index]
    DYDX_train = DYDX_all[train_index]
    X_validation = X_all[validation_index]
    Y_validation = Y_all[validation_index]
    DYDX_validation = DYDX_all[validation_index]
    X_test = X_all[test_index]
    Y_test = Y_all[test_index]
    DYDX_test = DYDX_all[validation_index]

    # convert numpy arrays to torch tensors
    X_train = torch.tensor(X_train, requires_grad=True, dtype=torch.float32).to(device)
    Y_train = torch.tensor(Y_train, requires_grad=True, dtype=torch.float32).to(device)
    DYDX_train = torch.tensor(DYDX_train, requires_grad=True, dtype=torch.float32).to(
        device
    )
    X_validation = torch.tensor(
        X_validation, requires_grad=True, dtype=torch.float32
    ).to(device)
    Y_validation = torch.tensor(
        Y_validation, requires_grad=True, dtype=torch.float32
    ).to(device)
    X_validation = torch.tensor(
        X_validation, requires_grad=True, dtype=torch.float32
    ).to(device)
    DYDX_validation = torch.tensor(
        DYDX_validation, requires_grad=True, dtype=torch.float32
    ).to(device)
    X_test = torch.tensor(X_test, requires_grad=True, dtype=torch.float32).to(device)
    Y_test = torch.tensor(Y_test, requires_grad=True, dtype=torch.float32).to(device)
    DYDX_test = torch.tensor(DYDX_test, requires_grad=True, dtype=torch.float32).to(
        device
    )

    return (
        X_train,
        Y_train,
        DYDX_train,
        X_validation,
        Y_validation,
        DYDX_validation,
        X_test,
        Y_test,
        DYDX_test,
    )


def prepare_linspace(length, num_points, min_load, max_load, device):

    x = np.linspace(0, length, num_points, endpoint=True)
    x = np.tile(x, (max_load - min_load + 1, 1))
    x = x[:, :, np.newaxis]

    w = np.linspace(min_load, max_load, (max_load - min_load) + 1, endpoint=True)[
        :, np.newaxis, np.newaxis
    ]
    w = np.repeat(w, num_points, axis=1)
    w = w * -1

    X_all = np.concatenate((x, w), axis=2)

    # split into training, testing and validation
    np.random.seed(42)
    num_samples = len(X_all)
    all_indices = np.random.permutation(num_samples)

    num_test = round(num_samples * 0.1)
    num_validation = round(num_samples * 0.2)

    test_index = all_indices[:num_test]
    validation_index = all_indices[num_test : num_test + num_validation]
    train_index = all_indices[num_test + num_validation :]

    X_train = X_all[train_index]
    X_validation = X_all[validation_index]
    X_test = X_all[test_index]

    X_train = torch.tensor(X_train, requires_grad=True, dtype=torch.float32).to(device)
    X_validation = torch.tensor(
        X_validation, requires_grad=True, dtype=torch.float32
    ).to(device)
    X_test = torch.tensor(X_test, requires_grad=True, dtype=torch.float32).to(device)

    return X_train, X_validation, X_test


def separate_domain_bc(X, Y, L, nodata=False):
    x_vals = X[:, 0].unsqueeze(1)
    is_left = torch.abs(x_vals - 0.0) < 1e-5
    is_right = torch.abs(x_vals - L) < 1e-5
    is_bc = is_left | is_right
    is_domain = ~is_bc

    X_domain = X[is_domain.squeeze()].detach().clone().requires_grad_()
    X_bc = X[is_bc.squeeze()].detach().clone().requires_grad_()
    # for the nodata model, return dummy Y values
    Y_domain = []
    Y_bc = []
    if not nodata:
        Y_domain = Y[is_domain.squeeze()].detach().clone().requires_grad_()
        Y_bc = Y[is_bc.squeeze()].detach().clone().requires_grad_()

    return X_domain, X_bc, Y_domain, Y_bc


def plot_training_loss(
    loss_PDE,
    loss_BC1,
    loss_BC2,
    loss_displacement,
    loss_slope,
    loss_total,
    num_epochs,
    name,
    nodata=False,
):

    fig = plt.figure(figsize=(10, 6))

    plt.plot(
        num_epochs, loss_PDE, label=r"Training PDE Loss ($EI\frac{d^4y}{dx^4} = w$)"
    )
    plt.plot(num_epochs, loss_BC1, label=r"Training BC1 Loss ($y=0$ for $x=0,L$)")
    plt.plot(
        num_epochs,
        loss_BC2,
        label=r"Training BC2 Loss ($EI\frac{d^2y}{dx^2}=0$ for $x=0,L$)",
    )
    if not nodata:
        plt.plot(num_epochs, loss_displacement, label="Training Displacement Data Loss")
        plt.plot(num_epochs, loss_slope, label="Training Slope Data Loss")
    plt.plot(
        num_epochs, loss_total, label="Total Training Loss", linewidth=2, color="black"
    )

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Loss vs Epochs")
    plt.legend(loc="upper right")
    plt.tight_layout()

    return fig


def analytical_solution(X, EI, L, w):
    anal_Y = w * X / (24.0 * EI) * (L**3 - 2 * L * X**2 + X**3)
    anal_DY = w / (24.0 * EI) * (L**3 - 6 * L * X**2 + 4 * X**3)
    anal_M = -w * X / 2.0 * (L - X)
    anal_V = -w * (L / 2.0 - X)
    anal_W = np.full_like(X, w)
    return anal_Y, anal_DY, anal_M, anal_V, anal_W


def plot_beam_results(Y, DY, D2Y, D3Y, D4Y, X, EI, L, w):

    anal_Y, anal_DY, anal_M, anal_V, anal_W = analytical_solution(X, EI, L, w)

    fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

    axs[0].plot(X, Y, label="Predicted", color="blue")
    axs[0].plot(X, anal_Y, label="Analytical", linestyle="--", color="black")
    axs[0].set_ylabel("Deflection (m)")
    axs[0].legend()

    axs[1].plot(X, DY, label="Predicted", color="green")
    axs[1].plot(X, anal_DY, label="Analytical", linestyle="--", color="black")
    axs[1].set_ylabel("Slope (rad)")
    axs[1].legend()

    axs[2].plot(X, D2Y * EI, label="Predicted", color="orange")
    axs[2].plot(X, anal_M, label="Analytical", linestyle="--", color="black")
    axs[2].set_ylabel("Moment (N*m)")
    axs[2].legend()

    axs[3].plot(X, D3Y * EI, label="Predicted", color="red")
    axs[3].plot(X, anal_V, label="Analytical", linestyle="--", color="black")
    axs[3].set_ylabel("Shear (N)")
    axs[3].legend()

    # the fourth deriv may appear noisy since W is a constant
    # set a range for the y-axis
    axs[4].plot(X, D4Y * EI, label="Predicted", color="purple")
    axs[4].plot(X, anal_W, label="Analytical", linestyle="--", color="black")
    axs[4].set_ylabel("Load (N/m)")
    axs[4].set_ylim(anal_W[0] + 5, anal_W[0] - 5)
    axs[4].legend()
    axs[4].set_xlabel("Beam position (m)")

    for ax in axs:
        ax.grid(True)

    plt.suptitle(f"Validation results under a {abs(w)} N/m load")
    plt.tight_layout()

    return fig
