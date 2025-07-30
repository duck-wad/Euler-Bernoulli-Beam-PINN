import torch
import torch.nn as nn
import torchvision.models as models

from matplotlib.backends.backend_pdf import PdfPages

from helper import *
from pinn_class import BeamPINN

# configure the device for GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    L = 1.0
    E = 1.0
    I = 1.0
    max_load = 100
    num_points = 101
    num_input = 2
    num_output = 1

    model_folder = "./models/"
    model_name = "256_2_0.0001_0.0001_0.01_1.0_1.0_50000.pt"

    model_path = model_folder + model_name
    temp = model_name.split("_")
    num_neurons = int(temp[0])
    num_layers = int(temp[1])

    os.makedirs("./testing results", exist_ok=True)

    X_test, _, _ = prepare_linspace(L, num_points, 1, max_load, device)

    model = BeamPINN(num_input, num_output, num_neurons, num_layers).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with PdfPages(f"./testing results/{model_name}_testresults.pdf") as pdf:
        for i in range(len(X_test)):
            X_beam = X_test[i]
            Y_test_pred = model(X_beam)

            (
                DY_test_pred,
                D2Y_test_pred,
                D3Y_test_pred,
                D4Y_test_pred,
            ) = model.return_derivatives(Y_test_pred, X_beam)
            # convert to numpy arrays
            Y_test_pred_np = Y_test_pred.detach().to("cpu").numpy()[:, 0]
            DY_test_pred_np = DY_test_pred.detach().to("cpu").numpy()[:, 0]
            D2Y_test_pred_np = D2Y_test_pred.detach().to("cpu").numpy()[:, 0]
            D3Y_test_pred_np = D3Y_test_pred.detach().to("cpu").numpy()[:, 0]
            D4Y_test_pred_np = D4Y_test_pred.detach().to("cpu").numpy()[:, 0]
            X_beam_np = X_beam.detach().to("cpu").numpy()[:, 0]
            load = X_beam.detach().to("cpu").numpy()[:, 1][0]  # assuming a UDL

            # plot the results
            fig = plot_beam_results(
                Y_test_pred_np,
                DY_test_pred_np,
                D2Y_test_pred_np,
                D3Y_test_pred_np,
                D4Y_test_pred_np,
                X_beam_np,
                E * I,
                L,
                load,
            )
            pdf.savefig(fig)
            plt.close(fig)
