import torch
import torch.nn as nn

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

        self.loss_function = nn.MSELoss(reduction = 'mean')

        nn.init.xavier_normal_(self.input_layer[0].weight.data,gain=1.0)
        nn.init.zeros_(self.input_layer[0].bias.data)
        nn.init.xavier_normal_(self.output_layer.weight.data,gain=1.0)
        nn.init.zeros_(self.output_layer.bias.data)

        for i in range(num_layers-1):
            layer = self.hidden_layer[i][0]
            nn.init.xavier_normal_(layer.weight.data,gain=1.0)
            nn.init.zeros_(layer.bias.data)

    def forward(self, X_batch):
        # torch.cat expects arrays of (N, 1), unsqueeze to get in that form
        temp = self.input_layer(X_batch)
        temp = self.hidden_layer(temp)
        temp = self.output_layer(temp)
        return temp
    
    def PDE_loss(self, Y_domain, X_domain, E, I):

        # need to do derivatives wrt the original X_domain tensor rather than splicing X_domain to get the 
        # x position tensor, that creates a new tensor and grad won't work
        dy = torch.autograd.grad(Y_domain, X_domain, torch.ones_like(Y_domain), retain_graph=True, create_graph=True)[0]
        d2y = torch.autograd.grad(dy, X_domain, torch.ones_like(dy), retain_graph=True, create_graph=True)[0]
        d3y = torch.autograd.grad(d2y, X_domain, torch.ones_like(d2y), retain_graph=True, create_graph=True)[0]
        d4y = torch.autograd.grad(d3y, X_domain, torch.ones_like(d3y), create_graph=True)[0]

        d4y_dx4 = d4y[:,0].unsqueeze(1)

        # loss is mean squared error
        f = E*I*d4y_dx4
        f_hat = X_domain[:,1].unsqueeze(1)
        return self.loss_function(f, f_hat)

    def boundary_loss(self, Y_bc, X_bc, E, I):
        # handle when Y_bc is empty
        if Y_bc.numel() == 0:
            return torch.tensor(0.0, device=Y_bc.device), torch.tensor(0.0, device=Y_bc.device)
        # for simply supported beam, y=0 and d2ydx2=0
        dy = torch.autograd.grad(Y_bc, X_bc, torch.ones_like(Y_bc), retain_graph=True, create_graph=True)[0]
        d2y = torch.autograd.grad(dy, X_bc, torch.ones_like(dy), create_graph=True)[0]

        d2y_dx2 = d2y[:,0].unsqueeze(1)
        return (self.loss_function(Y_bc, torch.zeros_like(Y_bc)), self.loss_function((E*I*d2y_dx2), torch.zeros_like(d2y_dx2)))
        
    def data_loss(self, Y_pred, Y_true):
        return self.loss_function(Y_pred, Y_true)