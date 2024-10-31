import torch
import torchode
from util_functions import create_matrix_from_arange
from constants import Model_Config


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ODEFunc(torch.nn.Module):
    def __init__(self, input_dim):
        super(ODEFunc, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, input_dim)
        )

    def forward(self, t, x):
        return self.net(x)


class ODERNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ODERNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn_cell = torch.nn.GRU(input_dim, hidden_dim, batch_first=True)  # RNN-like update
        
        self.ode_func = ODEFunc(hidden_dim)  # ODE dynamics
        self.term = torchode.ODETerm(self.ode_func)
        self.step_method = torchode.Dopri5(term=self.term)
        self.step_size_controller = torchode.IntegralController(atol=1e-3, rtol=1e-3, term=self.term)
        self.adjoint = torchode.AutoDiffAdjoint(self.step_method, self.step_size_controller)
        
        self.fc =  torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim) 
        )  
    
    def forward(self, x, t):
        batch_size = x.size(0)
        h = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)
        
        for i in range(t.size(1) - 1):

            out, h = self.rnn_cell(x[:, i, :].reshape(batch_size, 1, x.size(-1)), h)

            time_divs = create_matrix_from_arange(t[:, i], t[:, i+1], n = Model_Config.TIME_DIVS.value)


            problem = torchode.InitialValueProblem(y0=h.squeeze(0).to(device), t_eval=time_divs.to(device))
            sol = self.adjoint.solve(problem).ys.permute(1, 0, 2)[-1]
            h = sol.unsqueeze(0).contiguous()


        # Output the final prediction
        return self.fc(out)
