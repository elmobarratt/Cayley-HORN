import math
import torch


# Harmonic Oscillating Recurrent Network model with Cayley Weight Matrices
class CHORN(torch.nn.Module):
    def __init__(self, num_input, num_nodes, num_output, h, alpha, omega, gamma, learning_rule="backprop"):
        super().__init__()

        self.num_input = num_input
        self.num_nodes = num_nodes
        self.num_output = num_output
        self.learning_rule = learning_rule

        # hyperparameters h, alpha, omega, gamma
        self.h = h
        if isinstance(alpha, (int, float)):
            self.alpha = torch.full((self.num_nodes,), float(alpha))
        else:
            self.alpha = torch.as_tensor(alpha)
            # assert self.alpha.shape == (self.num_nodes,)   
        if isinstance(omega, (int, float)):
            self.omega = torch.full((self.num_nodes,), float(omega))
        else:
            self.omega = torch.as_tensor(omega)
            # assert self.omega.shape == (self.num_nodes,) 
        if isinstance(gamma, (int, float)):
            self.gamma = torch.full((self.num_nodes,), float(gamma))
        else:
            self.gamma = torch.as_tensor(gamma)
            # assert self.gamma.shape == (self.num_nodes,) 


        # precompute omega^2 for DHO equation
        self.omega_factor = self.omega * self.omega

        # precompute 2 * gamma for DHO equation
        self.gamma_factor = 2.0 * self.gamma

        # precompute recurrent gain factor
        self.gain_rec = 1. / math.sqrt(self.num_nodes)

        # input, recurrent and output layers
        self.i2h = torch.nn.Linear(num_input, num_nodes)
        self.i2h.bias.data.zero_()
        self.S = torch.nn.Parameter(0.01 * torch.randn(num_nodes, num_nodes))
        self.W = self.recurrent_weight()
        self.readout = PhaseEquivariantReadout(num_nodes, num_output)
        

    def recurrent_weight(self):
        A = self.S - self.S.T
        I = torch.eye(self.num_nodes, device=A.device, dtype=A.dtype)
        # Solve (I + A) W = (I - A)
        W = torch.linalg.solve(I + A, I - A)
        return W
    
    def dynamics_step(self, x_t, y_t, input_t):
        # sympletic Euler integration
        # integrate y_t
        y_t = y_t + self.h * (
            # input on y_t
            self.alpha * torch.tanh(
                self.i2h(input_t) # external input
                + self.gain_rec * (y_t @ self.W.T) # recurrent input from network
            )
            - self.omega_factor * x_t # natural frequency term
            - self.gamma_factor * y_t # damping term
        )

        # integrate x_t with updated y_t
        x_t = x_t + self.h * y_t

        return x_t, y_t

    def hebbian_update(self, lr=1e-4):
    
        # Ensure Hebbian signal exists
        if not hasattr(self, "hebb_signal"):
            raise RuntimeError(
                "hebb_signal not found. "
                "Run a forward pass in training mode before calling hebbian_update()."
            )
    
        with torch.no_grad():
            self.S += lr * self.hebb_signal


    def forward(self, batch, random_init=None, record=False):

        batch_size = batch.size(1)
        num_timesteps = batch.size(0)
    
        ret = {}
    
        if record:
            rec_x_t = torch.zeros(batch_size, num_timesteps, self.num_nodes)
            rec_y_t = torch.zeros(batch_size, num_timesteps, self.num_nodes)
            rec_phi_t = torch.zeros(batch_size, num_timesteps, self.num_nodes*self.num_nodes)
            rec_out_t = torch.zeros(batch_size, num_timesteps, self.num_output)
            ret['rec_x_t'] = rec_x_t
            ret['rec_y_t'] = rec_y_t
            ret['rec_phi_t'] = rec_phi_t
            ret['rec_out_t'] = rec_out_t
    
        # evenly spaced phase on unit circle to initialise dynamical system
        init_theta = torch.linspace(0, 2*torch.pi, self.num_nodes+1)[:-1]
        init_theta = init_theta.repeat(batch_size, 1)
    
        if random_init is not None:
            init_theta = init_theta + torch.randn_like(init_theta) * 0.01
            scale = 1 + torch.randn_like(init_theta) * 0.05
            x_0 = torch.cos(init_theta) * scale
            y_0 = torch.sin(init_theta) * scale
        else:
            x_0 = torch.cos(init_theta)
            y_0 = torch.sin(init_theta)
        

        
        if self.learning_rule == "hebbian":
            # Hebbian accumulating average - (batch, N, N)
            hebb_accumulator = torch.zeros(
                batch_size,
                self.num_nodes,
                self.num_nodes,
                device=batch.device
            )

        x_t = x_0
        y_t = y_0

        self.W = self.recurrent_weight()
            
    
        # evolve dynamical system
        for t in range(num_timesteps):
    
            x_t, y_t = self.dynamics_step(x_t, y_t, batch[t])
    
            if self.learning_rule == "hebbian":
                # change of variables to phase state
                theta = torch.atan2(y_t / self.omega, x_t)
        
                # pairwise phase differences
                # (batch, N, 1) - (batch, 1, N)
                dtheta = theta.unsqueeze(2) - theta.unsqueeze(1)
        
                # accumulate sine synchrony
                hebb_accumulator += torch.sin(dtheta)
    
            if record:
                rec_x_t[:, t, :] = x_t
                rec_y_t[:, t, :] = y_t
                rec_out_t[:, t, :], rec_phi_t[:, t, :] = self.readout(x_t, y_t, self.omega, t)
    
        if self.learning_rule == "hebbian":
            # Time and batch average
            hebb_mean = hebb_accumulator / num_timesteps
            hebb_mean = hebb_mean.mean(dim=0)
        
            # Store detached Hebbian signal for training loop use
            if self.training:
                self.hebb_signal = hebb_mean.detach()
    
        # phase equivariant readout
        output, _ = self.readout(x_t, y_t, self.omega, t)
    
        ret['output'] = output
        return ret

# Phase equivariant readout module
class PhaseEquivariantReadout(torch.nn.Module):
    def __init__(self, num_nodes, num_output):
        super().__init__()

        self.num_nodes = num_nodes

        # number of phase features
        n_phase = num_nodes * num_nodes

        self.readout = torch.nn.Linear(n_phase, num_output)

        # precompute upper-triangular indices (i < j)
        idx = torch.triu_indices(num_nodes, num_nodes, offset=1)
        self.register_buffer("idx_i", idx[0])
        self.register_buffer("idx_j", idx[1])

    def forward(self, x_t, y_t, omega, t):
        """
        x_t, y_t: (batch, num_nodes)
        omega: (num_nodes,) tensor - intrinsic frequencies
        t: scalar - timestep
        """

        # amplitude
        r = torch.sqrt(x_t**2 + (y_t / omega)**2 + 1e-8)

        # phase
        theta = torch.atan2(y_t / omega, x_t)
        # demodulated phase
        theta_tilde = theta - omega * t

        # pairwise differences
        dtheta = (
            theta_tilde[:, self.idx_i]
            - theta_tilde[:, self.idx_j]
        )

        cos_dtheta = torch.cos(dtheta)
        sin_dtheta = torch.sin(dtheta)

        # concatenate features
        phi = torch.cat([r, cos_dtheta, sin_dtheta], dim=1)

        return self.readout(phi),phi

def init_omega(omega0, num_nodes):
    sigma = 0.5 * omega0
    omega = omega0 + sigma * torch.randn(num_nodes)
    return torch.clamp(omega, min=1e-2)
    