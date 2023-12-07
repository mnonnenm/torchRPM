import numpy as np
import torch

def activation_gauss_diag_diag_conv1d(x,d): 
    """ returns natural parameters for d-dimensional Gaussian distribution, that is [Sig^-1*m, -1/2 Sig^-1].
    Assumes that only the diagonal of the covariance/precision matrix is needed. Requires dimensionality d.  
    Assumes x.shape[-2] = d*2 for means and variances, respectively, such as results from Conv1D layers.
    """
    return torch.transpose(torch.cat([x[...,:d,:], -torch.nn.Softplus()(x[...,d:,:])],axis=-2), -2, -1)

def activation_inv_gauss_diag_diag_conv1d(eta,d):
    """ inverse function for activation function for d-dimensional Gaussian distribution with diagonal covariance
    represented by its diagonal only. Returns pre-activations given natural parameters. Assumes x.shape[-2]=2*d. 
    """
    assert eta.shape[-1]/2. == d 
    x__d = eta[...,0].unsqueeze(-2) 
    x_d_ = - eta[...,1].unsqueeze(-2)
    return torch.cat([x__d, torch.log((torch.exp(x_d_)-1.).clamp_min(1e-6))], axis=-2)

def activation_gauss_diag_full_conv1d(x,d): 
    """ returns natural parameters for d-dimensional Gaussian distribution, that is [Sig^-1*m, -1/2 Sig^-1]
    Assumes that full covariance matrix Sigma is diagonal, i.e. the output of this function only computes
    non-negative variances and embeds them into a diagonal matrix. Requires dimensionality d.  
    Assumes x.shape[-2] = d*2 for means and variances, respectively, such as results from Conv1D layers.
    """
    mean = x[...,:d,:].reshape(*x.shape[:-2], -1)
    cov = torch.diag_embed(-torch.nn.Softplus()(x[...,d:,:].reshape(*x.shape[:-2], -1))).reshape(*x.shape[:-2], -1)
    return torch.cat([mean, cov],axis=-1)

def activation_inv_gauss_diag_full_conv1d(eta,d):
    """ inverse function for activation function for d-dimensional Gaussian distribution with diagonal covariance
    represented as full diagonal matrix. Returns pre-activations given natural parameters. Assumes x.shape[-2]=2*d. 
    """
    T = int((np.sqrt(4*eta.shape[-1] + 1) - 1)/(2*d))
    assert (d*T)*(d*T+1) == eta.shape[-1]
    x__d = eta[...,:d*T].reshape(*eta.shape[:-1],d,T) 
    x_d_ = -torch.diagonal(eta[...,d*T:].reshape(*eta.shape[:-1],d*T,d*T),dim1=-2,dim2=-1).reshape(*eta.shape[:-1],d,T)
    return torch.cat([x__d, torch.log((torch.exp(x_d_)-1.).clamp_min(1e-6))], axis=-2)


class Net_3xConv1D(torch.nn.Module):
    """ 3-layer feed-forward network with 1D convolutions, ReLU activations and variable output nolinearity"""    
    def __init__(self, n_in, n_out, n_hidden, activation_out=torch.nn.Identity()):
        super(Net_3xConv1D, self).__init__()
        self.activation_out = activation_out
        self.conv1 = torch.nn.Conv1d(n_in, n_hidden, kernel_size=1, bias=True)
        self.conv2 = torch.nn.Conv1d(n_hidden, n_hidden, kernel_size=1, bias=True)
        self.conv3 = torch.nn.Conv1d(n_hidden, n_out, kernel_size=1, bias=True)

    def forward(self, x):
        x = x.unsqueeze(-2) if x.ndim == 2 else x
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.conv3(x)
        return self.activation_out(x)