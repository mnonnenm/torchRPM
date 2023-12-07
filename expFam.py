import torch
import numpy as np

class ExpFam_base(torch.nn.Module):
    """
    Base class for Exponential family distribution with natural parametrization.
    """
    def __init__(self, natparam, log_partition=None, activation_out=torch.nn.Identity()):
        super().__init__()
        self.log_partition = log_partition
        self.activation_out = activation_out
        self.param_ = natparam if natparam.requires_grad else torch.nn.Parameter(natparam)

    def phi(self, eta=None):
        if eta is None:
            return self.log_partition(self.nat_param)
        else:
            return self.log_partition(eta)

    @property
    def nat_param(self):
        return self.activation_out(self.param_)

    @property
    def mean_param(self):
        return self.log_partition.nat2meanparam(self.nat_param)


class ExpFam(ExpFam_base):
    """
    Exponential family distribution with natural parametrization.
    """
    def __init__(self, natparam, log_partition=None, activation_out=torch.nn.Identity()):
        assert natparam.ndim == 1 or natparam.shape[0] == 1
        natparam = torch.atleast_2d(natparam)
        super().__init__(natparam, log_partition, activation_out)

    def phi(self, eta=None):
        if eta is None:
            return self.log_partition(self.nat_param)[0]
        else:
            return self.log_partition(eta)


class ExpFam_parametrizedParam(ExpFam_base):
    """
    Exponential family distribution with natural parametrization, whose natural parameter
    may be expressed in terms of other parameters.
    """
    def __init__(self, natparam, log_partition=None, activation_out=torch.nn.Identity()):
        super().__init__(natparam(), log_partition, activation_out)
        assert natparam().shape[0] == 1
        self.param_ = natparam

    @property
    def nat_param(self):
        return self.activation_out(self.param_())


class NatParamGauss_RBKernel(torch.nn.Module):
    """
    Natural parameter for Gaussian distribution with covariance
    matrix determined by radial basis function kernel. 
    """
    def __init__(self, mu, gamma, ts):
        super().__init__()
        self.gamma = gamma
        self.ts = ts
        self.mu = mu

    def kernelmat(self, ts, gamma=1.0):
        """ returns kernel matrix for evaluation time points t[i], i=1,..,D
            under radial basis function kernel with scale gamma.
        """
        ts = torch.atleast_2d(ts)
        assert ts.shape[0] == 1

        Sigma = torch.exp( - gamma * (ts.T - ts)**2 )
        return Sigma

    def forward(self, mu=None, ts=None, gamma=None):
        mu = self.mu if mu is None else mu
        ts = self.ts if ts is None else ts
        gamma = self.gamma if gamma is None else gamma

        Sigma = self.kernelmat(ts, gamma)
        Lambda = torch.inverse(Sigma)
        eta0_1 = torch.matmul(mu, Lambda)[...,0,:]
        eta0_2 = -0.5 * Lambda
        eta0 = torch.cat([eta0_1, 
                          eta0_2.reshape(*eta0_2.shape[:-2], np.prod(eta0_2.shape[-2:]))],
                         dim=-1)

        return eta0.unsqueeze(0)


class NatParamGauss_TriDiagonal(torch.nn.Module):
    """
    Natural parameter for Gaussian distribution with covariance
    matrix determined by its tri-diagonal inverse. 
    """
    def __init__(self, mu, diag_val_base, off_val_base):
        super().__init__()
        self.off_val_base = off_val_base
        self.diag_val_base = diag_val_base
        self.mu = mu

    def precisionmat(self, diag_val, off_val):
        """ returns symmetric tri-diagonal precision matrix 
            with specified on-diagonal and off-diagonal elements.
        """
        T = torch.numel(self.mu)
        Lambda = torch.diag(diag_val*torch.ones(T), 0) 
        Lambda += torch.diag(off_val*torch.ones(T-1), 1) 
        Lambda += torch.diag(off_val*torch.ones(T-1), -1)
        return Lambda

    def forward(self, mu=None, diag_val=None, off_val=None):
        mu = self.mu if mu is None else mu
        diag_val = self.diag_val if diag_val is None else diag_val
        off_val = self.off_val if off_val is None else off_val

        Lambda = self.precisionmat(diag_val, off_val)
        eta0_1 = torch.matmul(mu, Lambda)[...,0,:]
        eta0_2 = -0.5 * Lambda
        eta0 = torch.cat([eta0_1, 
                          eta0_2.reshape(*eta0_2.shape[:-2], np.prod(eta0_2.shape[-2:]))],
                         dim=-1)
        return eta0.unsqueeze(0)

    @property
    def diag_val(self):
        return torch.exp(self.diag_val_base)        

    @property
    def off_val(self):
        return self.diag_val/2. * torch.tanh(self.off_val_base)        

class SemiparametricConditionalExpFam(ExpFam_base):
    """
    Saturated conditional exponential family distribution with one set of 
    (natural) parameters per possible value of the conditioned-on variable.
    Meant as learnable posterior approximation in variational inference. 
    """
    def __init__(self, natparams, log_partition=None, activation_out=torch.nn.Identity()):
        assert natparams.ndim >= 2
        super().__init__(natparams, log_partition, activation_out)
        self.N = self.param_.shape[0]

    def phi(self, eta=None, nat_param_offset=0.):
        if eta is None:
            return self.log_partition(self.nat_param(nat_param_offset))
        else:
            return self.log_partition(eta + nat_param_offset)

    def nat_param(self, nat_param_offset=0.):
        return self.activation_out(self.param_) + nat_param_offset

    def mean_param(self, nat_param_offset=0.):
        return self.log_partition.nat2meanparam(self.nat_param(nat_param_offset))


class ConditionalExpFam(torch.nn.Module):
    """
    Conditional exponential family distribution, 
    i.e. natural parameters η(x) are data-dependent.
    """
    def __init__(self, model, log_partition=None):

        super().__init__()
        self.log_partition = log_partition
        self.model = model # model for natural parameter $\eta_j(\x_j)$

    def phi_x(self, x, eta_off=0.):
        return self.log_partition(self(x, eta_off))

    def nat_param(self, x, eta_off=0.):
        return self.model(x) + eta_off

    def mean_param(self, x, eta_off=0.):
        return self.log_partition.nat2meanparam(self.nat_param(x, eta_off=0.))

    def forward(self, x, eta_off=0.):
        return self.nat_param(x, eta_off=eta_off)

    def forward_sum(self, x, eta_off=0.):
        return self.model(x).sum(axis=0) + eta_off


#################################################################################
# Log-partition functions for some exponential families
#################################################################################

class LogPartition(torch.nn.Module):
    """
    Base class for log-partition functions in natural parametrization. 
    """
    def __init__(self):
        super().__init__()
    def forward(self, eta):
        pass
    def grad(self, eta):
        pass
    def hessian(self, eta):
        pass
    def nat2meanparam(self,eta):
        return self.grad(eta)


class LogPartition_vonMises(LogPartition):
    """
    Log-partition functions in natural parametrization for vonMises distribution. 
    """
    def __init__(self,d=1):
        super().__init__()
        self.d = d # number of (independent!) von Mises-distributed variables

    def forward(self, eta):
        r = torch.sqrt((eta.reshape(-1,self.d,2)**2).sum(axis=-1)).reshape(-1,self.d)
        return torch.log(torch.i0(r)).sum(axis=-1)

    def grad(self, eta):
        r = torch.sqrt((eta.reshape(-1,self.d,2)**2).sum(axis=-1)).reshape(-1,self.d,1)
        mu =  eta.reshape(-1,self.d,2) * (torch.special.i1(r)/torch.i0(r)/(r+1.*(r==0.)))
        return mu.reshape(-1,2*self.d)

    def hessian(self, eta):
        #r = torch.sqrt((eta.reshape(-1,self.k,2)**2).sum(axis=-1)).reshape(-1,self.k)
        #I10 = torch.special.i1(r) / torch.special.i0(r)
        #I20 = torch.special.i2(r) / torch.special.i0(r)
        #H = ((1 + I20) / (2 * r) - I10 / r**2 - I10^2 / r) / r * np.outer(eta,eta)  + (I10 / r) * I
        raise NotImplementedError()


class LogPartition_gauss_fixedMean(LogPartition):
    """
    Log-partition functions in natural parametrization for normal with fixed (co-)variance. 
    """
    def __init__(self, SigmaInv=torch.eye(1)):
        super().__init__()
        self.D = SigmaInv.shape[0]
        assert all([SigmaInv.shape[j] == self.D for j in range(len(SigmaInv.shape))])
        self.SigmaInv = SigmaInv

    def forward(self, eta):
        return 0.5 * torch.sum(eta * torch.matmul(eta, self.SigmaInv.T), axis=-1)

    def grad(self, eta):
        return torch.matmul(eta, self.SigmaInv.T)

    def hessian(self, eta):
        return self.SigmaInv.T


class LogPartition_gauss1D(LogPartition):
    """
    Log-partition functions in natural parametrization for univariate normal distribution. 
    phi(η) = - η1^2 /(4η2) − log(−2η2)/2
    """    
    def __init__(self):
        super().__init__()

    def forward(self, eta):
        #return - torch.sum(x[:,:D], torch.matmul(x[:,:D], 0.25*torch.inv(x[:,D:].reshape(-1,D,D)),axis=-1) - 0.5 * torch.logdet(-2*x[:,D:].reshape(-1,D,D))
        eta1, eta2 = eta[...,0], eta[...,1]
        return - 0.25 * eta1**2 / eta2 - 0.5 * torch.log(- 2.0* eta2)

    def grad(self, eta):
        eta1, eta2 = eta[...,0], eta[...,1]
        return torch.stack([- 0.5 * eta1 / eta2, 0.25 * (eta1/eta2)**2 - 0.5/eta2 ], axis=-1)

    def hessian(self, eta):
        eta1, eta2 = eta[...,0], eta[...,1]
        hess = 0.5 * torch.stack([-1./eta2, eta1/eta2**2, 
                                  eta1/eta2**2, 1./eta2**2-(eta1**2/eta2**3)], 
                                 axis=-1)
        return hess.reshape(*hess.shape[:-1],2,2)


class LogPartition_gauss_diagonal(LogPartition):
    """
    Log-partition functions in natural parametrization for factorising multivariate normal distribution. 
    phi(η) = - sum_d η1[d]^2 /(4 η2[d,d]) − log(−2 η2[d,d])/2
    Assumes scaled precision matrix η2 to be diagonal and η2 = eta[...,1,:] is only the diagonal elements.
    """
    def __init__(self, d=None):
        super().__init__()
        assert (d is None) or (d > 0)
        self.d = d
        self.marginal_log_partition = LogPartition_gauss1D()

    def forward(self, eta):
        #return - torch.sum(x[:,:D], torch.matmul(x[:,:D], 0.25*torch.inv(x[:,D:].reshape(-1,D,D)),axis=-1) - 0.5 * torch.logdet(-2*x[:,D:].reshape(-1,D,D))
        assert eta.shape[-1] == 2*self.d
        eta1, eta2 = eta[...,:self.d], eta[...,self.d:]
        return (- 0.25 * eta1**2 / eta2 - 0.5 * torch.log(- 2.0* eta2)).sum(axis=-1)

    def grad(self, eta):
        assert eta.shape[-1] == 2*self.d
        eta1, eta2 = eta[...,:self.d], eta[...,self.d:]
        return torch.cat([- 0.5 * eta1 / eta2, 0.25 * (eta1/eta2)**2 - 0.5/eta2 ], axis=-1)

    def hessian(self, eta):
        assert eta.shape[-1] == 2*self.d
        eta1, eta2 = eta[...,:self.d], eta[...,self.d:]
        hess = 0.5 * torch.cat([torch.cat([torch.diag_embed(-1./eta2), 
                                           torch.diag_embed(eta1/eta2**2)], 
                                          axis=-1),
                                torch.cat([torch.diag_embed(eta1/eta2**2), 
                                           torch.diag_embed(1./eta2**2-(eta1**2/eta2**3))], 
                                          axis=-1)], 
                               axis=-2)
        return hess


class LogPartition_gauss_diagonal_fullparam(LogPartition_gauss_diagonal):
    """
    Log-partition functions in natural parametrization for factorising multivariate normal distribution. 
    phi(η) = - sum_d η1[d]^2 /(4 η2[d,d]) − log(−2 η2[d,d])/2
    Assumes scaled precision matrix η2 to be diagonal and extracts the diagonal from matrix-sized η2.
    """    
    def forward(self, eta):
        return super().forward(self.extract_diagonal(eta))

    def grad(self, eta):
        return super().grad(self.extract_diagonal(eta))

    def hessian(self, eta):
        return super().hessian(self.extract_diagonal(eta))

    def extract_diagonal(self,eta):
        d = self.d
        assert eta.shape[-1] == (d+1) * d
        eta1, eta2 = eta[...,:d], torch.diagonal(eta[...,d:].reshape(*eta.shape[:-1],d,d),dim1=-2,dim2=-1)
        return torch.cat([eta1, eta2], dim=-1) 

    def embed_diagonal(self, eta):
        d = self.d
        assert eta.shape[-2] == d
        assert eta.shape[-1] == 2
        eta1, eta2 = eta[...,0], torch.diag_embed(eta[...,1],dim1=-2,dim2=-1).reshape(*eta.shape[:-2],d**2)
        return torch.cat([eta1, eta2], dim=-1) 

    def decorrelate_natparam(self, eta):
        d = self.d
        eta1, eta2 = eta[...,:d], eta[...,d:].reshape(*eta.shape[:-1],d,d)
        cov = torch.inverse(-2 * eta2)
        mean = torch.bmm(eta1.unsqueeze(-2), cov).squeeze(-2)
        var = torch.diagonal(cov, dim1=-2, dim2=-1)
        return torch.cat([mean/var, -0.5 * torch.diag_embed(1./var).reshape(*eta2.shape[:-2], d**2)], dim=-1)

    def full2diag_gaussian(self, eta):
        eta_uncorr = self.decorrelate_natparam(eta)
        eta_diag = self.extract_diagonal(eta_uncorr)
        return eta_diag.reshape(*eta_diag.shape[:-1], 2,self.d).transpose(-1,-2), eta_uncorr

    def diag2full_gaussian(self, eta_diag):
        eta = self.embed_diagonal(eta_diag)
        return eta

class LogPartition_gauss(LogPartition):
    """
    Log-partition functions in natural parametrization for  multivariate normal distribution. 
    phi(η) = - 1/4 η1' η2^-1 η1 − log(|−2 η2|)/2
    Only implements the log-partition evaluation and mean2parameter transform, no gradients, no hessians !
    """
    def __init__(self, d=None):
        super().__init__()
        assert (d is None) or (d > 0)
        self.d = d
        self.marginal_log_partition = LogPartition_gauss1D()

    def forward(self, eta):
        assert eta.shape[-1] == (self.d+1) * self.d
        d = self.d
        eta1, eta2 = eta[...,:d], eta[...,d:].reshape(*eta.shape[:-1],d,d)
        phi = - 0.25* torch.sum(eta1 * torch.linalg.solve(eta2, eta1), axis=-1)
        phi -=  0.5 * torch.logdet(-2*eta2)
        return phi 

    def grad(self, eta):
        assert eta.shape[-1] == (self.d+1) * self.d
        d = self.d
        eta1, eta2 = eta[...,:d], eta[...,d:].reshape(*eta.shape[:-1],d,d)
        Sig = torch.inverse(-2 * eta2)
        firstMoment = torch.bmm(eta1.unsqueeze(-2), Sig).squeeze(-2)
        secondMoment = (Sig + firstMoment.unsqueeze(-2)*firstMoment.unsqueeze(-1))
        return torch.cat([firstMoment, secondMoment.reshape(*Sig.shape[:-2], d**2)], dim=-1)

    def hessian(self, eta):
        raise NotImplementedError()

    def extract_diagonal(self,eta):
        d = self.d
        assert eta.shape[-1] == (d+1) * d
        eta1, eta2 = eta[...,:d], torch.diagonal(eta[...,d:].reshape(*eta.shape[:-1],d,d),dim1=-2,dim2=-1)
        return torch.cat([eta1, eta2], dim=-1) 

    def embed_diagonal(self, eta):
        d = self.d
        assert eta.shape[-2] == d
        assert eta.shape[-1] == 2
        eta1, eta2 = eta[...,0], torch.diag_embed(eta[...,1],dim1=-2,dim2=-1).reshape(*eta.shape[:-2],d**2)
        return torch.cat([eta1, eta2], dim=-1) 

    def decorrelate_natparam(self, eta):
        d = self.d
        eta1, eta2 = eta[...,:d], eta[...,d:].reshape(*eta.shape[:-1],d,d)
        cov = torch.inverse(-2 * eta2)
        mean = torch.bmm(eta1.unsqueeze(-2), cov).squeeze(-2)
        var = torch.diagonal(cov, dim1=-2, dim2=-1)
        return torch.cat([mean/var, -0.5 * torch.diag_embed(1./var).reshape(*eta2.shape[:-2], d**2)], dim=-1)

    def full2diag_gaussian(self, eta):
        eta_uncorr = self.decorrelate_natparam(eta)
        eta_diag = self.extract_diagonal(eta_uncorr)
        return eta_diag.reshape(*eta_diag.shape[:-1], 2,self.d).transpose(-1,-2), eta_uncorr

    def diag2full_gaussian(self, eta_diag):
        eta = self.embed_diagonal(eta_diag)
        return eta


def Dbreg_Gaussian(mu, mu_):
    """
    Bregman Divergence between two Gaussian distributions with variable covariances
    in mean parametrizations (i.e. mu's give first and second moment).
    """
    sig2mu = (mu[:,1] - mu[:,0]**2)
    sig2mu_ = (mu_[:,1] - mu_[:,0]**2)
    grad_mu_ = torch.stack([mu_[:,0] / sig2mu_, - 0.5 / sig2mu_ ], dim=-1)    
    return (torch.sum(grad_mu_ * (mu_ - mu),axis=-1) - 0.5 * torch.log(sig2mu/sig2mu_)).reshape(-1)


class LogPartition_discrete(LogPartition):
    """
    Log-partition function for categorical distribution with D possible state values.
    """
    def __init__(self, D=None):
        super().__init__()
        self.D = D # number of possible states
    def forward(self, eta):
        #return phi(eta) = 0. ExpFam is normalized by its parameters being on the D-simplex!
        return torch.zeros(*eta.shape[:-1], dtype=eta.dtype)
        
    def grad(self, eta):
        return torch.zeros(size=(*eta.shape[:-1], self.D),dtype=eta.dtype)

    def hessian(self, eta):
        return torch.zeros(size=(*eta.shape[:-1], self.D, self.D),dtype=eta.dtype)

    def nat2meanparam(self,eta):
        return  torch.nn.Softmax(dim=-1)(eta) # = exp(eta) for valid eta - softmax just to be safe
