import torch
import numpy as np
import scipy.stats as stats

class ImplicitRPM(torch.nn.Module):
    """
    Recognition-Parametrized Model (RPM)
    """
    def __init__(self, rec_models, latent_prior, pxjs):

        super().__init__()
        self.J = len(rec_models)
        assert len(pxjs) == self.J

        self.rec_models = torch.nn.ModuleList(rec_models)
        self.latent_prior = latent_prior
        self.pxjs = pxjs

    def eval(self, xjs):
        assert len(xjs) == self.J
        N = xjs[0].shape[0]
        assert all([xjs[j].shape[0] == N for j in range(self.J)])

        logp0 = sum([pxj.log_prob(xj) for pxj,xj in zip(self.pxjs,xjs)])
        phi, etaz = self.latent_prior.log_partition, self.latent_prior.param

        # reparametrize eta_j <- eta_j + eta_z to ensure \sum_j \eta_j - (J-1)\eta_z is valid natural param...
        eta_off = (1.0 * etaz).reshape(1,-1) 
        logw = phi(sum([m(xj,eta_off) for m,xj in zip(self.rec_models, xjs)]) - (self.J-1) * etaz)
        logw = logw - sum([m.phi(xj,eta_off) for m,xj in zip(self.rec_models,xjs)]) + (self.J-1) * self.latent_prior()

        return logp0.reshape(N) + logw

    def loss_sm(self, xjs):
        
        phi, etaz = self.latent_prior.log_partition, self.latent_prior.param
        eta_off = (1.0 * etaz).reshape(1,-1) 

        # d\etaj / dxj
        jac_js = [torch.func.jacrev(m.forward_sum,argnums=0)(xj,eta_off)[0].transpose(1,0) for m,xj in zip(self.rec_models,xjs)]

        eta_js = [m(xj,eta_off) for m,xj in zip(self.rec_models,xjs)]
        eta_tot = sum(eta_js) - (self.J-1) * etaz

        # terms with gradients of log-partition
        mu_tot = phi.grad(eta_tot)
        mu_js = [m.log_partition.grad(eta_j) for m,eta_j in zip(self.rec_models,eta_js)]

        # terms with Hessian of log-partition
        cov_tot = phi.hessian(eta_tot)
        cov_js = [m.log_partition.hessian(eta_j) for m,eta_j in zip(self.rec_models,eta_js)]

        loss = 0.
        for j in range(self.J):

            dlogp0dx = torch.func.jacrev(self.pxjs[j].log_prob_sum)(xjs[j])
            ddx = ((mu_tot - mu_js[j]) * jac_js[j].squeeze(-1)).sum(axis=-1)
            loss = loss + 0.5 * ddx**2 + dlogp0dx.squeeze(0).squeeze(-1) * ddx

            m = self.rec_models[j]
            jac1_j = torch.func.jacrev(m.forward_sum,argnums=0)
            def jac1_j_sum(x,eta_off):
                return jac1_j(x,eta_off).squeeze(0).squeeze(-1).sum(axis=1)
            jac2_js = torch.func.jacrev(jac1_j_sum,argnums=0)
            tmp = jac2_js(xjs[j],eta_off).squeeze(-1)
            d2etadx2 = torch.stack([tmp[0], tmp[1]],axis=0).transpose(1,0)

            d2dx2 = ((mu_tot - mu_js[j]) * d2etadx2).sum(axis=-1)
            d2dx2 = d2dx2 + (torch.matmul(cov_tot - cov_js[j], jac_js[j]) * jac_js[j]).sum(axis=(1,2))
            loss = loss + d2dx2
        return loss
    
    def training_step_sm(self, batch, batch_idx):
        # score matching loss
        xjs = batch
        loss = self.loss_sm(xjs)
        return loss.sum()

    def eval_sum(self, xjs):
        return self.eval(xjs).sum(axis=0)

    def eval_tensor(self, txjs):
        assert txjs.shape[1] == self.J
        xjs = [txjs[:,j] for j in range(self.J)]
        return self.eval(xjs)

    def forward(self, xjs):
        return [self.rec_models[j](xjs[j])+self.latent_prior.param for j in range(self.J)]


    """
    def training_step(self, batch, batch_idx):
        # Implementation of a still-birth idea thinking that one could
        # enforce an implicit RPM to match the data marginals p(xj) *despite* 
        # the implicit RPM formulation assuming that to hold already.
        # Does not work at all.
        xjs = batch
        J,M = self.J, xjs[0].shape[0]
        phi, etaz = self.latent_prior.log_partition, self.latent_prior.param

        loss = 0
        for j in range(J):
            xj = xjs[j]
            mj = self.rec_models[j]
            xnotj = [xj for xj in xjs]
            xnotj.pop(j)
            mnotj = [mj for mj in self.rec_models]
            mnotj.pop(j)
            def logZj(xj):
                param_all = mj(xj) - (J-1)*etaz + sum([m(x) for m,x in zip(mnotj, xnotj)])
                phinotj = sum([m.phi(x) for m,x in zip(mnotj, xnotj)])
                return torch.logsumexp(phi(param_all) - phinotj,dim=0) - np.log(M)
            phij = (J-1)*self.latent_prior() + logZj(xj)
            loss = loss + torch.mean(mj.phi(xj) - phij)

        return loss
    """

    def training_step(self, batch, batch_idx, lmbda=0.0):
        xjs = batch
        log_probs = self.eval(xjs)
        if lmbda > 0. :
            loss_copula = self.training_step_copula(batch, batch_idx)
        else:
            loss_copula = 0.0

        return - torch.mean(log_probs) + lmbda * loss_copula

    def training_step_copula(self, batch, batch_idx):
        # Greedily match the factor normalizers Fj(Z) to the prior P(Z) via their moments
        xjs = batch
        J,M = self.J, xjs[0].shape[0]
        phi, etaz = self.latent_prior.log_partition, self.latent_prior.param
        eta_off = (1.0 * etaz) # dirty fix for \sum_j \eta_j - (J-1)\eta_z having to be a valid natural param...

        loss_copula = 0
        for j in range(J):
            mj = self.rec_models[j]
            #diff = torch.mean(phi.grad(mj(xjs[j],eta_off)) ,axis=0) - phi.grad(etaz)
            #loss_copula = loss_copula + torch.sum(diff*diff,axis=-1)
            Epx_mu = torch.mean(phi.grad(mj(xjs[j],eta_off)),axis=0).reshape(1,-1)
            mu0 = phi.grad(etaz).reshape(1,-1)
            loss_copula = loss_copula + Dbreg_Gaussian(Epx_mu, mu0)
        return loss_copula

class ImplicitRecognitionFactor_ExpFam(torch.nn.Module):
    """
    Implicit conditional exponential family
    """
    def __init__(self, model, log_partition=None):

        super().__init__()
        self.log_partition = log_partition
        self.model = model # model for natural parameter $\eta_j(\x_j)$

    def phi(self, x, eta_off=0.):

        return self.log_partition(self.forward(x, eta_off))

    def forward(self, x, eta_off=0.):

        return self.model(x) + eta_off

    def forward_sum(self, x, eta_off=0.):

        return self.model(x).sum(axis=0) + eta_off

class ImplicitPrior_ExpFam(torch.nn.Module):
    """
    Implicit exponential family
    """
    def __init__(self, natparam, log_partition=None, activation_out=torch.nn.Identity()):

        super().__init__()
        self.log_partition = log_partition
        self.activation_out = activation_out
        self.param_ = torch.nn.Parameter(natparam)

    def phi(self):

        return self.log_partition(self.param)

    def forward(self):

        return self.phi()

    @property
    def param(self):

        return self.activation_out(self.param_)


class ObservedMarginal(torch.nn.Module):
    def __init__(self, dist):
        super().__init__()
        self.dist = dist

    def log_prob(self, x):
        return self.dist.log_prob(x)

    def log_prob_sum(self, x):
        return self.dist.log_prob(x).sum(axis=0)

    def sample(self,shape):
        return self.dist.sample(shape)


class IndependentMarginal(torch.nn.Module):
    def __init__(self, pxjs, dims):
        super().__init__()
        self.J = len(pxjs)
        self.pxjs = pxjs
        self.dims = dims

    def sample_n(self, n):
        return [pxj.sample((n,dim)) for pxj,dim in zip(self.pxjs, self.dims)]


class GaussianCopula_ExponentialMarginals(torch.nn.Module):
    def __init__(self, P, rates, dims):
        super().__init__()
        self.dims = dims
        self.D = sum(dims)
        self.J = len(dims)
        assert self.J == P.shape[0]
        assert np.all(P == P.T)
        assert np.all(np.diag(P)==1.0)
        self.P = P
        self.logdetP = np.linalg.slogdet(P)
        self.logdetP = self.logdetP[0] * self.logdetP[1]
        self.Pinv = np.linalg.pinv(P)
        self.A = np.linalg.cholesky(P)
        self.rates = rates
        #assert self.J == len(pxjs)
        #self.pxjs = [torch.distributions.exponential.Exponential(rate=rates[j]) for j in range(J)]

    def sample_n(self, n):        
        zz = stats.norm.cdf(self.A.dot(np.random.normal(size=(self.D,n))))
        xx = []
        for j in range(self.J):
            xx.append(torch.tensor((-1.0/self.rates[j])*np.log(zz[sum(self.dims[:j]):sum(self.dims[:j+1]),:]),dtype=torch.float32).T)
        return xx

    def log_probs(self, x):
        rates = np.asarray(self.rates).reshape(1,-1)
        z = stats.norm.ppf(1. - np.exp(-rates * x) )
        p = - 0.5 * self.logdetP - 0.5 * (z * np.dot(z, self.Pinv - np.eye(self.P.shape[0]))).sum(axis=-1)
        p = p + np.log(rates).sum() - np.sum( rates * x, axis=1 )
        return p

    #def sample_independent(self, sample_shape):
    #    return [pxj.sample((n,dim)) for pxj,dim in zip(self.pxjs, self.dims)]


class LogPartition(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        pass
    def grad(self, x):
        pass
    def hessian(self, x):
        pass


class LogPartition_gauss_fixedMean(LogPartition):
    def __init__(self, SigmaInv=torch.eye(1)):
        super().__init__()
        self.D = SigmaInv.shape[0]
        assert all([SigmaInv.shape[j] == self.D for j in range(len(SigmaInv.shape))])
        self.SigmaInv = SigmaInv

    def forward(self, x):
        return 0.5 * torch.sum(x * torch.matmul(x, self.SigmaInv.T), axis=-1)

    def grad(self, x):
        return torch.matmul(x, self.SigmaInv.T)

    def hessian(self, x):
        return self.SigmaInv.T

    
class LogPartition_gauss(LogPartition):
    # - η1^2 /(4η2) − log(−2η2)/2
    def __init__(self, d=None, D=None):
        super().__init__()
        self.d = d
        self.D = D
    def forward(self, x):
        #return - torch.sum(x[:,:D], torch.matmul(x[:,:D], 0.25*torch.inv(x[:,D:].reshape(-1,D,D)),axis=-1) - 0.5 * torch.logdet(-2*x[:,D:].reshape(-1,D,D))
        assert self.d in [None,1]
        return - 0.25 * x[:,0] * x[:,0] / x[:,1] - 0.5 * torch.log(- 2.0* x[:,1])
        
    def grad(self, x):
        assert self.d in [None,1]
        return torch.stack([- 0.5 * x[:,0] / x[:,1], 0.25 * (x[:,0]/x[:,1])**2 - 0.5/x[:,1] ], axis=-1)

    def hessian(self, x):
        assert self.d in [None,1]
        hess = 0.5 * torch.stack([-1./x[:,1], x[:,0]/x[:,1]**2, x[:,0]/x[:,1]**2, 1./x[:,1]**2-(x[:,0]**2/x[:,1]**3)], axis=-1)
        return hess.reshape(*hess.shape[:-1],2,2)

def Dbreg_Gaussian(mu, mu_):
    sig2mu = (mu[:,1] - mu[:,0]**2)
    sig2mu_ = (mu_[:,1] - mu_[:,0]**2)
    grad_mu_ = torch.stack([mu_[:,0] / sig2mu_, - 0.5 / sig2mu_ ], dim=-1)    
    return (torch.sum(grad_mu_ * (mu_ - mu),axis=-1) - 0.5 * torch.log(sig2mu/sig2mu_)).reshape(-1)

