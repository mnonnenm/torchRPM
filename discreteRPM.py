import torch
import numpy as np


class discreteRPM_softmaxForm(torch.nn.Module):
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

    def log_probs(self, idx_n):

        xjis = [self.pxjs[j].x for j in range(self.J)]                                 # N - D
        #gji = [m.affine_all_z(xj) for m,xj in zip(self.rec_models, xjis)]             # N - K  x J
        #log_Zj = [torch.logsumexp(gji[j],axis=0).unsqueeze(0) for j in range(self.J)] # 1 - K  x J
        #log_aji = [gji[j][idx_n] - log_Zj[j] for j in range(J)  ]                     # b - K  x J
        gji = [m.affine_all_z(xj[idx_n]) for m,xj in zip(self.rec_models, xjis)]       # N - K  x J
        log_Zj = [torch.logsumexp(gji[j],axis=0).unsqueeze(0) for j in range(self.J)]  # 1 - K  x J
        log_aji = [gji[j] - log_Zj[j] for j in range(self.J) ]                         # b - K  x J
        log_joint = self.latent_prior.log_probs() + sum(log_aji)                       # b - K
        return torch.logsumexp(log_joint,axis=-1)                                      # b

    def eval(self, idx_n):

        xjis = [self.pxjs[j].x for j in range(self.J)]                                # N - D
        gji = [m.affine_all_z(xj) for m,xj in zip(self.rec_models, xjis)]             # N - K  x J
        log_Zj = [torch.logsumexp(gji[j],axis=0).unsqueeze(0) for j in range(self.J)] # 1 - K  x J
        log_aji = [gji[j][idx_n] - log_Zj[j] for j in range(self.J)  ]                # b - K  x J
        log_joint = self.latent_prior.log_probs() + sum(log_aji)                      # b - K
        log_w =  torch.logsumexp(log_joint,axis=-1)                                   # b
        posterior = torch.exp(log_joint - log_w.unsqueeze(-1))                        # b - K
        return log_w[:,0], posterior[:,0]

    def training_step(self, idx_n, batch_idx):
        # score matching loss
        loss = - self.log_probs(idx_n).mean() # average negative log(w(x))
        return loss.sum()


class discreteRPM_localLatents(torch.nn.Module):
    """
    Recognition-Parametrized Model (RPM)
    """
    def __init__(self, rec_models, latent_prior_g, latent_priors_j, px_alljs):

        super().__init__()
        self.J = len(rec_models)
        assert len(px_alljs.pxjs) == self.J

        self.rec_models = torch.nn.ModuleList(rec_models)
        self.latent_prior_g = latent_prior_g
        self.latent_priors_j = torch.nn.ModuleList(latent_priors_j)
        self.K_j = [len(prior_j.param_) for prior_j in latent_priors_j]
        self.px_alljs = px_alljs

    def log_probs(self, xjs, idx_n):
        J = self.J
        assert len(xjs) == J
        N = xjs[0].shape[0]
        assert all([xjs[j].shape[0] == N for j in range(self.J)])

        log_fnji = [m.log_probs(xj) for m,xj in zip(self.rec_models, xjs)]                               # N-Kj-K  x J
        log_hnj = [ pxj.model.log_probs_unnormalized(idx_n) for pxj in self.px_alljs.pxjs]               # N-Kj    x J
        log_pxnji = [ log_fnji[j] + log_hnj[j].unsqueeze(-1) for j in range(J) ]                         # N-Kj-K  x J
        log_Fji = [torch.logsumexp(log_pxnji[j],axis=0) for j in range(J) ]                              #   Kj-K  x J
        log_pj = [self.latent_priors_j[j].log_probs() for j in range(J)]                                 #   Kj    x J
        log_pxzj_zg = [log_pxnji[j]+(log_pj[j].unsqueeze(-1)-log_Fji[j]).unsqueeze(0) for j in range(J)] # N-Kj-K  x J
        log_px_zg = sum([torch.logsumexp(log_pxzj_zg[j],axis=1) for j in range(J)])                      # N -  K
        log_px = torch.logsumexp(self.latent_prior_g.log_probs().unsqueeze(0) + log_px_zg,axis=1)        # N

        return log_px 

    def eval(self, xjs, idx_n):
        J = self.J
        assert len(xjs) == J
        N = xjs[0].shape[0]
        assert all([xjs[j].shape[0] == N for j in range(self.J)])

        log_fnji = [m.log_probs(xj) for m,xj in zip(self.rec_models, xjs)]                               # N-Kj-K  x J
        log_hnj = [ pxj.model.log_probs()[idx_n,0] for pxj in self.px_alljs.pxjs]                        # N-Kj    x J
        log_pxnji = [ log_fnji[j] + log_hnj[j].unsqueeze(-1) for j in range(J) ]                         # N-Kj-K  x J
        log_Fji = [torch.logsumexp(log_pxnji[j],axis=0) for j in range(J) ]                              #   Kj-K  x J
        log_pj = [self.latent_priors_j[j].log_probs() for j in range(J)]                                 #   Kj    x J
        log_pxzj_zg = [log_pxnji[j]+(log_pj[j].unsqueeze(-1)-log_Fji[j]).unsqueeze(0) for j in range(J)] # N-Kj-K  x J

        log_px_zg_j = [torch.logsumexp(log_pxzj_zg[j],axis=1) for j in range(J)]                         # N -  K  x J
        log_px_zg = sum(log_px_zg_j)                                                                     # N -  K
        log_pg = self.latent_prior_g.log_probs()                                                         #      K
        log_pzgx = log_pg.unsqueeze(0) + log_px_zg                                                       # N -  K 
        log_px = torch.logsumexp(log_pzgx,axis=1)                                                        # N

        log_pzg_x = log_pzgx - log_px.unsqueeze(1) # posterior p(zg | x)        
        log_pzj_xs = []                            # posteriors p(zj | x)
        for j in range(J):
            i_not_j =  [i for i in range(J) if i != j]
            log_pzinotj_x = sum([log_px_zg_j[i] for i in i_not_j])                                       # N -  K
            log_pzjzg_x = log_pxzj_zg[j]+(log_pzinotj_x+log_pg.unsqueeze(0)).unsqueeze(1)                # N-Kj-K
            log_pzj_x = torch.logsumexp(log_pzjzg_x,axis=-1) - log_px.unsqueeze(1)                       # N-Kj
            log_pzj_xs.append(log_pzj_x)

        return log_pzj_xs, log_pzg_x, log_px
    
    def training_step(self, batch, batch_idx):
        # score matching loss
        xjs = batch[0]
        idx_n = batch[1]
        loss = - self.log_probs(xjs, idx_n).mean() # average negative log(w(x))
        return loss.sum()

    def eval_sum(self, xjs):
        return self.eval(xjs).sum(axis=0)

    def eval_tensor(self, txjs):
        assert txjs.shape[1] == self.J
        xjs = [txjs[:,j] for j in range(self.J)]
        return self.eval(xjs)

    def forward(self, xjs):
        return [self.rec_models[j](xjs[j])+self.latent_prior.param for j in range(self.J)]



class discreteRPM(torch.nn.Module):
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
        J = self.J
        assert len(xjs) == J
        N = xjs[0].shape[0]
        assert all([xjs[j].shape[0] == N for j in range(self.J)])

        log_fnji = torch.stack([m.log_probs(xj)for m,xj in zip(self.rec_models, xjs)], axis=1)  # N-J-K
        Fji  = torch.exp(log_fnji).mean(axis=0)                                                 #   J-K
        log_prod_Fj = torch.log(Fji).sum(axis=0).reshape(1,-1)                                  # 1 - K
        log_prod_fj = log_fnji.sum(axis=1)                                                      # N - K
        log_prod_frac = log_prod_fj - log_prod_Fj                                               # N - K

        log_joint_factor = self.latent_prior.log_probs() + log_prod_frac                        # N - K
        logw = torch.log(torch.exp(log_joint_factor).sum(axis=-1))                              # N
        posterior = torch.exp(log_joint_factor - logw.reshape(-1,1))                            # N - K

        log_prior_tilda = torch.log(posterior.mean(axis=0)).reshape(1,-1)                       # 1 - K
        log_joint_tilda = log_prior_tilda + log_prod_frac                                       # N - K
        w_tilda = torch.exp(log_joint_tilda).sum(axis=-1)                                       # N

        log_joint_tilda_j = log_fnji + (log_prior_tilda - torch.log(Fji)).reshape(1,*Fji.shape) # N-J-K        
        w_tilda_j = torch.exp(log_joint_tilda_j).sum(axis=-1)                                   # N-J        
        posterior_tilda_j = torch.exp(log_joint_tilda_j) / w_tilda_j.reshape(-1,J,1)            # N-J-K

        return logw, posterior, w_tilda_j, posterior_tilda_j #logp0.reshape(N) + logw

    def training_step(self, batch, batch_idx):
        # score matching loss
        xjs = batch
        loss = - self.eval(xjs)[0].mean() # average negative log(w(x))
        return loss.sum()

    def eval_sum(self, xjs):
        return self.eval(xjs).sum(axis=0)

    def eval_tensor(self, txjs):
        assert txjs.shape[1] == self.J
        xjs = [txjs[:,j] for j in range(self.J)]
        return self.eval(xjs)

    def forward(self, xjs):
        return [self.rec_models[j](xjs[j])+self.latent_prior.param for j in range(self.J)]


class discreteRPVAE(torch.nn.Module):
    """
    Variational auto-encoder for recognition-Parametrized Model (RPM)
    """
    def __init__(self, rec_models, latent_prior, pxjs):

        super().__init__()
        self.J = len(rec_models)
        assert len(pxjs) == self.J

        self.rec_models = torch.nn.ModuleList(rec_models)
        self.latent_prior = latent_prior
        self.pxjs = pxjs

    def eval(self, xjs):
        J = self.J
        assert len(xjs) == J
        N = xjs[0].shape[0]
        assert all([xjs[j].shape[0] == N for j in range(self.J)])

        log_fnji = torch.stack([m.log_probs(xj)for m,xj in zip(self.rec_models, xjs)], axis=1)  # N-J-K
        Fji  = torch.exp(log_fnji).mean(axis=0)                                                 #   J-K
        log_prod_Fj = torch.log(Fji).sum(axis=0).reshape(1,-1)                                  # 1 - K
        log_prod_fj = log_fnji.sum(axis=1)                                                      # N - K
        log_prod_frac = log_prod_fj - log_prod_Fj                                               # N - K

        log_joint_factor = self.latent_prior.log_probs() + log_prod_frac                        # N - K
        logw = torch.log(torch.exp(log_joint_factor).sum(axis=-1))                              # N

        posterior = torch.exp(log_joint_factor - logw.reshape(-1,1))                             # N - K

        #log_prior_tilda = torch.log(posterior.mean(axis=0)).reshape(1,-1)                       # 1 - K
        #log_joint_tilda = log_prior_tilda + log_prod_frac                                       # N - K
        #w_tilda = torch.exp(log_joint_tilda).sum(axis=-1)                                       # N

        #log_joint_tilda_j = log_fnji + (log_prior_tilda - torch.log(Fji)).reshape(1,*Fji.shape) # N-J-K        
        #w_tilda_j = torch.exp(log_joint_tilda_j).sum(axis=-1)                                   # N-J        
        #posterior_tilda_j = torch.exp(log_joint_tilda_j) / w_tilda_j.reshape(-1,J,1)            # N-J-K

        return logw, posterior

    def elbo(self, xjs):
        J = self.J
        assert len(xjs) == J
        N = xjs[0].shape[0]
        assert all([xjs[j].shape[0] == N for j in range(self.J)])

        log_fnji = torch.stack([m.log_probs(xj)for m,xj in zip(self.rec_models, xjs)], axis=1)  # N-J-K
        Fji  = torch.exp(log_fnji).mean(axis=0)                                                 #   J-K
        log_prod_Fj = torch.log(Fji).sum(axis=0).reshape(1,-1)                                  # 1 - K
        log_prod_fj = log_fnji.sum(axis=1)                                                      # N - K

        log_q = (1-J) * self.latent_prior.log_probs() + log_prod_fj                             # N - K
        lognorm_q = torch.logsumexp(log_q, axis=1).unsqueeze(-1)                                # N - 1 
        log_q = log_q - lognorm_q                                                               # N - K
        
        log_ratio = lognorm_q + (J * self.latent_prior.log_probs() - log_prod_Fj)               # N - K
        elbo = (torch.exp(log_q) * log_ratio).sum(axis=1)                                       # N 

        return elbo

    def training_step(self, batch, batch_idx):
        # score matching loss
        xjs = batch
        loss = - self.elbo(xjs).mean() 
        return loss.sum()


class discretenonCondIndRPM(torch.nn.Module):
    """
    Recognition-Parametrized Model (RPM) without conditional independence assumption
    """
    def __init__(self, rec_model, latent_prior, pxjs, full_F=True):

        super().__init__()

        self.rec_model = rec_model
        self.latent_prior = latent_prior
        self.pxjs = pxjs
        self.J = pxjs.J
        self.full_F = full_F

    def eval(self, xs):
        # xs be of shape batchsize-J-D
        # xs = torch.stack(xjs, axis=1)                                  # 
        J = self.J

        if self.full_F: # compute normalizer across all N datapoints
            all_xjs = [pxj.x for pxj in self.pxjs.pxjs]
            N = all_xjs[0].shape[0]
            assert all([N == xj.shape[0] for xj in all_xjs])

        else:           # compute normalizer only across datapoints in current minibatch
            all_xjs = [xs[:,j] for j in range(J)] # hard assumes that j=1,,.,J instances are stacked in axis=1
            N = all_xjs[0].shape[0] # N = batchsize from here on !

        shuffle_ids = torch.cartesian_prod(*[torch.arange(N,dtype=torch.long) for j in range(J)])
        xshuffled = torch.stack([all_xjs[j][shuffle_ids[:,j]] for j in range(J)], axis=1) # N^J-J-K !!!!
        m = self.rec_model
        log_fni = m.log_probs(xs)                                           # batchsize - K
        log_prior =  self.latent_prior.log_probs().reshape(1,-1)            # 1         - K
        log_fxs = m.log_probs(xshuffled)                                    # N^J       - K 
        denom = torch.logsumexp(log_fxs,dim=0).reshape(1,-1) - np.log(N**J) # 1         - K
        log_px = torch.logsumexp(log_fni + log_prior - denom, dim=-1)       # batchsize - 1

        return log_px

    def training_step(self, batch, batch_idx):
        # score matching loss
        xs = batch
        loss = - self.eval(xs).mean() # average negative log(w(x))
        return loss.sum()

    def forward(self, xs):
        return self.rec_model(xs)+self.latent_prior.param

class Prior_discrete(torch.nn.Module):
    """
    Discrete distribution for use as RPM prior.
    """
    def __init__(self, param, activation_out=None):

        super().__init__()
        if activation_out is None:
            def activation_out(x):
                assert x.ndim == 1
                return torch.nn.LogSoftmax(dim=0)(x)
        self.activation_out = activation_out
        self.param_ = torch.nn.Parameter(param)

    def log_probs(self, x=None):

        return self.activation_out(self.param_)


class RecognitionFactor_discrete(torch.nn.Module):
    """
    Discrete conditional distribution
    """
    def __init__(self, model):

        super().__init__()
        self.model = model # model up to softmax layer

    def log_probs(self, x):
        assert x.ndim >= 2 # N-by-K, where K is number of choices, or N-by-something-by-K
        return torch.nn.LogSoftmax(dim=-1)(self.model(x))
    
    def forward(self, x):
        return self.log_probs(x) # categorical distribution: eta(xj) = P(Z|xj) in vector form


class RecognitionFunction_discrete(torch.nn.Module):
    """
    Discrete recognition function g(xj, Z) *without* any normalization constraints on \int e^g(xj,Z) dZ
    """
    def __init__(self, model):

        super().__init__()
        self.model = model # model up to softmax layer

    def affine_all_z(self, x):
        assert x.ndim >= 2 # N-by-K, where K is number of choices, or N-by-something-by-K
        out_all = self.model(x)                     # N-by-...-by-K+1
        return out_all[...,:-1] + out_all[...,-1:]
    
    def forward(self, x):
        return self.affine_all_z(x) # categorical distribution: eta(xj) = P(Z|xj) in vector form


class RecognitionFunction_discrete_norm(torch.nn.Module):
    """
    Discrete recognition function g(xj, Z) with normalization constraint \int e^g(xj,Z) dZ = 1
    """
    def __init__(self, model):

        super().__init__()
        self.model = model # model up to softmax layer

    def affine_all_z(self, x):
        assert x.ndim >= 2 # N-by-K, where K is number of choices, or N-by-something-by-K
        return torch.nn.LogSoftmax(dim=-1)(self.model(x)[...,:-1])
    
    def forward(self, x):
        return self.affine_all_z(x) # categorical distribution: eta(xj) = P(Z|xj) in vector form

    
class RecognitionFactor_scaled_discrete(torch.nn.Module):
    """
    Scaled discrete conditional distribution
    """
    def __init__(self, model):

        super().__init__()
        self.model = model # model up to softmax layer

    def log_probs(self, x):
        assert x.ndim >= 2 # N-by-K, where K is number of choices, or N-by-something-by-K
        return torch.nn.LogSoftmax(dim=-1)(self.model(x)[:,:-1]) + self.model(x)[:,-1:]
    
    def forward(self, x):
        return self.log_probs(x) # categorical distribution: eta(xj) = P(Z|xj) in vector form

class RecognitionFactor_zj_discrete(torch.nn.Module):
    """
    Discrete conditional distribution 
    conditioned on both a continuous x and a categorical z : model now needs to return K log-odds
    per x and z=[1,..,K] indexes which one gets used. 
    """
    def __init__(self, model, K):

        super().__init__()
        self.model = model # model up to softmax layer
        self.K = K         # number of possible values for categorical (local) latent variable z

    def log_probs(self, x, z=None):
        assert x.ndim >= 2 # N-by-K, where K is number of choices, or N-by-something-by-K
        if z is None:
            return torch.nn.LogSoftmax(dim=-1)(self.model(x).reshape(x.shape[0],self.K,-1) )
        else:
            assert len(x) == len(z)
            assert max(z) <= self.K-1
            return torch.nn.LogSoftmax(dim=-1)(self.model(x).reshape(x.shape[0],self.K,-1) )[torch.arange(len(z)),z]

    def forward(self, x, z=None):
        return self.log_probs(x, z) # categorical distribution: eta(xj) = P(Z|xj) in vector form

class WeightedEmpiricalDistribution(torch.nn.Module):
    """
    Rudimentary representation of a weighted mixture distribution of Dirac delta peaks.
    """
    def __init__(self, x, model):
        super().__init__()
        self.x = x
        self.N = x.shape[0]
        self.model = model

    def log_probs(self, n, z): # n is index of data point x[n,:] - otherwise log-prob = - inf !
        return self.model.log_probs(z)[n]

    def eval(self, n, z):
        return self.log_probs(n, z)

    def sample(self,z):
        i = np.random.choice(self.N, size=1, p=torch.exp(self.model.log_probs(z)).detach().numpy())
        return self.x[i]


class EmpWeightModel_RPM(torch.nn.Module):
    """
    Discrete conditional distribution
    """
    def __init__(self, model, x):

        super().__init__()
        self.x = x
        self.model = model # model up to softmax layer

    def log_probs(self):
        log_alpha = torch.nn.LogSoftmax(dim=-1)(self.model(self.x)) # alpha(zj | xj) for all zj, xj pairs with xj in xx    
        return log_alpha - torch.logsumexp(log_alpha, axis=0).unsqueeze(0)

    def log_probs_unnormalized(self,idx_n):
        log_alpha = torch.nn.LogSoftmax(dim=-1)(self.model(self.x[idx_n])) # alpha(zj | xj) for all zj, xj pairs with xj in minibatch 
        return log_alpha

    def forward(self):
        return self.log_probs() # categorical distribution: eta(xj) = P(Z|xj) in vector form


class WeightModel(torch.nn.Module):
    """
    Saturated conditional categorical distribution
    conditioned on discrete latent variable z. 
    """
    def __init__(self, M):
        super().__init__()        
        self.M = M
        self.K = M.shape[0]
        self.N = M.shape[1]

    def log_probs(self, z=None):
        if z is None:
            return torch.nn.LogSoftmax(dim=-1)(self.M)
        else:
            return torch.nn.LogSoftmax(dim=-1)(self.M)[z]

    def forward(self, z=None):
        return self.log_probs(z)

class RPMWeightedEmpiricalMarginals(torch.nn.Module):
    """
    Placeholder object to represent all marginal data distribution pj(xj), j=1,...,J
    used among others to define a recognition-parametrized model.
    """
    def __init__(self, pxjs):
        super().__init__()
        self.J = len(pxjs)
        self.N = pxjs[0].x.shape[0]
        assert all([self.N == pxjs[j].x.shape[0] for j in range(self.J)])
        self.pxjs = torch.nn.ModuleList(pxjs) #[EmpiricalDistribution(x=xj) for xj in xjs]
