import torch
import numpy as np
from expFam import ConditionalExpFam, SemiparametricConditionalExpFam

class RPM(torch.nn.Module):
    """
    Recognition-Parametrized Model (RPM) for exponential family recognition factors/prior.
    - note we choose the exponential family of the approximate posterior q(Z) to lie in the family of the RPM prior p(Z),
      rather than in the family of the factors f(Z|xj).
    """
    def __init__(self, rec_factors, latent_prior, px, q, nu=None, iviNatParametrization='classic',
                 stack_xjs_new_axis=True, full_N_for_Fj=True):
        super().__init__()
        self.J = len(rec_factors)
        self.joint_model = torch.nn.ModuleList([torch.nn.ModuleList(rec_factors), latent_prior, px])
        self.q = q # recognition model, can be saturated (i.e. q(Z|{X}_n=1^N) = q_n(Z)) or amortized
        if nu is None or torch.is_tensor(nu): # model for inner variational parameters. 
            self.nu = nu                      # can be either of amortized, saturated or 
        else:                                 # 'analytic' (based on an assumption that Fj(Z) -> p(Z) which
            self.nu = torch.nn.ModuleList(nu) # won't ever be absolutely true in practice).
        self.N = px.pxjs[0].x.shape[0]
        assert np.all([px.pxjs[j].x.shape[0] == self.N for j in range(self.J)])
        assert iviNatParametrization in ('classic', 'delta')
        self.iviNatParametrization = iviNatParametrization
        self.stack_xjs_new_axis = stack_xjs_new_axis
        self.full_N_for_Fj = full_N_for_Fj

    def stack_xj(self, xjs):
        if self.stack_xjs_new_axis:
            return torch.stack(xjs, axis=1)
        else: # e.g. if data is images, stack along channel axis !
            return torch.cat(xjs, axis=1)

    def comp_eta_q(self, xjs, eta_0 = None, idx_data = None):
        if eta_0 is None:
            prior = self.joint_model[1]
            eta_0 = prior.nat_param
        if self.q == 'use_theta': # q becomes analytic if Fj(Z) = p(Z) for all j
            eta_j = self.factorNatParams(xjs, eta_off=eta_0)
            eta_q = (1-self.J) * eta_0 + eta_j.sum(axis=1)
            return eta_q, eta_j
        elif isinstance(self.q, SemiparametricConditionalExpFam): # saturated q(Z|{X}_n=1^N) = q_n(Z)
            eta_q = self.q.nat_param(nat_param_offset=eta_0)[idx_data]
            return eta_q
        elif isinstance(self.q, ConditionalExpFam):               # amortized q(ZXX) with eta_q(X) given by NN. 
            x = self.stack_xj(xjs)
            eta_q = self.q(x) + eta_0 # reparametrize to ensure valid implicit likelihoods (no negative variance etc.)
            return eta_q
        else:
            raise NotImplementedError()

    def comp_q_log_normalizer(self, eta_q, eta_j, eta_0):
        # log int q(Z) \prod_j fj(Z|xj) dZ = phi(\sum_j etaj + (1-J)eta0) - \sum_j phi(etaj) + (1-J) phi(eta0),
        # assuming that q(Z) = p(Z)^(1-J) and that p(Z) and all fj(Z|xj) share the same exponential family. 
        rec_factors, prior = self.joint_model[0], self.joint_model[1]
        log_norm_q = prior.log_partition(eta_q)
        log_norm_q -= (1-self.J) * prior.log_partition(eta_0)
        log_norm_q -= self.factorLogPartition(eta_j).sum(axis=1)

        return log_norm_q

    def elbo_innervi(self, xjs, idx_data=None):
        """ compute evidence lower bound across data: 1/N sum_n elbo(x_n), where
            elbo(x) = (eta_0 + sum_j etaj(xj) - eta_q)' Eq[t(Z)] 
                    + Phi(eta_q) - Phi(eta_0)  - sum_j Phi(etaj(xj))
                    - sum_j Eq[log 1/N sum_n chi(Z) exp( etaj(xj_n)'t(Z) - Phi(etaj(xj_n)) )]
            Note that the terms E_q(Z|X)[log Fj(Z)] are tricky for continuous latents, hence we will approximate and
            introduce a lower bound to the lower bound. 
            Also note that the code ignores a term E_q(Z|X)[ log chi(Z) - log chi_q(Z) + sum_j chi_j(Z)],
            where chi, chi_q and chi_j are the base measures of p(Z), q(Z|X) and fj(Z|xj), respectively!
                xjs : list of len(xjs) = J of batchsize-by-d tensors containing partial observations xj. 
                idx_data : indices of data points in xjs, which we nead because our posterior
                approximation q(Z|X^n)= q^n(Z) is saturated and hence a funciton of data index n.
        """

        J = self.J
        assert len(xjs) == J
        if type(xjs[0]) in [tuple, list]:
            N = xjs[0][0].shape[0] 
            assert all([xjs[j][0].shape[0] == N for j in range(self.J)])
        else:
            N = xjs[0].shape[0] 
            assert all([xjs[j].shape[0] == N for j in range(self.J)])
        rec_factors, prior = self.joint_model[0], self.joint_model[1]

        # elbo(x) = (eta_0 + sum_j etaj(xj) - eta_q)' Eq[t(Z)]  + ...
        eta_0 = prior.nat_param
        etajs = self.factorNatParams(xjs, eta_off=eta_0)

        eta_q = self.comp_eta_q(xjs, eta_0, idx_data)
        EqtZ = prior.log_partition.nat2meanparam(eta_q)

        elbo = ((eta_0 + etajs.sum(axis=1) - (1+J) * eta_q) * EqtZ).sum(axis=-1) # +J because lower bound, see below

        # elbo(x) -= Phi(eta0)  + sum_j Phi(etaj(xj)) - Phi(eta_q)
        phi_0 = prior.phi()
        phijs = torch.stack([m.phi_x(xj, eta_off=eta_0) for m,xj in zip(rec_factors,xjs)],axis=1)
        elbo -=  phijs.sum(axis=1) + phi_0 - (1+J) * self.q.log_partition(eta=eta_q) # again +J, see below

        # elbo(x) -= sum_j Eq[log 1/N sum_n chi(Z)  exp( etaj(xj_n)'t(Z) - Phi(etaj(xj_n)) )]
        # where we are lower-bounding  - Eq[log Fj(Z)] >= - log int Fj(Z)/hj(Z) dZ - Eq[log hj(Z)] + H[q]
        # with hj(Z) = exp(nu_j't(Z)) introducing a new variational parameter nu_j (per data point n).
        # Quick hack: select nu_j = eta_0 - eta_q (thanks to Hugo Soulat)!
        nu_j = self.iviNatParams(eta_0, idx_data, eta_q, xjs)  # batch_size-by-J-by-D 
        # 1. compute int Fj(Z)/hj(Z) dZ by the normalizers of *all* components of mixture Fj(Z)
        idx_all = np.arange(N) if self.full_N_for_Fj else idx_data
        xjs_all = [pxj.x[idx_all] for pxj in self.joint_model[2].pxjs]
        etajs_all = self.factorNatParams(xjs=xjs_all, eta_off=eta_0) # N-by-J-by-D
        phijs_all = self.factorLogPartition(etajs_all)               # N-by-J
        phi_jnm = prior.log_partition(etajs_all.unsqueeze(1) - nu_j.unsqueeze(0)) # N-by-batch_size-by-J
        elbo -= torch.logsumexp(phi_jnm - phijs_all.unsqueeze(1), dim=0).sum(axis=1) - J*np.log(N)
        # 2. Eq[log hj(Z)] = nu_j' Eq[t(Z)]
        elbo -= (nu_j.sum(axis=1) * EqtZ).sum(axis=-1)
        # 3. We already added- H[q] - Eq[log chi(Z)] = eta_q'Eq[t(Z)] - Phi(eta_q) above !

        return elbo

    def elbo_innervi__q_theta(self, xjs, idx_data=None):
        """ compute evidence lower bound across data: 1/N sum_n elbo(x_n), where
            elbo(X) =  log int p(Z)^{1-J} \prod_j fj(Z|xj) dZ + J * E_q(Z|X)[log p(Z)] - \sum_j E_q(Z|X)[log Fj(Z)],
            which results from choosing q(Z|X) \propto p(Z)^{1-J} prod_j fj(Z|xj). Assumes that all base measures and
            sufficient statistics of p(Z) and all fj(Z|xj) are identical !
            Note that the terms E_q(Z|X)[log Fj(Z)] are tricky for continuous latents, hence we will approximate and
            introduce a lower bound to the lower bound. 
            Also note that the code ignores a term J*E_q(Z|X)[log chi(Z)], where chi is the base measure of p(Z)!
                xjs : list of len(xjs) = J of batchsize-by-d tensors containing partial observations xj. 
                idx_data : indices of data points in xjs, which we nead because our posterior
                approximation q(Z|X^n)= q^n(Z) is saturated and hence a funciton of data index n.
        """
        rec_factors, prior = self.joint_model[0], self.joint_model[1]
        eta_0 = prior.nat_param
        eta_q, eta_j = self.comp_eta_q(xjs, eta_0)
        EqtZ = prior.log_partition.nat2meanparam(eta_q)

        # elbo(X) = int q(Z) \prod_j fj(Z|xj) dZ
        elbo = self.comp_q_log_normalizer(eta_q, eta_j, eta_0)

        # elbo(X) += J * E_q(Z|X)[log p(Z)]
        elbo += self.J * (torch.sum(EqtZ * eta_0, axis=-1) - prior.phi())

        # elbo(X) -= \sum_j E_q(Z|X)[log Fj(Z)]
        # where we are lower-bounding  - Eq[log Fj(Z)] >= - log int Fj(Z)/hj(Z) dZ - Eq[log hj(Z)] + H[q]
        # with hj(Z) = exp(nu_j't(Z)) introducing a new variational parameter nu_j (per data point n).
        nu_j = self.iviNatParams(eta_0, idx_data, eta_q, xjs)  # batch_size-by-J-by-D 
        # 1. compute int Fj(Z)/hj(Z) dZ by the normalizers of *all* components of mixture Fj(Z)
        idx_all = np.arange(N) if self.full_N_for_Fj else idx_data
        xjs_all = [pxj.x[idx_all] for pxj in self.joint_model[2].pxjs]
        etajs_all = self.factorNatParams(xjs=xjs_all, eta_off=eta_0) # N-by-J-by-D
        phijs_all = torch.stack([rec_factors[j].log_partition(etajs_all[:,j]) for j in range(self.J)],axis=1)
        phi_jnm = prior.log_partition(etajs_all.unsqueeze(1) - nu_j.unsqueeze(0)) # N-by-batch_size-by-J
        elbo -= (torch.logsumexp(phi_jnm - phijs_all.unsqueeze(1), dim=0).sum(axis=1) - self.J*np.log(etajs_all.shape[0]))
        # 2. - Eq[log hj(Z)] = - nu_j' Eq[t(Z)]
        elbo -= (nu_j.sum(axis=1) * EqtZ).sum(axis=-1)
        # 3. H[q] = - eta_q'Eq[t(Z)] + Phi(eta_q) 
        elbo -= self.J * (torch.sum(EqtZ * eta_q, axis=-1) - prior.log_partition(eta_q))

        return elbo

    def factorNatParams(self, xjs=None, eta_off=0.):
        """ computes natural parameters for conditional densities fj(Z|xj) """
        rec_factors, pxjs = self.joint_model[0], self.joint_model[2].pxjs
        if xjs is None:
            # assuming data marginals p(xj) are given by empirical distributions that store their data {xj_n}_n 
            return torch.stack([m(pxj.x) for m,pxj in zip(rec_factors, pxjs)],axis=1) + eta_off
        else:
            assert len(xjs) == self.J
            return torch.stack([m(xj) for m,xj in zip(rec_factors, xjs)],axis=1) + eta_off

    def factorLogPartition(self, etajs, log_partitions=None):
        """ computes natural parameters for conditional densities fj(Z|xj) """
        rec_factors, pxjs = self.joint_model[0], self.joint_model[2].pxjs
        log_partitions = [m.log_partition for m in rec_factors] if log_partitions is None else log_partitions
        return torch.stack([log_partitions[j](etajs[:,j]) for j in range(self.J)],axis=1) # N-by-J-by-something


    def iviNatParams(self, eta_0, idx_data=None, eta_q=None, xjs=None):
        """ compute inner variational bound natural parameters """
        nu_j_base = (eta_0 - eta_q).unsqueeze(1).expand((-1,self.J,-1))  # batch_size-by-J-by-D
        if self.nu is None:
            return nu_j_base
        elif torch.is_tensor(self.nu): 
            assert self.nu.requires_grad
            if self.iviNatParametrization == 'classic':
                return - self.nu[idx_data]
            elif self.iviNatParametrization == 'delta':
                return nu_j_base - self.nu[idx_data]        
        elif np.all([isinstance(self.nu[j], ConditionalExpFam) for j in range(self.J)]): # fully amortized case !
            x = self.stack_xj(xjs)
            if self.iviNatParametrization == 'classic':
                return - torch.stack([m(x) for m in self.nu], axis=1)
            elif self.iviNatParametrization == 'delta':
                return nu_j_base - torch.stack([m(x) for m in self.nu], axis=1)

    def elbo_revJensen(self, xjs, idx_data=None):
        # elbo(X) =  int q(Z) \prod_j fj(Z|xj) dZ - \sum_j E_q(Z|X)[log Fj(Z)/p(Z)]

        rec_factors, prior = self.joint_model[0], self.joint_model[1]
        eta_0 = prior.nat_param
        eta_q, eta_j = self.comp_eta_q(xjs, eta_0)

        # elbo(X) += int q(Z) \prod_j fj(Z|xj) dZ
        elbo = self.comp_q_log_normalizer(eta_q, eta_j, eta_0)

        # elbo(X) -= \sum_j E_q(Z|X)[log Fj(Z)/p(Z)]
        # we use a 'reverse Jensen' bound E_q(Z|X)[log 1/N \sum_n fj(Z|xn)/p(Z)] <= 1/n \sum_n E_q(Z|X)[log fj(Z|xn)/p(Z)]
        # and hope for the best !
        phiq = prior.log_partition(eta_q)
        phi0 = prior.phi()
        elbo += phiq - phi0 

        etajs_all = self.factorNatParams(eta_off=eta_0)        # N-by-J-by-D
        phijs_all = torch.stack([rec_factors[j].log_partition(etajs_all[:,j]) for j in range(self.J)],axis=1)
        N, J = etajs_all.shape[0], self.J
        elbo += J * np.log(N) 

        etaZ = eta_q.unsqueeze(0).unsqueeze(2) - eta_0.unsqueeze(0).unsqueeze(1) + etajs_all.unsqueeze(1) # N-by-batchsize-by-J-by-T
        phiZ = torch.stack([rec_factors[j].log_partition(etaZ[:,:,j]) for j in range(self.J)],axis=2)
        elbo -= torch.logsumexp(phiZ - phijs_all.unsqueeze(1),dim=0).sum(axis=1)

        return elbo

    def elbo(self, xjs, idx_data=None):
        if isinstance(self.q, SemiparametricConditionalExpFam) or isinstance(self.q, ConditionalExpFam):
            return self.elbo_innervi(xjs, idx_data)
        elif self.q == 'use_theta':
            return self.elbo_innervi__q_theta(xjs, idx_data)
        else: 
            raise NotImplementedError()

    def training_step(self, batch, idx_data, batch_idx=None):
        xjs = batch
        loss = - self.elbo(xjs, idx_data).mean(axis=0) # average negative evidence lower bound
        return loss

    def test_step(self, batch, idx_data, batch_idx=None):
        xjs = batch
        if isinstance(self.q, SemiparametricConditionalExpFam) or torch.is_tensor(self.nu):
            return torch.nan # not amortized - would need extra optimization over q^(n) resp. nu_j^(n) !
        with torch.no_grad():
            loss = - self.elbo(xjs, idx_data).mean(axis=0) # average negative evidence lower bound
        return loss


class RPMtemp(RPM):
    """ Temporally structured recognition-paremetrized variational autoencoder. 
    Uses the same natural parameterization of exponential families as the non-termporal RP-VAE class,
    but actually only works on Gaussian distriutions for prior p(Z) and recognition factors fj(Zt|xjt)
    due to strongly relying on Gaussian marginalization properties.
    """
    def comp_eta_q(self, xjs, eta_0=None, eta_0_uncorr=None, idx_data = None):
        # eta_q(x) = \sum_j etaj(xj) + eta0 - J eta0_uncorreletad
        # Note that factor parameters etaj(xj) = eta0_unc + gj(xj) are reparametrized to ensure  
        # valid posteriors Zt|xj (e.g. non-negative covariance in case of Gaussian conditionals). 
        prior = self.joint_model[1]
        eta_0 = prior.nat_param if eta_0 is None else eta_0
        eta_0_uncorr = prior.log_partition.decorrelate_natparam(eta_0) if eta_0_uncorr is None else eta_0_uncorr 

        if self.q == 'use_theta':
            eta_j = self.factorNatParams(xjs, eta_off=eta_0_uncorr)                      # N-by-J-by-D
            eta_q = eta_0 - self.J * eta_0_uncorr + eta_j.sum(axis=1)                    # N - by  - D
            etajs_diag = prior.log_partition.extract_diagonal(self.factorNatParams(xjs, eta_off=eta_0_uncorr))
            etajs_diag = etajs_diag.reshape(*etajs_diag.shape[:-1],2,-1).transpose(-1,-2) # N-by-J-by-T-by-2
            return eta_q, etajs_diag
        elif isinstance(self.q, SemiparametricConditionalExpFam):
            eta_q = self.q.nat_param(nat_param_offset=eta_0)[idx_data]
            return eta_q
        elif isinstance(self.q, ConditionalExpFam):
            x = torch.cat(xjs, axis=1) #= torch.stack(xjs, axis=1)
            eta_q = self.q(x) + eta_0
            return eta_q
        else:
            raise NotImplementedError()

    def comp_q_log_normalizer(self, eta_q, etajs_diag, eta_0, eta_0_diag):
        # log int q(Z) \prod_j fj_unc(Z|xj) dZ = phi(\sum_j etaj + eta0 - J eta0_uncorr) 
        #                                     - \sum_j phi(etaj) + phi(eta0) - J phi(eta0_uncorr),
        # assuming that q(Z) = p(Z)/(prod_t p(Zt)^J and that p(Zt) and all fj(Zt|xj) share the same exponential family. 
        rec_factors, prior = self.joint_model[0], self.joint_model[1]
        marginal_log_partition_js = [m.log_partition.marginal_log_partition for m in rec_factors] 
        log_norm_q = prior.log_partition(eta_q)
        log_norm_q -= prior.log_partition(eta_0)
        log_norm_q += self.J * prior.log_partition.marginal_log_partition(eta_0_diag).sum(axis=1)
        log_norm_q -= self.factorLogPartition(etajs_diag, log_partitions=marginal_log_partition_js).sum(axis=(-2,-1))

        return log_norm_q

    def elbo_innervi(self, xjs, idx_data=None):
        """ compute evidence lower bound across data: 1/N sum_n elbo(x_n), where
            elbo(x) = (eta_0 - eta_q)'Eq[t(Z)] + sum_j sum_t etaj(xjt)'Eq[t(Zt)]  
                    + Phi(eta_q) - Phi(eta_0)  - sum_j sum_t Phi_t(etaj(xjt))
                    - sum_j sum_t Eq[log 1/N sum_n chi(Zt) exp( etaj(xjt_n)'t(Zt) - Phi(etaj(xjt_n)) )]
            Note that the terms E_q(Zt|X)[log Fj(Zt)] are tricky for continuous latents, hence we will
            approximate and introduce a lower bound to the lower bound. 
            Also note that the code ignores terms E_q(Z|X)[log chi(Z)-log chi_q(Z)]+sum_j sum_t E_q(Zt|X)[chi_j(Zt)],
            where chi, chi_q and chi_j are the base measures of p(Z), q(Z|X) and fj(Zt|xj), respectively!
                xjs : list of len(xjs) = J of batchsize-by-d tensors containing partial observations xj. 
                idx_data : indices of data points in xjs, which we nead because our posterior
                approximation q(Z|X^n)= q^n(Z) is saturated and hence a funciton of data index n.
        """
        J = self.J
        assert len(xjs) == J
        if type(xjs[0]) in [tuple, list]:
            N = xjs[0][0].shape[0] 
            assert all([xjs[j][0].shape[0] == N for j in range(self.J)])
        else:
            N = xjs[0].shape[0] 
            assert all([xjs[j].shape[0] == N for j in range(self.J)])
        rec_factors, prior = self.joint_model[0], self.joint_model[1]

        # elbo(x) = (eta_0 - eta_q)'Eq[t(Z)] + sum_j sum_t etaj(xjt)'Eq[t(Zt)]  + ...
        full2diag_gaussian = prior.log_partition.full2diag_gaussian
        eta_0 = prior.nat_param
        eta_0_diag, eta_0_uncorr = full2diag_gaussian(eta_0)
        eta_q  = self.comp_eta_q(xjs, eta_0, eta_0_uncorr, idx_data)
        eta_q_diag, eta_q_uncorr = full2diag_gaussian(eta_q)
        T = eta_q_diag.shape[-2] # number of time points
        EqtZ = prior.log_partition.nat2meanparam(eta_q)
        EqtZ1toT = prior.log_partition.extract_diagonal(prior.log_partition.nat2meanparam(eta_q_uncorr))
        EqtZ1toT = EqtZ1toT.reshape(*EqtZ1toT.shape[:-1], 2, T).transpose(-1,-2)
        etajs_diag = prior.log_partition.extract_diagonal(self.factorNatParams(xjs, eta_off=eta_0_uncorr))
        etajs_diag = etajs_diag.reshape(*etajs_diag.shape[:-1],2,T).transpose(-1,-2)
        elbo = ((eta_0-eta_q)* EqtZ).sum(axis=-1) + (etajs_diag.sum(axis=1) * EqtZ1toT).sum(axis=(-2,-1))

        # elbo(x) -= Phi(eta0)  + sum_j sum_t Phi_t(etaj(xjt)) - Phi(eta_q)
        phi_0 = prior.phi()
        marginal_log_partition_prior = prior.log_partition.marginal_log_partition 
        marginal_log_partition_js = [m.log_partition.marginal_log_partition for m in rec_factors] 
        phijs = self.factorLogPartition(etajs_diag, log_partitions=marginal_log_partition_js)        # N-by-J-by-T
        elbo -=  phijs.sum(axis=(-2,-1)) + phi_0 - self.q.log_partition(eta=eta_q)

        # elbo(x) -= sum_j sum_t Eq[log 1/N sum_n chi(Zt) exp( etaj(xjt_n)'t(Zt) - Phi(etaj(xjt_n)) )]
        # where we are lower-bounding  - Eq[log Fj(Zt)] >= - log int Fj(Zt)/hj(Zt) dZ - Eq[log hj(Zt)] + H[q(Zt)]
        # with hj(Zt) = exp(nu_jt't(Zt)) introducing a new variational parameter nu_jt (per data point n).
        # Quick hack: select nu_jt = eta_0 - eta_q(xt) (thanks to Hugo Soulat)!
        nu_jt = self.iviNatParams(eta_0_diag, idx_data, eta_q_diag, xjs)                    # batch_size-by-J-by-D 
        # 1. compute int Fj(Z)/hj(Z) dZ by the normalizers of *all* components of mixture Fj(Z)
        etajs_all = prior.log_partition.extract_diagonal(self.factorNatParams(eta_off=eta_0_uncorr)) # N-by-J-by-D
        etajs_all = etajs_all.reshape(*etajs_all.shape[:-1],2,T).transpose(-1,-2)                    # N-by-J-by-T-by-2
        phijs_all = self.factorLogPartition(etajs_all, log_partitions=marginal_log_partition_js)     # N-by-J-by-T
        phi_jnmt = marginal_log_partition_prior(etajs_all.unsqueeze(1) - nu_jt.unsqueeze(0))         # N-by-batch_size-by-J-by-T
        elbo -= (torch.logsumexp(phi_jnmt - phijs_all.unsqueeze(1), dim=0).sum(axis=(-2,-1)) - J*T*np.log(self.N))
        # 2. Eq[log hj(Z)] = nu_j' Eq[t(Z)]
        elbo -= (nu_jt.sum(axis=1) * EqtZ1toT).sum(axis=(-2,-1))
        # 3. H[q] = - eta_q'Eq[t(Z)] + Phi(eta_q)
        elbo -= J * (torch.sum(EqtZ1toT * eta_q_diag, axis=-1) - marginal_log_partition_prior(eta_q_diag)).sum(axis=-1)

        return elbo

    def elbo_innervi__q_theta(self, xjs, idx_data=None):
        """ compute evidence lower bound across data: 1/N sum_n elbo(x_n), where
            elbo(X) =  log int q(Z) \prod_j fj_unc(Z|xj) dZ + J * E_q(Z|X)[log p_unc(Z)] - \sum_t \sum_j E_q(Zt|X)[log Fj(Zt)],
            which results from choosing q(Z|X) \propto p(Z)/(prod_t p(Zt))^J prod_j fj(Z|xj). Assumes that all base measures
            and sufficient statistics of p(Z) and all fj(Z|xj) are identical !
            Note that the terms E_q(Zt|X)[log Fj(Zt)] are tricky for continuous latents, hence we will approximate and
            introduce a lower bound to the lower bound. 
            Also note that the code ignores a term J*E_q(Z|X)[log chi(Z)], where chi is the base measure of p(Z)!
                xjs : list of len(xjs) = J of batchsize-by-d tensors containing partial observations xj. 
                idx_data : indices of data points in xjs, which we nead because our posterior
                approximation q(Z|X^n)= q^n(Z) is saturated and hence a funciton of data index n.
        """
        # elbo(X) =  log int q(Z) \prod_j fj_unc(Z|xj) dZ + J * E_q(Z|X)[log p_unc(Z)] - \sum_t \sum_j E_q(Zt|X)[log Fj(Zt)]
        # for q(Z) = p(Z) / (p_unc(Z))^J and p_unc(Z) = \prod_t p(Zt), fj_unc(Z|xj) = \prod_t fj(Zt|xjt)
        rec_factors, prior = self.joint_model[0], self.joint_model[1]

        full2diag_gaussian = prior.log_partition.full2diag_gaussian
        eta_0 = prior.nat_param
        eta_0_diag, eta_0_uncorr = full2diag_gaussian(eta_0)
        eta_q, etajs_diag = self.comp_eta_q(xjs, eta_0, eta_0_uncorr)
        eta_q_diag, eta_q_uncorr = full2diag_gaussian(eta_q)

        T = eta_q_diag.shape[-2] # number of time points
        EqtZ = prior.log_partition.nat2meanparam(eta_q)
        EqtZ1toT = prior.log_partition.extract_diagonal(prior.log_partition.nat2meanparam(eta_q_uncorr))
        EqtZ1toT = EqtZ1toT.reshape(*EqtZ1toT.shape[:-1], 2, T).transpose(-1,-2)
        marginal_log_partition = prior.log_partition.marginal_log_partition 
        marginal_log_partition_js = [m.log_partition.marginal_log_partition for m in rec_factors] 

        # elbo(X) = int q(Z) \prod_j fj_unc(Z|xj) dZ
        elbo = self.comp_q_log_normalizer(eta_q, etajs_diag, eta_0, eta_0_diag)

        # elbo(X) += J * E_q(Z|X)[log p_unc(Z)]
        elbo += self.J * (torch.sum(EqtZ1toT * eta_0_diag, axis=-1) - marginal_log_partition(eta_0_diag)).sum(axis=-1)

        # elbo(X) -= \sum_j \sum_t E_q(Zt|X)[log Fj(Zt)]
        # where we are lower-bounding  - Eq[log Fj(Zt)] >= - log int Fj(Zt)/hj(Zt) dZ - Eq[log hj(Zt)] + H[q(Zt|X)]
        # with hj(Z) = exp(nu_jt't(Zt)) introducing a new variational parameter nu_j (per data point n and time point t).
        # Note that \sum_t below is implicit below by summing over eta*t(Z) instead of eta_t*t(Zt).
        nu_jt = self.iviNatParams(eta_0_diag, idx_data, eta_q_diag, xjs)                    # batch_size-by-J-by-D 
        # 1. compute int Fj(Zt)/hj(Zt) dZ by the normalizers of *all* components of mixture Fj(Z)
        etajs_all = prior.log_partition.extract_diagonal(self.factorNatParams(eta_off=eta_0_uncorr)) # N-by-J-by-D
        etajs_all = etajs_all.reshape(*etajs_all.shape[:-1],2,T).transpose(-1,-2)                    # N-by-J-by-T-by-2
        phijs_all = self.factorLogPartition(etajs_all, log_partitions=marginal_log_partition_js)     # N-by-J-by-T
        phi_jnmt = marginal_log_partition(etajs_all.unsqueeze(1) - nu_jt.unsqueeze(0)) # N-by-batch_size-by-J-by-T
        elbo -= (torch.logsumexp(phi_jnmt - phijs_all.unsqueeze(1), dim=0).sum(axis=(-2,-1)) - self.J*T*np.log(self.N))
        # 2. - Eq[log hj(Zt)] = - nu_jt' Eq[t(Zt)]
        elbo -= (nu_jt.sum(axis=1) * EqtZ1toT).sum(axis=(-2,-1))
        # 3. H[q] = - eta_q'Eq[t(Z)] + Phi(eta_q)
        elbo -= self.J * (torch.sum(EqtZ1toT * eta_q_diag, axis=-1) - marginal_log_partition(eta_q_diag)).sum(axis=-1)

        return elbo

    def iviNatParams(self, eta_0_diag, idx_data=None, eta_q_diag=None, xjs=None):
        """ compute inner variational bound natural parameters """
        nu_jt_base = (eta_0_diag - eta_q_diag).unsqueeze(1).expand((-1,self.J,-1,-1))  # batch_size-by-J-by-T-by-2
        if self.nu is None:
            return nu_jt_base
        elif torch.is_tensor(self.nu): 
            assert self.nu.requires_grad
            if self.iviNatParametrization == 'classic':
                return - self.nu[idx_data]
            elif self.iviNatParametrization == 'delta':
                return nu_jt_base - self.nu[idx_data]
        elif np.all([isinstance(self.nu[j], ConditionalExpFam) for j in range(self.J)]): # fully amortized case !
            x = self.stack_xj(xjs)
            if self.iviNatParametrization == 'classic':
                return - torch.stack([m(x) for m in self.nu], axis=1)
            elif self.iviNatParametrization == 'delta':
                return nu_jt_base - torch.stack([m(x) for m in self.nu], axis=1)


    def elbo(self, xjs, idx_data=None):
        if isinstance(self.q, SemiparametricConditionalExpFam) or isinstance(self.q, ConditionalExpFam):
            return self.elbo_innervi(xjs, idx_data)
        elif self.q == 'use_theta':
            return self.elbo_innervi__q_theta(xjs, idx_data)
        else: 
            raise NotImplementedError()

    def training_step(self, batch, idx_data, batch_idx=None):
        xjs = batch
        loss = - self.elbo(xjs, idx_data).mean(axis=0) # average negative evidence lower bound
        return loss

    def test_step(self, batch, idx_data, batch_idx=None):
        xjs = batch
        if isinstance(self.q, SemiparametricConditionalExpFam) or torch.is_tensor(self.nu):
            return torch.nan # not amortized - would need extra optimization over q^(n) resp. nu_j^(n) !
        with torch.no_grad():
            loss = - self.elbo(xjs, idx_data).mean(axis=0) # average negative evidence lower bound
        return loss


class EmpiricalDistribution(torch.nn.Module):
    """
    Rudimentary representation of a mixture distribution of Dirac delta peaks.
    """
    def __init__(self, x):
        super().__init__()
        self.x = x
        self.N = x.shape[0]

    def eval(self, x):
        return 1.0 if x in self.x else 0.0

    def sample(self):
        i = np.random.choice(self.N, size=1)
        return self.x[i]

class RPMEmpiricalMarginals(torch.nn.Module):
    """
    Placeholder object to represent all marginal data distribution pj(xj), j=1,...,J
    used among others to define a recognition-parametrized model.
    """
    def __init__(self, xjs):
        super().__init__()
        self.J = len(xjs)
        self.N = xjs[0].shape[0]
        assert all([self.N == xjs[j].shape[0] for j in range(self.J)])
        self.pxjs = [EmpiricalDistribution(x=xj) for xj in xjs]

