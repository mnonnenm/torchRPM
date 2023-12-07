import torch
import numpy as np
import copy

from expFam import SemiparametricConditionalExpFam, ConditionalExpFam, LogPartition_gauss, LogPartition_gauss_diagonal
import expFam
import recognition_nets
import rpm


def pre_train_q(q, rec_factors, xjs, epochs, batch_size, temporal=False, eta_0=None):

    N, J = len(xjs[0]), len(rec_factors)
    eta_j_base = torch.stack([m(xj) for m,xj in zip(rec_factors, xjs)],axis=1)
    target = eta_j_base.sum(axis=1).detach()

    if isinstance(q, ConditionalExpFam):
        x = torch.stack(xjs, dim=1)
        def prediction(x):
            return q(x)
    elif isinstance(q, SemiparametricConditionalExpFam):
        x = torch.arange(N)
        def prediction(x):
            return q.nat_param(nat_param_offset=0.)[x]
    else:
        raise NotImplementedError()

    optimizer = torch.optim.Adam(q.parameters(), lr=1e-3)
    loss_fun = torch.nn.MSELoss()

    ds = torch.utils.data.TensorDataset(x, target)
    dl = torch.utils.data.DataLoader(dataset=ds, 
                                     batch_size=batch_size, 
                                     shuffle=True, 
                                     drop_last=True)

    print('\n')
    print('fitting model.')
    ls,t,break_flag = np.zeros(epochs*(N//batch_size)),0,False
    for i in range(epochs):
        for batch in dl:
            optimizer.zero_grad()
            loss = loss_fun(prediction(batch[0]),batch[1])
            loss.backward()
            optimizer.step()
            ls[t] = loss.detach().numpy()
            t+=1
            if np.isnan(ls[t-1]):
                break_flag = True
                break
        if break_flag:
            ls[t:] = np.nan
            print('NaN loss, stopping at epoch '+str(i)+'/'+str(epochs))
            break
        if np.mod(i, epochs/10) == 0:
            print('epoch #'+str(i)+'/'+str(epochs)+', loss='+str(ls[t-1]))
    print('done fitting.')
    return ls

def pre_train_ivi(nu, rec_factors, xjs, epochs, batch_size, temporal=False, eta_0=None, iviNatParametrization='classic'):

    N, J = len(xjs[0]), len(rec_factors)
    eta_j_base = torch.stack([m(xj) for m,xj in zip(rec_factors, xjs)],axis=1)
    if temporal:
        assert not eta_0 is None
        dim_Z = int(np.sqrt(4*eta_j_base.shape[-1]+1)/2.-0.5)
        full2diag_gaussian = expFam.LogPartition_gauss(d=dim_Z).full2diag_gaussian
        if iviNatParametrization == 'classic':
            eta_q = eta_0 + eta_j_base.sum(axis=1)
            eta_q_diag, _ = full2diag_gaussian(eta_q)
            eta_0_diag, _ = full2diag_gaussian(eta_0)
            target = (eta_q_diag - eta_0_diag).detach().unsqueeze(1).expand(-1,J,-1,-1)
            dim_Z = eta_0.shape[-2]
            full2diag_gaussian = expFam.LogPartition_gauss(d=dim_Z).full2diag_gaussian
        elif iviNatParametrization == 'delta':
            target = -0.1*torch.ones(N, J, dim_Z, 2) # small precisions - not zero due to inv(Softplus)!
            target[...,0] = 0.0                      # zero means
    else:
        if iviNatParametrization == 'classic':
            target = eta_j_base.sum(axis=1).detach().unsqueeze(1).expand(-1,J,-1)
        elif iviNatParametrization == 'delta':
            T = int(np.sqrt(4*eta_0.shape[-1]+1)/2.-0.5)
            target = -0.1*torch.ones(N, J, T*(T+1)) # small precisions - not zero due to inv(Softplus)!
            target[...,:T] = 0.0                    # zero means
        
    if np.all([isinstance(nu[j], ConditionalExpFam) for j in range(J)]):
        x = torch.stack(xjs, dim=1)
        if temporal:
            def prediction(x):
                return torch.stack([m(x) for m in nu], axis=1)
        else:
            def prediction(x):
                return torch.stack([m(x) for m in nu], axis=1)
    elif torch.is_tensor(nu):
        x = torch.arange(N)
        def prediction(x):
            return nu[x]
    else:
        raise NotImplementedError()

    idx = np.arange(8)

    optimizer = torch.optim.Adam(torch.nn.ModuleList(nu).parameters(), lr=1e-3)
    loss_fun = torch.nn.MSELoss()

    ds = torch.utils.data.TensorDataset(x, target)
    dl = torch.utils.data.DataLoader(dataset=ds, 
                                     batch_size=batch_size, 
                                     shuffle=True, 
                                     drop_last=True)

    print('\n')
    print('fitting model.')
    ls,t,break_flag = np.zeros(epochs*(N//batch_size)),0,False
    for i in range(epochs):
        for batch in dl:
            optimizer.zero_grad()
            loss = loss_fun(prediction(batch[0]),batch[1])
            loss.backward()
            optimizer.step()
            ls[t] = loss.detach().numpy()
            t+=1
            if np.isnan(ls[t-1]):
                break_flag = True
                break
        if break_flag:
            ls[t:] = np.nan
            print('NaN loss, stopping at epoch '+str(i)+'/'+str(epochs))
            break
        if np.mod(i, epochs/10) == 0:
            print('epoch #'+str(i)+'/'+str(epochs)+', loss='+str(ls[t-1]))
    print('done fitting.')
    return ls


def pretrain_vparams_elbo(model, epochs, batch_size, lr):

    xjs, N = [pxj.x for pxj in model.joint_model[2].pxjs], model.N

    optimizers = []
    if not model.q == 'use_theta': 
        optimizers.append(torch.optim.Adam(model.q.parameters(), lr))
    if not model.nu is None:
        if torch.is_tensor(model.nu):            
            optimizers.append(torch.optim.Adam((model.nu,), lr))
        else:
            optimizers.append(torch.optim.Adam(torch.nn.ModuleList(model.nu).parameters(), lr))

    print('\n')
    print('pretraining variational parameters q(Z|X), nu(Z|X), using variational bound.')
    ls,t,break_flag = np.zeros(epochs*(N//batch_size)),0,False

    if optimizers == []:
        print('done fitting.')
        return ls

    ds = torch.utils.data.TensorDataset(*xjs, torch.arange(N))
    dl = torch.utils.data.DataLoader(dataset=ds, 
                                     batch_size=batch_size, 
                                     shuffle=True, 
                                     drop_last=True)
    for i in range(epochs):
        for batch in dl:
            for optimizer in optimizers:
                optimizer.zero_grad()
            loss = model.training_step(batch=batch[:-1], 
                                       idx_data=batch[-1], 
                                       batch_idx=t)
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            ls[t] = loss.detach().numpy()
            t+=1
            if np.isnan(ls[t-1]):
                break_flag = True
                break
        if break_flag:
            ls[t:] = np.nan
            print('NaN loss, stopping in epoch '+str(i)+'/'+str(epochs))
            break
        if np.mod(i, epochs/10) == 0:
            print('epoch #'+str(i)+'/'+str(epochs)+', train loss='+str(ls[t-1]))
    print('done fitting.')
    

def init_gaussian_rpm(N, J, K, T, pxjs,
                      init_rb_bandwidth, obs_locs,
                      rpm_variant, temporal, amortize_ivi,
                      epochs, batch_size,
                      dim_T=2, n_hidden=20, 
                      iviNatParametrization='classic',
                      normalized_factors=True,
                      optim_init_q=True, optim_init_ivi=True, optim_vae_params=False):

    dim_Z = K*T # total latent dimensionality
    assert rpm_variant in ['VI', 'VAE', 'amortized']
    assert amortize_ivi in ['none', 'use_q', 'full']
    assert iviNatParametrization in ['classic', 'delta']
    xjs = [pxj.x for pxj in pxjs.pxjs]

    # define correlated Gaussian prior in natural parametrization
    log_partition_prior = LogPartition_gauss(d=dim_Z)

    gamma = torch.nn.Parameter(torch.tensor(init_rb_bandwidth)) 
    eta_0 = expFam.NatParamGauss_RBKernel(
        mu=torch.zeros((1,len(obs_locs))), 
        gamma=gamma, 
        ts=obs_locs.T
        )
    #diag_val_base = torch.nn.Parameter(torch.tensor(init_diag_val))
    #off_val_base = torch.nn.Parameter(torch.tensor(init_off_val))
    #eta_0 = expFam.NatParamGauss_TriDiagonal(mu = torch.zeros(1,T), 
    #                                         diag_val_base=diag_val_base,
    #                                         off_val_base=off_val_base)
    latent_prior = expFam.ExpFam_parametrizedParam(
        natparam=eta_0, 
        log_partition=log_partition_prior, 
        activation_out=torch.nn.Identity()
    )
    eta_0 = latent_prior.nat_param.detach()

    # define Gaussian factors fj(Z|xj) in natural parametrization
    log_partition = LogPartition_gauss(d=dim_Z)
    Net = recognition_nets.Net_3xConv1D
    def activation_out(x):
        return recognition_nets.activation_gauss_diag_full_conv1d(x,d=K)
    def get_rec_net():
        return Net(1, dim_T, n_hidden=n_hidden, activation_out=activation_out)
    if normalized_factors:
        def get_rec_model():
            m = get_rec_net()
            return ConditionalExpFam(model=m, log_partition=log_partition)
    else:
        class Net_3xfc(torch.nn.Module):
            def __init__(self, n_in, n_out, n_hidden, activation_out=torch.nn.Identity()):
                super(Net_3xfc, self).__init__()
                self.activation_out = activation_out
                self.fc1 = torch.nn.Linear(n_in, n_hidden, bias=True)
                self.fc2 = torch.nn.Linear(n_hidden, n_hidden, bias=True)
                self.fc3 = torch.nn.Linear(n_hidden, n_out, bias=True)
            def forward(self, x):
                x = torch.nn.functional.relu(self.fc1(x))
                x = torch.nn.functional.relu(self.fc2(x))
                x = self.fc3(x)
                return self.activation_out(x)

        class LogPartWrapper(torch.nn.Module):
            """
            Base class for log-partition functions in natural parametrization. 
            """
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, eta):
                return self.model(eta).squeeze(-1)
            def marginal_log_partition(self, eta):
                return self(eta.reshape(-1,dim_T)).reshape(eta.shape[:-1])

        def get_hj():
            net = Net_3xfc(dim_T, 1, n_hidden=n_hidden)
            return LogPartWrapper(net)
        def get_rec_model():
            m = get_rec_net()
            lp = get_hj() # code actually never checks if log_partition is valid !
            return ConditionalExpFam(model=m, log_partition=lp)
        print("-  initializing model with *non*-normalized factors f(Z|xj) !")
    rec_factors = [get_rec_model() for j in range(J)]

    # define recognition model q(Z|X)
    ls_q = None
    if rpm_variant == 'VI':
        eta_q_base = torch.normal(mean=0.0, std=torch.ones(N, dim_T, dim_Z))
        q = expFam.SemiparametricConditionalExpFam(
            natparams=eta_q_base, 
            log_partition=log_partition, 
            activation_out=activation_out
        )
        if optim_init_q:
            print('pre-training q(Z|X) to match \sum_j etaj(xj) + (1-J)eta0')
            ls_q = pre_train_q(q, rec_factors, xjs, J*epochs, batch_size,
                               temporal, eta_0)
        eta_q_init = activation_out(eta_q_base) # + eta0
    elif rpm_variant == 'VAE':
        m_q =  Net(J, dim_T, n_hidden=n_hidden, activation_out=activation_out)
        q = ConditionalExpFam(model=m_q, log_partition=log_partition)
        if optim_init_q:
            print('pre-training q(Z|X) to match \sum_j etaj(xj) + (1-J)eta0')
            ls_q = pre_train_q(q, rec_factors, xjs, J*epochs, batch_size,
                               temporal, eta_0)
        eta_q_init = q.nat_param(torch.stack(xjs,dim=1)) # + eta0
    elif rpm_variant == 'amortized':
        if optim_init_q:
            print('- not optimizing initial q since q is analytic given RPM!')
        q, ls_q = 'use_theta', None # handled internally (different loss for this case)
        eta_j_base = torch.stack([m(pxj.x) for m,pxj in zip(rec_factors, pxjs.pxjs)],axis=1)
        eta_q_init = eta_j_base.sum(axis=1) # + (1-J)eta0
    else:
        print('rpm_variant', rpm_variant)
        print('temporal', temporal)
        raise NotImplementedError()

    # define parameters for inner variational bound
    ls_nu = None
    if temporal :
        activation_nu = recognition_nets.activation_gauss_diag_diag_conv1d
        log_partition_nu = LogPartition_gauss_diagonal(d=dim_Z)
    else:
        activation_nu = recognition_nets.activation_gauss_diag_full_conv1d      
        log_partition_nu = LogPartition_gauss(d=dim_Z)
    def activation_out_nu(x):
        return activation_nu(x,d=K)
    if amortize_ivi == 'none': # non-parametric: one parameter per datapoint
        nu_base = torch.normal(mean=0.0, std=0.0*torch.ones(N, J, dim_T, dim_Z))
        inv_activation=recognition_nets.activation_inv_gauss_diag_full_conv1d
        if optim_init_ivi:
            print('choosing initial values for ivi parameters to match q')
            if iviNatParametrization == 'classic':
                nu_target = eta_q_init.detach()
            elif iviNatParametrization == 'delta':
                nu_target = -0.1*torch.ones(N, K*T*(K*T+1)) # small precisions - not zero due to inv(Softplus)!
                nu_target[:,:K*T] = 0.0                     # zero means
            nu_base = inv_activation(nu_target,d=K)
            nu_base = torch.normal(mean=nu_base.unsqueeze(1).expand((-1,J,-1,-1)),
                                   std=0. * torch.ones((N, J, dim_T, dim_Z))/1000.)
        nu = torch.nn.parameter.Parameter(activation_out_nu(nu_base))
    elif amortize_ivi == 'full': # train separate networks for \tilde{eta}j(X)
        if rpm_variant == 'VAE':
            #nu = [copy.deepcopy(q) for J in range(J)]
            m_nu =  copy.deepcopy(m_q)
            m_nu.activation_out = activation_out_nu
            nu = [ConditionalExpFam(model=copy.deepcopy(m_nu), log_partition=log_partition_nu) for J in range(J)]            
            if optim_init_ivi and iviNatParametrization == 'delta':
                print('pre-training ivi parameters to match q(Z|X)')
                ls_nu = pre_train_ivi(nu, rec_factors, xjs, epochs, batch_size,
                                      temporal, eta_0, iviNatParametrization)
        elif rpm_variant in ['VI', 'amortized']:
            def get_ivi_net():
                return Net(J, dim_T, n_hidden=n_hidden, activation_out=activation_out_nu)
            def get_ivi_model():
                m = get_ivi_net()
                return ConditionalExpFam(model=m, log_partition=log_partition_nu)
            nu = [get_ivi_model() for J in range(J)]
            if optim_init_ivi:
                print('pre-training ivi parameters to match q(Z|X)')
                ls_nu = pre_train_ivi(nu, rec_factors, xjs, epochs, batch_size,
                                      temporal, eta_0, iviNatParametrization)
    elif amortize_ivi == 'use_q': # Ansatz: use natparam of q(Z|X) for all j  
        if optim_init_ivi:
            print('- not optimizing ivi parameters because they match q(Z|X) by design.')
        nu = None

    # instantiate appropriate RPM model class
    RPM = rpm.RPMtemp if temporal else rpm.RPM
    model = RPM(rec_factors,latent_prior=latent_prior,px=pxjs,nu=nu,q=q,iviNatParametrization=iviNatParametrization)

    if optim_vae_params: 
        pretrain_vparams_elbo(model, 2*epochs, batch_size, lr=1e-3)

    return model, ls_q, ls_nu