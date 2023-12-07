from configargparse import ArgParser
import torch
import numpy as np
import os

import rpm
from utils_setup import init_gaussian_rpm

def run_exp_poisson_balls(N, T, K, J, init_rb_bandwidth,
                          rpm_variant, temporal, amortize_ivi,
                          epochs, batch_size, lr,
                          store_results, model_seed, data_seed,
                          optim_init_q=True, optim_init_ivi=True, 
                          optim_vae_params=False,n_hidden=20,
                          normalized_factors=True,
                          iviNatParametrization='classic'):

    store_results = (store_results!=0) # ArgpParser and booleans...
    temporal = (temporal!=0)

    visualize_results = False # script form

    assert rpm_variant in ['VI', 'VAE', 'amortized']
    assert amortize_ivi in ['none', 'use_q', 'full']
    batch_size = int(np.minimum(batch_size, N))
    assert np.all(np.array([N,T,K,J,epochs,batch_size,lr]) > 0)

    if temporal:
        identifier = rpm_variant + '_temp_' + amortize_ivi
    else:
        identifier = rpm_variant + '_' + amortize_ivi
    identifier = identifier + '_N_' + str(N) + '_seed_' + str(model_seed)
    root = os.curdir    
    print('\n')
    print('running exp ' + identifier + ' from directory ' + root)
    print('\n')
    print('(N,T,K,J,batch_size) = ' + str((N,T,K,J,batch_size)))
    #print('initial prior precision matrix log-diagonal: ' + str(init_diag_val))
    #print('initial prior precision matrix off-diagonal factor: ' + str(init_off_val))
    print('initial prior RBF kernel bandwidth:' + str(init_rb_bandwidth))
    
    dim_js = [T for j in range(J)]  # dimensions of marginals
    dim_Z = K*T                     # total dimension of latent
    dim_T = 2                       # sufficient statistics (per latent)

    dtype = torch.float32
    torch.set_default_dtype(dtype)

    #########################################################################
    # Generate data
    #########################################################################

    from utils_data_external import sample_poisson_balls
    from rpm import RPMEmpiricalMarginals

    np.random.seed(data_seed)
    torch.manual_seed(data_seed)

    observations_all, true_latent_all, obs_locs = sample_poisson_balls(
        num_observation=2*N, dim_observation=J, len_observation=T)
    xjs = [observations_all[0][:N,...,j] for j in range(J)]
    xjs_test = [observations_all[0][N:,...,j] for j in range(J)] 
    true_latent_ext = true_latent_all[:N,:]
    true_latent_ext_test = true_latent_all[N:,:]

    #########################################################################
    # Setup for RPM 
    #########################################################################

    np.random.seed(model_seed)
    torch.manual_seed(model_seed)

    pxjs = RPMEmpiricalMarginals(xjs)
    model, ls_q, ls_nu = init_gaussian_rpm(
                      N, J, K, T, pxjs, 
                      init_rb_bandwidth, obs_locs,
                      rpm_variant, temporal, amortize_ivi,
                      epochs//10, batch_size,
                      dim_T, n_hidden, 
                      iviNatParametrization, normalized_factors,
                      optim_init_q, optim_init_ivi, optim_vae_params)

    pxjs_test = RPMEmpiricalMarginals(xjs_test)    
    RPM = rpm.RPMtemp if temporal else rpm.RPM
    test_model = RPM(model.joint_model[0],model.joint_model[1],
                     px=pxjs_test,nu=model.nu,q=model.q)

    #########################################################################
    # Train model
    #########################################################################

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ds = torch.utils.data.TensorDataset(*xjs, torch.arange(N))
    dl = torch.utils.data.DataLoader(dataset=ds, 
                                     batch_size=batch_size, 
                                     shuffle=True, 
                                     drop_last=True)

    print('\n')
    print('fitting model.')
    ls,t,break_flag = np.zeros(epochs*(N//batch_size)),0,False
    ls_test = np.zeros(epochs)
    for i in range(epochs):
        for batch in dl:
            optimizer.zero_grad()
            loss = model.training_step(batch=batch[:-1], 
                                       idx_data=batch[-1], 
                                       batch_idx=t)
            loss.backward()
            optimizer.step()
            ls[t] = loss.detach().numpy()
            t+=1
            if np.isnan(ls[t-1]):
                break_flag = True
                break
        ls_test[i] = test_model.test_step(batch=xjs_test, 
                                          idx_data=torch.arange(N), 
                                          batch_idx=0)
        if break_flag:
            ls[t:] = np.nan
            ls_test[i:] = np.nan
            print('NaN loss, stopping in epoch '+str(i)+'/'+str(epochs))
            break
        if np.mod(i, epochs/10) == 0:
            print('epoch #'+str(i)+'/'+str(epochs)+', train loss='+str(ls[t-1]) + ', test loss=' + str(ls_test[i]))
    print('done fitting.')

    #########################################################################
    # Store results
    #########################################################################

    if store_results:
        import subprocess

        res_dir = 'fits'

        try:
            os.mkdir(os.path.join(root, res_dir, identifier))
        except:
            pass
        fn_base = os.path.join(root, res_dir, identifier, identifier)
        print('\n')
        print('saving results in directory ' + fn_base)
        # get current git commit in case classes change etc.
        fetch_commit = subprocess.Popen(['git', 'rev-parse', 'HEAD'], 
                                        shell=False, 
                                        stdout=subprocess.PIPE)
        git_commit_id = fetch_commit.communicate()[0].strip().decode("utf-8")
        fetch_commit.kill()
        print('current git commit: ' + git_commit_id)

        exp = {
            'T' : T,
            'J' : J,
            'K' : K,
            'N' : N,
            'rpm_variant' : rpm_variant,
            'temporal' : temporal,
            'amortize_ivi' : amortize_ivi,
            'epochs' : epochs,
            'batch_size' : batch_size,    
            'lr' : lr,
            'model_seed' : model_seed,
            'data_seed' : data_seed,
            'init_rb_bandwidth' : init_rb_bandwidth,
            #'init_diag_val' : init_diag_val,
            #'init_off_val' : init_off_val,
            'git_commit_id' : git_commit_id,
            'optim_init_q' : optim_init_q, 
            'optim_init_ivi' : optim_init_ivi, 
            'optim_vae_params' : optim_vae_params
        } 
        np.savez(fn_base + '_exp_dict', exp)

        np.save(fn_base + '_loss_train', ls)
        np.save(fn_base + '_loss_test', ls_test)
        np.save(fn_base + '_loss_pretrain_q', ls_q)
        np.save(fn_base + '_loss_pretrain_nu', ls_nu)

        np.save(fn_base + '_train_data', torch.stack(xjs, dim=1).detach().numpy())
        np.save(fn_base + '_test_data', torch.stack(xjs_test, dim=1).detach().numpy())
        np.save(fn_base + '_train_latents', true_latent_ext.detach().numpy())
        np.save(fn_base + '_test_latents', true_latent_ext_test.detach().numpy())

        torch.save(model.state_dict(), fn_base + '_rpm_state_dict')
        torch.save(optimizer.state_dict(), fn_base + '_optimizer_state_dict')
        print('done saving results.')


    #########################################################################
    # Evaluate model & visualize results
    #########################################################################

    if visualize_results: 

        from utils_data_external import linear_regression_1D_latent as regLatent
        from utils_data_external import plot_poisson_balls
        import matplotlib.pyplot as plt

        prior = model.joint_model[1]
        eta_0 = prior.nat_param
        if rpm_variant == 'amortized':
            eta_q, _ = model.comp_eta_q(xjs, eta_0)
        else: 
            eta_q = model.comp_eta_q(xjs, idx_data=np.arange(N), eta_0=eta_0)
        EqtZ = prior.log_partition.nat2meanparam(eta_q)

        mu = EqtZ[:,:T]
        sig2 = torch.diagonal(EqtZ[:,T:].reshape(-1,T,T),dim1=-2,dim2=-1) - mu**2

        latent_true, latent_mean_fit, latent_variance_fit, R2 = regLatent(
            latent_true = true_latent_ext,
            latent_mean_fit = mu.unsqueeze(-1), 
            latent_variance_fit = sig2)

        plt.plot(ls)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()

        plot_poisson_balls(observations, 
                           obs_locs=obs_locs.squeeze(-1), 
                           latent_mean_fit=latent_mean_fit.squeeze(-1), 
                           latent_variance_fit=latent_variance_fit)


def setup_poisson_balls(conf_exp=None):
    p = ArgParser()
    p.add_argument('-c', '--conf-exp', is_config_file=True, help='config file path', default=conf_exp)
    p.add_argument('-store_results', type=int, default=1, help='boolean (1=True, 0=False), if to store results of experiment.')

    p.add_argument('--model_seed', type=int, required=True, help='random seed for experiment')
    p.add_argument('--N', type=int, required=True, help='number of training data points')
    p.add_argument('--rpm_variant', type=str, required=True, help='RPM training variant: VI, VAE, amortized')
    p.add_argument('--amortize_ivi', type=str, required=True, help='method for innver variational bound parameters')
    p.add_argument('--temporal', type=int, default=0, help='boolean (1=True, 0=False), if to use temporally structured RPM')
    p.add_argument('--data_seed', type=int, default=0, help='random seed for experiment')
    p.add_argument('--T', type=int, default=50, help='number of time points')
    p.add_argument('--J', type=int, default=10, help='number of cond.indep. marginals')
    p.add_argument('--K', type=int, default=1, help='dimensionality of latents per time point')
    p.add_argument('--init_rb_bandwidth', type=float, default=1000.0, help='initial bandwidth for RB kernel function on prior cov')
    #p.add_argument('--init_diag_val', type=float, default=0.4, help='initial log-diagonal for tri-diagonal prior precision matrix')
    #p.add_argument('--init_off_val', type=float, default=-1.0, help='initial tanh-scaling for off-diagonal of prior precision matrix')
    p.add_argument('--n_hidden', type=int, default=20, help='number of units per hidden layer in three-layer networks')
            
    p.add_argument('--batch_size', type=int, default=8, help='batch-size')
    p.add_argument('--epochs', type=int, default=2000, help='epochs')
    p.add_argument('--lr', type=float, default=1e-3, help='learning rate for ADAM')

    args = p.parse_args() if conf_exp is None else p.parse_args(args=[])
    return vars(args)