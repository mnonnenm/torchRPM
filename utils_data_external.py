import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
from scipy.stats import norm, bernoulli

def generate_2D_latent(T, F, omega, z0, noise=0.0):
    """ code taken from AISTATS publication, found at 
    https://github.com/gatsby-sahani/rpm-aistats-2023/blob/10533e70cb96bfb18f2463a16913de2aaacc25e1/AISTATS2023/RPGPFA/utils.py#L215
    """
    # Generate a 2D oscillation

    # Number of Time point
    L = T * F

    # Number of trajectories
    N = z0.shape[0]

    # Time Vector
    t = np.arange(L) / F

    # Rotation angle
    Omega = torch.tensor([2*np.pi * omega / F])

    # Rotation Matrix
    rotation = torch.tensor(
        [[torch.cos(Omega), -torch.sin(Omega)],
         [torch.sin(Omega), torch.cos(Omega)]])
    zt = torch.zeros(N, L + 1, 2)

    noise_mvn = MultivariateNormal(torch.zeros(2),
                                   (noise+1e-20) * torch.eye(2))

    # Loop over init
    for n in range(N):

        # Init
        zt[n, 0] = z0[n]

        # Loop Over time point
        for tt in range(L):
            zc = zt[n, tt]
            zz = torch.matmul(rotation, zc)

            if noise>0:
                zz += 0*noise_mvn.sample()

            zt[n, tt+1] = zz

    return zt, t


def sample_poisson_balls(num_observation, dim_observation, len_observation, 
                         F=10, omega=0.5, vari_th=1, scale_th=0.6, base_rate=10):
    """ code adapted from AISTATS publication, found at 
    https://github.com/gatsby-sahani/rpm-aistats-2023/blob/main/AISTATS2023/RPGPFA/demo_rp_gpfa_poisson_ball.ipynb
    """

    # Length of Each sample [sec]
    T_ = int(len_observation / F)

    # Random initializations
    theta = 2*np.pi*np.random.rand(num_observation)
    z0 = torch.tensor(np.array([np.cos(theta), np.sin(theta)]).T)
    zt, _ = generate_2D_latent(T_, F, omega, z0)

    # True Latent
    true_latent_ext = zt[:, 1:, 0] .unsqueeze(-1)

    # From Latent Position to Pixel Rate
    mean_rate_loc = torch.linspace(-1, 1, dim_observation).unsqueeze(0).unsqueeze(0)
    rate_model = (vari_th**2) * torch.exp(-(mean_rate_loc - true_latent_ext)**2 / scale_th**2)

    observation_locations = torch.linspace(0, 1, len_observation).unsqueeze(-1)

    # Sample Observations
    observations = (torch.poisson(base_rate*rate_model),)
    
    return observations, true_latent_ext, observation_locations


def sample_textured_balls(num_observation, dim_observation, len_observation, 
                          scale_th, sigma2, shape_max_0, F=10, omega=0.5,dtype=torch.float32):
    """ code adapted from AISTATS publication, found at 
    https://github.com/gatsby-sahani/rpm-aistats-2023/blob/main/AISTATS2023/RPGPFA/demo_rp_gpfa_textured_ball..ipynb
    """
    
    # Length of Each sample [sec]
    T_ = int(len_observation / F)

    # Random initializations
    theta = 2*np.pi*np.random.rand(num_observation)
    z0 = torch.tensor(np.array([np.cos(theta), np.sin(theta)]).T)
    zt, _ = generate_2D_latent(T_, F, omega, z0)

    # True Latent
    true_latent = zt[:, 1:, 0] .unsqueeze(-1)

    # Max and min value of the latent
    latent_max = true_latent.max()
    latent_min = true_latent.min()

    # Build Observation from 1st latent
    pixel_loc = torch.linspace(latent_min, latent_max, dim_observation).unsqueeze(0).unsqueeze(0)

    # Distance Pixel - Ball
    distance_pixel_ball = (torch.exp(-(pixel_loc - true_latent) ** 2 / scale_th ** 2)).numpy()

    # From Rate to shape parameter
    shape_min = np.sqrt(1 - sigma2)
    shape_max = shape_max_0 - shape_min
    shape_parameter = shape_max * distance_pixel_ball + shape_min

    # From shape to samples
    loc0 = shape_parameter
    var0 = np.ones(shape_parameter.shape) * sigma2

    # Bernouilli Parameter
    ber0 = (1 - var0) / (1 - var0 + loc0 ** 2)

    # Mean of the First Peak
    loc1 = loc0

    # Mean of The Second Peak
    loc2 = - loc1 * ber0 / (1 - ber0)

    # Bernouilli Sample
    pp = bernoulli.rvs(ber0)

    # Assign to one distribution
    loc_cur = pp * loc1 + (1 - pp) * loc2

    # Sample from the distribution
    observation_samples = norm.rvs(loc_cur, np.sqrt(var0))

    observation_locations = torch.linspace(0, 1, len_observation).unsqueeze(-1)

    return (torch.tensor(observation_samples,dtype=dtype),), true_latent, observation_locations


def diagonalize(z):
    """ code taken from AISTATS publication, found at
    https://github.com/gatsby-sahani/rpm-aistats-2023/blob/10533e70cb96bfb18f2463a16913de2aaacc25e1/AISTATS2023/RPGPFA/utils.py#L177C1-L180C13
    """
    Z = torch.zeros((*z.shape, z.shape[-1]), device=z.device, dtype=z.dtype)
    Z[..., range(z.shape[-1]), range(z.shape[-1])] = z
    return Z


def linear_regression_1D_latent(latent_true, latent_mean_fit, latent_variance_fit, inducing_mean=None):
    """ code adapted from AISTATS publication, found at
    https://github.com/gatsby-sahani/rpm-aistats-2023/blob/10533e70cb96bfb18f2463a16913de2aaacc25e1/AISTATS2023/RPGPFA/utils_process.py#L170
    """

    # Dimension of the problem
    num_observation, len_observation, dim_latent = latent_true.shape
    dim_latent_fit = latent_mean_fit.shape[-1]
    shape_true_cur = (num_observation, len_observation, dim_latent)
    shape_true_tmp = (num_observation * len_observation, dim_latent)

    shape_fit_cur = (num_observation, len_observation, dim_latent_fit)
    shape_fit_tmp = (num_observation * len_observation, dim_latent_fit)

    # This linear regression removes degeneraciers only in the 1D case
    assert dim_latent == 1
    #assert dim_latent_fit == 1

    # Reshape And Diagonalize
    latent_true = latent_true[:num_observation].reshape(shape_true_tmp)
    latent_mean_fit = latent_mean_fit.reshape(shape_fit_tmp)
    latent_variance_fit = latent_variance_fit.reshape(shape_fit_tmp)

    # Recenter Rescale True Latent
    latent_true -= latent_true.mean()
    latent_true /= latent_true.max()

    # Recenter Fit
    mean0 = latent_mean_fit.mean(dim=0, keepdim=True)
    latent_mean_fit -= mean0

    # Renormalise Latent
    norm0 = latent_mean_fit.abs().max(dim=0)[0].unsqueeze(0)
    latent_mean_fit /= norm0
    latent_variance_fit /= norm0 ** 2

    # Linear Regression
    matmul, inv = torch.matmul, torch.linalg.inv
    Id = diagonalize(torch.ones(dim_latent_fit))
    beta_lr = matmul(inv(matmul(latent_mean_fit.transpose(dim1=-1, dim0=-2), latent_mean_fit) + 0.01 * Id),
                   matmul(latent_mean_fit.transpose(dim1=-1, dim0=-2), latent_true))

    latent_mean_fit = matmul(latent_mean_fit, beta_lr)
    latent_variance_fit = matmul(beta_lr.transpose(dim1=-1, dim0=-2), matmul(diagonalize(latent_variance_fit), beta_lr)).squeeze(-1)

    # norm1 = 1 / matmul(inv(matmul(latent_mean_fit.transpose(dim1=-1, dim0=-2), latent_mean_fit)),
    #                matmul(latent_true.transpose(dim1=-1, dim0=-2), latent_mean_fit))

    # # Renormalise Latent
    # latent_mean_fit /= norm1
    # latent_variance_fit /= norm1 ** 2

    # R square value
    R2 = 1 - ((latent_mean_fit - latent_true)**2).sum() / ((latent_true)**2).sum()

    # Reshape all
    latent_true = latent_true.reshape(shape_true_cur)
    latent_mean_fit = latent_mean_fit.reshape(shape_true_cur)
    latent_variance_fit = latent_variance_fit.reshape(shape_true_cur)

    return latent_true, latent_mean_fit, latent_variance_fit, R2


def plot_poisson_balls(observations, obs_locs=None, latent_mean_fit=None, latent_variance_fit=None ):

    N,T,J = observations[0].shape
    maxfigs = 5 # maxmimum plots if N is large

    # Colors
    color_position = torch.cat((
        torch.linspace(0, 0, J).unsqueeze(1),
        torch.linspace(0, 1, J).unsqueeze(1),
        torch.linspace(0, 0, J).unsqueeze(1)),
        dim=1)

    # Select which observation to plot
    plot_index = np.arange( np.minimum(N,maxfigs))

    plt.figure(figsize=(5 * len(plot_index), 3 * 2))
    # Plot Observations
    for egobs_id in range(len(plot_index)):

        obs_cur = observations[0][egobs_id]
        xx = np.arange(obs_cur.shape[0]) / obs_cur.shape[0]

        egobs = plot_index[egobs_id]

        plt.subplot(2, len(plot_index), egobs_id + 1)
        plt.imshow(obs_cur.transpose(-1, -2), aspect='auto', origin='lower', cmap='gray',extent=[0, 1, -1, 1])
        plt.title('Observation ' + str(egobs+1) + '/' + str(N))
        plt.yticks([])

        if not latent_mean_fit is None:
            assert not obs_locs is None
            assert len(obs_locs) == latent_mean_fit.shape[-1]
            plt.plot(obs_locs.detach().numpy(), 
                     latent_mean_fit[egobs_id].detach().numpy(), 
                     'y', linewidth='2.5')

        plt.subplot(2, len(plot_index), egobs_id + 1 + len(plot_index))
        for dd in range(J):
            plt.scatter(xx, obs_cur[:, dd], color=color_position[dd].numpy(), label=str(dd))
        if egobs_id == 0:
            plt.legend(title="Pixel id.", prop={'size': 7}, loc=1)
            plt.ylabel('Count')
        plt.xlabel('Time [a.u]')
