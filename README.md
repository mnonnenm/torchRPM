# RPM implementation in pytorch

Development repository of Marcel Nonnenmacher on the exponential family recognition-parametrised model (RPM) [1]
written in 2023 while working with Maneesh Sahani at the Gatsby Computational Neuroscience Unit.

## Contents

The main model variants covered by this code base are:
- discrete latent variables, i.e. $p(Z), f(Z|x_j)$ are categorical. 
- continous latent-variable from a fixed exponential family.
- unnormalized recognition factors $f(Z|x_j) = e^{g(Z,x_j)}$, where $g_j(Z,x_j) = g_j(x_j)'t(Z) + h_j(x_j)$ for some $h_j$ other than the negative log-partition function of exponential-family $p(Z)$.
- a standard and a time-series version of the RPM (similar to the latent GP from [1], but with discrete time). Time-series variants rely strongly on marginalization properties of multivariate densities $f(Z_{t=1,..,T}|xj)$ to have tractable marginals $f(Z_t|xj)$, so they are largely restricted to Gaussian latents.
- RP-VAEs and time-series verions: Reparametrized variant of RPM with amortized $q(Z|X)$ that assumes $F_j(Z) = p(Z)$ only in $q$ (i.e. not in the implicit likelihood $p(x_j|Z)$), which not only makes $q$ tractable and computable given the current generative model, but also simplifies the ELBO.  
- a tested and subsequently abandoned RPM version  (implicitRPM.py) directly defining $\omega(x) = \int p(x) \prod_j f_j(Z|x_j)/F_j(Z) dZ$ under an assumption that Fj(Z)=p(Z) which I hoped to make 'less false' by encouraging model marginals $p(xj)$ to match the data. It later turned out that matching $x_j$-marginals will enforce $F_j(Z)=p(Z)$ (and hence make the assumption true) only in the case dim($\eta_j(x_j)$) <= dim($x_j$), so the intended use-case for implicit DDC won't work.
- the case of continous latent variables generally has intractable ELBO, and this repository implementents the `inner variational bound' (lower bound to ELBO) of [1].
  
Expermiments: main experiments imlemented so far include 
- the bouncing balls examples from the RPM paper [1] that can be found in utils_data_external.py, 
- a two-dimensional Gaussian copula with 1D marginals (either Gaussian or exponentially distributed), see notebooks.
- several variants of the peer-supervision task on MNIST used for testing discrete RPMs, see notebooks. 



## Scope

The main focus of this work was 
- to experiment with the RPM itself and understand its core functioning (i.e. discretization of observed space under cond. independence),                                   
- to implement various fitting methods that go beyond the basic saturated VI of the original AISTATS publication [1], such as minibatch SGD, amortized q(Z|X) etc. 
- to write the RPM in general exponential family form, in constrast to the specific implementations for the two special cases of Gaussian (process) $p(Z), f(Z|x_j), q(Z|X)$, and discrete latents covered in [1].

In the process of the above, we realized several new aspects of the recognition-parametrised model, such that 
- recognition factors need not be normalized -- the argument for their shape as being log-affine in sufficient statistics $t(Z)$ of the latent prior is indeed a conjugacy argument!
- such conjugacy can indeed be learned by matching the data distribution via maximum likelihood,  
- a reparametrization of the RPM recognition model can substantially speed up variational EM.

This repository is a personal copy of my original development repository that got transfered to the 'Gatsby-Sahani' github organization.

[1] Walker, William I., et al. "Unsupervised representation learning with recognition-parametrised probabilistic models." International Conference on Artificial Intelligence and Statistics. PMLR, 2023.
