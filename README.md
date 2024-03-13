# RPM implementation in pytorch

Development repository of Marcel Nonnenmacher on the exponential family [recognition-parametrised model](https://proceedings.mlr.press/v206/walker23a.html) [1] while working with Maneesh Sahani at the Gatsby Computational Neuroscience Unit in 2023.

## Contents

The main model variants covered by this code base are:
- discrete latent variables, i.e. $p(Z), f(Z|x_j)$ are categorical. 
- continous latent-variable from a fixed exponential family $p(Z|\eta) \propto \chi(Z) e^{\eta^\top{}t(Z)}$. [Several exponential families](https://github.com/mnonnenm/torchRPM/blob/main/expFam.py) implemented, and more can be added easily.
- a [standard version](https://github.com/mnonnenm/torchRPM/blob/ceac9b6e1c79ca3c2be1dfd0363b411be73f5906/rpm.py#L8) and a [time-series version](https://github.com/mnonnenm/torchRPM/blob/ceac9b6e1c79ca3c2be1dfd0363b411be73f5906/rpm.py#L247) of the RPM (similar to the latent GP from [1], but with discrete time). Time-series variants rely strongly on multivariate densities $f(Z_{t=1,..,T}|xj)$ to have tractable marginals $f(Z_t|xj)$, so they are largely restricted to Gaussian latents.
- reparametrized continuous-latent RPM variant with amortized $q(Z|X)$ assuming $F_j(Z) = p(Z)$ in $q$ (i.e. not in the implicit likelihood $p(x_j|Z)$). Makes $q$ tractable and computable given the current generative model, simplifies the ELBO and can greatly speed up non-amortized inference. Use options ``iviNatParametrization='delta'`` and pass as recognition model ``q='use_theta'`` to a (standard or temporal) continuous-latent RPM.
- the case of continous latent variables generally has intractable ELBO, and this repository implementents the `inner variational bound' (lower bound to ELBO) of [1]. Allows [saturated recognition models](https://github.com/mnonnenm/torchRPM/blob/ceac9b6e1c79ca3c2be1dfd0363b411be73f5906/expFam.py#L139) $q(Z|X^n) = q(Z|\eta^n)$ and [amortized recognition models](https://github.com/mnonnenm/torchRPM/blob/ceac9b6e1c79ca3c2be1dfd0363b411be73f5906/expFam.py#L163) $q(Z|X^n) = q(Z|\eta(X^n))$ with learned deep network-based $\eta(X)$.
- [unnormalized recognition factors](https://github.com/mnonnenm/torchRPM/blob/950deec56cc8a3e5e10dcd3df722b15cbc7abff6/discreteRPM.py#L360) $f(Z|x_j) = e^{g(Z,x_j)}$, where $g_j(Z,x_j) = g_j(x_j)'t(Z) + h_j(x_j)$ for some $h_j$ other than the negative log-partition function of exponential-family $p(Z)$.
- further RPM variants implemented for research purposes (e.g. an implitic RPM directly defining $\omega(x) = \int p(Z) \prod_j f_j(Z|x_j)/F_j(Z) dZ$ under an assumption that $F_j(Z)=p(Z)$ while trying to enforce $p(x_j)$ to match the data).
  
Expermiments: main experiments imlemented so far include 
- the bouncing balls examples from the RPM paper [1] that can be found in [utils_data_external.py](https://github.com/mnonnenm/torchRPM/blob/main/utils_data_external.py), 
- a two-dimensional Gaussian copula with 1D marginals (either Gaussian or exponentially distributed), see [notebooks](https://github.com/mnonnenm/torchRPM/tree/main/notebooks).
- several variants of the peer-supervision task on MNIST [1] used for testing discrete RPMs, see [notebooks](https://github.com/mnonnenm/torchRPM/tree/main/notebooks). 



## Scope

The main focus of this work was 
- to experiment with the RPM itself and understand its core functioning (i.e. discretization of observed space under cond. independence),                                   
- to implement various fitting methods that go beyond the basic saturated VI of the original AISTATS publication [1], such as minibatch SGD (option ``full_N_for_Fj=False`` for RPM() class), amortized q(Z|X) etc. 
- to write the RPM in general exponential family form, in constrast to the specific implementations for the two special cases of Gaussian (process) $p(Z), f(Z|x_j), q(Z|X)$, and discrete latents covered in [1].

In the process of the above, we realized several new aspects of the recognition-parametrised model, such that 
- recognition factors need not be normalized -- the argument for their shape as being log-affine in sufficient statistics $t(Z)$ of the latent prior is indeed a conjugacy argument!
- such conjugacy can indeed be learned by matching the data distribution via maximum likelihood,  but equivalence between matching the marginals and $F_j(Z)=p(Z)$ only hold for dim($x_j$) $>$ dim(t(Z)),
- a reparametrization of the RPM recognition model based on a locally made assumption of $F_j(Z)=p(Z)$ can substantially speed up variational EM.

This repository is a personal copy of my original development repository that got transfered to the 'Gatsby-Sahani' github organization.

[1] Walker, William I., et al. "Unsupervised representation learning with recognition-parametrised probabilistic models." International Conference on Artificial Intelligence and Statistics. PMLR, 2023.
