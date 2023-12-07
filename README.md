# RPM implementation in pytorch

Development repository of Marcel Nonnenmacher on the exponential family recognition-parametrised model (RPM) [1]
written in 2023 while working with Maneesh Sahani at the Gatsby Computational Neuroscience Unit.

The main focus of this work was 1) to experiment with the RPM itself and understand its core functioning
                                   (which turns out to be discretization of observed space under cond. independence).
                                2) to implement various fitting methods that go beyond the basic saturated VI of the
                                   original AISTATS publication [1], such as minibatch SGD, amortized q(Z|X) etc. 
Also contains basic implementations of several RPM variants for testing purposes, some of which never made much sense. 

This repository is a copy of the development repository that got transfered to the 'Gatsby-Sahani' github organization.

- expFam RPM (rpm.py) : latent variable Z is continuous, i.e. p(Z), f(Z|xj) are some continuous exponential family.
                        Note this case generally has intractable ELBO, and so far only the inner variational bound
                        is implemented. f(Z|xj) can be conditional expFam or some general 
                        log f(Z|xj) = g(Z,xj) = gj(xj)'t(Z) + hj(xj) that doesn't have to be normalized. 
- temporal expFam RPM (rpm.py) : time-series version of the RPM. Relies strongly on marginalization properties of
                                 multivariate densities f(Z_{t=1,..,T}|xj) to have tractable f(Z_t|xj), so largely
                                 restricted to Gaussian f(Z|xj), p(Z).
- RPVAEs and tempRPVAEs (rpm.py, merged into RPM class resp tempRPM class): Variant of RPM with amortized q(Z|xj)
                        that assumes Fj(Z) = p(Z) only in q(Z|xj) (i.e. not in p(xj|Z)), which not only makes q(Z|xj)
                        tractable and computable given the current generative model, but also simplifies the ELBO.
                        ELBO still intractable for continous Z and needs e.g. the inner variational bound. 
                        Automatically used when providing a (temp-)RPM with argument q='use_theta'. 

- Discrete RPM (discreteRPM.py) : latent variable Z is discrete, i.e. p(Z), f(Z|xj) are categorical. Also allows
                                  f(Z|xj) = e^g(Z,xj) unnormalized, i.e. g(Z,xj) = g(xj)'t(Z) + h(xj) for some h.  
                                  Likelihood is tractable, so no tricks implemented. 
- implicitRPM (implicitRPM.py) : abandoned RPM version directly defining w(x) = int p(x) prod fj(Z|xj)/Fj(Z) dZ
                                 under an assumption that Fj(Z)=p(Z) which I hoped to make 'less false' by
                                 encouraing p_model(xj) = p_data(xj). Turns out matching xj-marginals will enforce
                                 Fj(Z)=p(Z) (and hence make the assumption true) only in the case dim(eta) <= dim(xj),
                                 so the intended use-case for implicit DDC won't work.
  
Expermiments: main experiments imlemented so far include 
- the bouncing balls examples from the RPM paper [1] that can be found in utils_data_external.py, 
- a two-dimensional Gaussian copula with 1D marginals (either Gaussian or exponentially distributed), see notebooks.
- several variants of the peer-supervision task on MNIST used for testing discrete RPMs, see notebooks. 

[1] Walker, William I., et al. "Unsupervised representation learning with recognition-parametrised probabilistic models." International Conference on Artificial Intelligence and Statistics. PMLR, 2023.
