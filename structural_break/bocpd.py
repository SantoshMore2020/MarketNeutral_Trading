import math
import numpy as np
import random
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import pandas as pd

# -------------------------
# BOCPD implementation
# (Gaussian observation with Normal-Gamma conjugate prior)
# -------------------------
class BOCPD:
    def __init__(self, hazard_lambda=200.0):
        # hazard_lambda is used to construct constant hazard: H = 1 / hazard_lambda
        self.hazard_lambda = hazard_lambda
        self.H = 1.0 / hazard_lambda
        self.max_run_length = 1000  # cap for numerical tractability
        # run-length posterior (vector) initialized with r=0 prob=1
        self.log_r = np.array([0.0])  # in log domain to avoid underflow

        # parameters for Normal-Gamma prior per run length component:
        # We will maintain arrays for conjugate posterior parameters per run length
        # Normal-Gamma prior: mu ~ Normal(mu0, kappa0^-1), tau (precision) ~ Gamma(alpha0, beta0)
        self.mu0 = 0.0
        self.kappa0 = 1.0
        self.alpha0 = 1.0
        self.beta0 = 1.0
        # store arrays for kappa, mu, alpha, beta for each possible run-length
        self.kappa = np.array([self.kappa0])
        self.mu = np.array([self.mu0])
        self.alpha = np.array([self.alpha0])
        self.beta = np.array([self.beta0])

    def reset(self):
        self.__init__(self.hazard_lambda)

    def _predictive_logpdf(self, x):
        # Student-t predictive log density from Normal-Gamma posterior (vectorized)
        # See Adams & MacKay; mixture of t-distributions
        # For each run length component i, compute predictive log pdf of x
        nu = 2.0 * self.alpha
        scale_sq = (self.beta * (self.kappa + 1.0)) / (self.alpha * self.kappa)
        # Student-t with nu degrees, loc=mu, scale = sqrt(scale_sq)
        # log pdf:
        # logpdf = lgamma((nu+1)/2) - lgamma(nu/2) - 0.5*log(nu*pi*scale_sq) - ((nu+1)/2)*log(1 + (x-mu)^2/(nu*scale_sq))
        from math import lgamma, log, pi
        xm = x - self.mu
        logpdfs = []
        for i in range(len(nu)):
            nu_i = nu[i]
            scale_sq_i = scale_sq[i]
            term1 = lgamma((nu_i + 1.0) / 2.0) - lgamma(nu_i / 2.0)
            term2 = -0.5 * math.log(nu_i * math.pi * scale_sq_i)
            term3 = - (nu_i + 1.0) / 2.0 * math.log(1.0 + (xm[i]**2) / (nu_i * scale_sq_i) + 1e-12)
            logpdfs.append(term1 + term2 + term3)
        return np.array(logpdfs)

    def update(self, x):
        """
        Online update with new scalar observation x.
        Returns: change_prob (P(r_t = 0))
        """
        # Maintain log domain run-length posterior for stability
        # Step 1: compute predictive probabilities for each run-length hypothesis
        # For simplicity, vectorize predictive using Student-t approx: but the implementation above loops.
        # Expand arrays if needed
        R = len(self.log_r)
        # compute predictive logpdf for each component. Need mu, kappa, alpha, beta arrays.
        # For numerical stability and speed, we will compute predictive using an approximate Gaussian for now:
        # predictive mean = mu_i, predictive var = beta_i / (alpha_i * kappa_i) * (kappa_i + 1)
        pred_means = self.mu
        pred_var = (self.beta * (self.kappa + 1.0)) / (self.alpha * self.kappa)
        # log predictive Gaussian:
        log_pred = -0.5 * np.log(2 * np.pi * pred_var + 1e-12) - 0.5 * ((x - pred_means) ** 2) / (pred_var + 1e-12)
        # Step 2: growth probabilities (extend each run)
        log_growth = self.log_r + log_pred + np.log(1.0 - self.H + 1e-12)
        # Step 3: change probabilities (new run starting at 0) coming from any previous r
        # compute evidence for new run: previous run-len probs * predictive * hazard
        log_cp_terms = self.log_r + log_pred + np.log(self.H + 1e-12)
        # logsumexp for cp
        max_log = np.max(log_cp_terms)
        log_cp = max_log + np.log(np.sum(np.exp(log_cp_terms - max_log)))
        # new run-length posterior: r=0 has log_cp; r>0 have log_growth shifted (r+1 -> r)
        new_log_r = np.concatenate(([log_cp], log_growth))
        # normalize
        logZ = np.log(np.sum(np.exp(new_log_r - new_log_r.max()))) + new_log_r.max()
        new_log_r = new_log_r - logZ
        # cap length
        if len(new_log_r) > self.max_run_length:
            new_log_r = new_log_r[:self.max_run_length]
        self.log_r = new_log_r

        # Update conjugate parameters for each run-length component.
        # Create new arrays: the run length increments: for r=0 set to prior; for r>0 update from previous component + x
        Rnew = len(self.log_r)
        new_kappa = np.zeros(Rnew)
        new_mu = np.zeros(Rnew)
        new_alpha = np.zeros(Rnew)
        new_beta = np.zeros(Rnew)
        # r=0: reset to prior
        new_kappa[0] = self.kappa0
        new_mu[0] = self.mu0
        new_alpha[0] = self.alpha0
        new_beta[0] = self.beta0
        # r>0: update from previous (r-1) component
        for r in range(1, Rnew):
            # previous index r-1 must exist in old arrays (if old shorter, use prior)
            if r-1 < len(self.kappa):
                kp = self.kappa[r-1]
                mu_p = self.mu[r-1]
                alpha_p = self.alpha[r-1]
                beta_p = self.beta[r-1]
            else:
                kp = self.kappa0; mu_p = self.mu0; alpha_p = self.alpha0; beta_p = self.beta0
            # update conjugate Natural update for single observation x:
            new_kappa[r] = kp + 1.0
            new_mu[r] = (kp * mu_p + x) / new_kappa[r]
            new_alpha[r] = alpha_p + 0.5
            # update beta: see Bayesian normal-gamma update
            delta = x - mu_p
            new_beta[r] = beta_p + 0.5 * (kp * (delta**2)) / (new_kappa[r])
        # assign back
        self.kappa = new_kappa
        self.mu = new_mu
        self.alpha = new_alpha
        self.beta = new_beta

        # change probability is posterior mass at r=0 (exp(log_r[0]))
        change_prob = float(np.exp(self.log_r[0]))
        return change_prob

