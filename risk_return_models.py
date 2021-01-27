import numpy as np
import pandas as pd
import strat_models
import networkx as nx
import cvxpy as cp
import torch

def huber_return_prox(Y, nu, theta, t, M):
    if Y is None:
        return nu

    ###torch
    if np.allclose(nu, 0):
        nu_tch = torch.from_numpy(np.random.randn(nu.shape[0])/1000)
    else:
        nu_tch = torch.from_numpy(nu)
    n,nk = Y[0].shape
    theta_tch = torch.from_numpy(theta).requires_grad_(True)
    loss = torch.nn.SmoothL1Loss(beta=M)
    optim = torch.optim.LBFGS([theta_tch], lr=1, max_iter=50)

    def closure():
        optim.zero_grad()
        l = torch.sum( (theta_tch - nu_tch)**2 )/(2*t)
        for y in Y[0].T:
            l += (1/1)*loss(theta_tch, torch.from_numpy(y))
        l.backward()
        return l
    
    optim.step(closure)
    return theta_tch.data.numpy()

class huber_return_loss(strat_models.Loss):
    """
    f(theta) = (1/2)*norm(u,2)**2       if norm(u,2) <= M
                M(norm(u,2) - M/2)      if norm(u,2) > M
    """
    def __init__(self, M=None):
        if M is None:
            raise ValueError("M must be a number.")
        super().__init__()
        self.isDistribution = True
        self.M = M

    def evaluate(self, theta, data):
        assert "Y" in data
        return None

    def setup(self, data, G):
        Y = data["Y"]
        Z = data["Z"]

        K = len(G.nodes())

        shape = (data["n"],)
        theta_shape = (K,) + shape

        #preprocess data
        for y, z in zip(Y, Z):
            vertex = G._node[z]
            if "Y" in vertex:
                vertex["Y"] += [y]
            else:
                vertex["Y"] = [y]

        Y_data = []
        for i, node in enumerate(G.nodes()):
            vertex = G._node[node]
            if 'Y' in vertex:
                Y = vertex['Y']
                Y_data += [Y]
                del vertex['Y']
            else:
                Y_data += [None]

        cache = {"Y": Y_data, "n":data["n"], "theta_shape":theta_shape, "shape":shape, "K":K}
        return cache

    def prox(self, t, nu, warm_start, pool, cache):
        """
        Proximal operator for joint covariance estimation
        """
        res = pool.starmap(huber_return_prox, zip(cache["Y"], nu, warm_start, t*np.ones(cache["K"]), self.M*np.ones(cache["K"])))
        return np.array(res)

    def logprob(self, data, G):
        
        logprobs = []
        
        for y,z in zip(data["Y"], data["Z"]):
            n, nk = y.shape
            y_bar = np.mean(y, axis=1).flatten()
            
            if (y_bar == np.zeros(n)).all():
                continue            
            
            mu = G._node[z]["theta"].copy().flatten()
            
            lp = 0
            for i in range(nk):
                lp += sum((mu - y[:,i])**2) / (2*nk)            
            logprobs += [-lp]

        return logprobs

    def sample(self, data, G):
        """
        Samples from ~N(mu_z, I)
        """
        Z = turn_into_iterable(data["Z"])
        mus = [G._node[z]["theta"] for z in Z]

        n = mus[0].shape[0]
        return [np.random.multivariate_normal(mu, np.eye(n)) for mu in mus]

def joint_cov_prox(Y, nu, theta, t):
    """
    Proximal operator for joint covariance estimation
    """
    if Y is None:
        s, Q = np.linalg.eigh(nu)
        s[s <= 0] = 1e-8
        return Q @ np.diag(s) @ Q.T
    
    n, nk = Y[0].shape
    Yemp = Y[0]@Y[0].T/nk
    
    s, Q = np.linalg.eigh(nu/(t*nk)-Yemp)
    w = ((t*nk)*s + np.sqrt(((t*nk)*s)**2 + 4*(t*nk)))/2
    return Q @ np.diag(w) @ Q.T

class covariance_max_likelihood_loss(strat_models.Loss):
    """
    f(theta) = Trace(theta @ Y) - logdet(theta)
    """
    def __init__(self):
        super().__init__()
        self.isDistribution = True

    def evaluate(self, theta, data):
        assert "Y" in data
        return np.trace(theta @ data["Y"]) - np.linalg.slogdet(theta)[1]

    def setup(self, data, G):
        Y = data["Y"]
        Z = data["Z"]

        K = len(G.nodes())

        shape = (data["n"], data["n"])
        theta_shape = (K,) + shape

        #preprocess data
        for y, z in zip(Y, Z):
            vertex = G._node[z]
            if "Y" in vertex:
                vertex["Y"] += [y]
            else:
                vertex["Y"] = [y]

        Y_data = []
        for i, node in enumerate(G.nodes()):
            vertex = G._node[node]
            if 'Y' in vertex:
                Y = vertex['Y']
                Y_data += [Y]
                del vertex['Y']
            else:
                Y_data += [None]

        cache = {"Y": Y_data, "n":data["n"], "theta_shape":theta_shape, "shape":shape, "K":K}
        return cache

    def prox(self, t, nu, warm_start, pool, cache):
        """
        Proximal operator for joint covariance estimation
        """
        res = pool.starmap(joint_cov_prox, zip(cache["Y"], nu, warm_start, t*np.ones(cache["K"])))
        return np.array(res)

    def logprob(self, data, G):
        
        logprobs = []
        
        for y,z in zip(data["Y"], data["Z"]):
            n, nk = y.shape
            Y = (y@y.T)/nk
            
            if (np.zeros((n,n)) == Y).all():
                continue            
            
            theta = G._node[z]["theta_tilde"]
            logprobs += [np.linalg.slogdet(theta)[1] - np.trace(Y@theta)]

        return logprobs

    def sample(self, data, G):
        Z = turn_into_iterable(data["Z"])
        sigmas = [np.linalg.inv(G._node[z]["theta"]) for z in Z]

        n = sigmas[0].shape[0]
        return [np.random.multivariate_normal(np.zeros(n), sigma) for sigma in sigmas]