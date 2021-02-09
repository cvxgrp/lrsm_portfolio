import numpy as np
import cvxpy as cp
import pandas as pd
import networkx as nx

import os
import pickle
import strat_models

def get_data_dict(df_Y, df_Z, num_assets, num_quantiles=10):
    #This gets the indices of each z
    Zidx = []
    Zs = []
    cols = df_Z.columns
    for z1 in range(num_quantiles):
        c1 = df_Z[cols[0]]==z1
        for z2 in range(num_quantiles):
            c2 = df_Z[cols[1]]==z2
            for z3 in range(num_quantiles):
                c3 = df_Z[cols[2]]==z3        
                if not df_Z[c1&c2&c3].empty:
                    Zidx += [df_Z[c1&c2&c3].index.tolist()]
                    Zs += [(z1, z2, z3)]  
                        
    Ys = []
    for idx in Zidx:
        Ys += [ df_Y.loc[idx].values.T ]
        
    return dict(Y=Ys, Z=Zs, n=num_assets)

def make_G(w1, w2, w3):
    G_vix = nx.path_graph(10)
    G_inflation = nx.path_graph(10)
    G_mort = nx.path_graph(10)

    strat_models.set_edge_weight(G_vix, w1)
    strat_models.set_edge_weight(G_inflation, w2)
    strat_models.set_edge_weight(G_mort, w3)

    G = strat_models.cartesian_product([G_vix, G_inflation, G_mort])
    
    return G.copy()

def corr(preds, df):
    return np.corrcoef(preds.flatten(), df.values.flatten())[0,1]

def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

def backtest(returns, Z_returns, benchmark, means, covs,
            lev_lim, bottom_sec_limit, upper_sec_limit, shorting_cost, tcost, 
            MAXRISK, kappa=None, bid_ask=None):
    
    _, num_assets = returns.shape
    T, K = Z_returns.shape
    Zs_time = np.zeros((T-1, K))

    value_strat, value_benchmark = 1, 1
    vals_strat, vals_benchmark = [value_strat], [value_benchmark]
    
    benchmark_returns = benchmark.loc[Z_returns.index].copy().values.flatten()

    W = [np.zeros(18)]
    W[0][8] = 1 #vti
    for date in range(1,T):

        dt = Z_returns.iloc[date].name.strftime("%Y-%m-%d")

        node = tuple(Z_returns.iloc[date])

        Zs_time[date-1, :] = [*node]

        if date == 1:
            w_prev = W[0].copy()
        else:
            w_prev = W[-1].flatten().copy()

        w = cp.Variable(num_assets)

        #adding returns and covariances for the day
        SIGMA = covs[node]
        MU = means[node]

        roll = 15

        #get last 15 days tcs, lagged by one! This doesnt include today's date
        tau = np.maximum( bid_ask.loc[:dt].iloc[-(roll+1):-1].mean().values , 0)/2

        obj = - w@(MU + 1) + shorting_cost*(kappa@cp.neg(w)) + tcost*(cp.abs(w-w_prev))@tau
        cons = [
            cp.quad_form(w, SIGMA) <= MAXRISK*100*100,
            sum(w) == 1,
            cp.norm1(w) <= lev_lim,
            bottom_sec_limit*np.ones(num_assets) <= w,
            w <= upper_sec_limit*np.ones(num_assets),
        ]
        prob_sm = cp.Problem(cp.Minimize(obj), cons)


        prob_sm.solve(verbose=False)    
        
        returns_date = 1+returns[date, :]     

        #get TODAY's bid ask spread. We use this in computing the transaction costs.
        tau_sim = bid_ask.loc[dt].values.flatten()/2

        value_strat *= returns_date@w.value - (kappa@cp.neg(w)).value - (cp.abs(w-w_prev)@tau_sim).value
        vals_strat += [value_strat]

        value_benchmark *= 1+benchmark_returns[date]
        vals_benchmark += [value_benchmark]

        w_prev = w.value.copy() * returns_date
        W += [w_prev.reshape(-1,1)]
        
    vals = pd.DataFrame(data=np.vstack([vals_strat, vals_benchmark]).T,
                        columns=["policy", "benchmark"],
                        index=Z_returns.index.rename("Date"))
    
    Zs_time = pd.DataFrame(data=Zs_time,
                           index=Z_returns.index[1:],
                           columns=Z_returns.columns)
    
    #calculate sharpe
    rr = vals.pct_change()
    sharpes = np.sqrt(250)*rr.mean()/rr.std()
    returns = 250*rr.mean()

    return vals, Zs_time, W, sharpes, returns