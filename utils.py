import numpy as np
import cvxpy as cp
import pandas as pd

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
                for z4 in range(num_quantiles):
                    c4 = df_Z[cols[3]]==z4
                    if not df_Z[c1&c2&c3&c4].empty:
                        Zidx += [df_Z[c1&c2&c3&c4].index.tolist()]
                        Zs += [(z1, z2, z3, z4)]  
                        
    Ys = []
    for idx in Zidx:
        Ys += [ df_Y.loc[idx].values.T ]
        
    return dict(Y=Ys, Z=Zs, n=num_assets)

def CORR_SM(preds, df):
    return np.corrcoef(preds.flatten(), df.values.flatten())[0,1]

def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

def backtest(returns, Z_returns, benchmark, means, covs,
            lev_lim, bottom_sec_limit, upper_sec_limit, shorting_cost, tcost, 
            MAXRISK, tau=None, kappa=None):
    
    _, num_assets = returns.shape
    T, K = Z_returns.shape
    Zs_time = np.zeros((T-1, K))

    value_strat, value_benchmark = 1, 1
    vals_strat, vals_benchmark = [value_strat], [value_benchmark]
    
    benchmark_returns = benchmark.loc[Z_returns.index].copy().values.flatten()

    """
    On the fly computing for stratified model policy    
    """
    w_prev = cp.Parameter(num_assets+1)
    w = cp.Variable(num_assets+1)

    SIGMA = cp.Parameter((num_assets+1, num_assets+1), PSD=True)
    MU = cp.Parameter(num_assets+1)

    obj = - w@(MU + 1) + shorting_cost*(kappa@cp.neg(w)) + tcost*(cp.abs(w-w_prev)@tau)
    cons = [
        cp.quad_form(w, SIGMA) <= MAXRISK*100*100,
        sum(w) == 1,
        cp.norm1(w) <= lev_lim,
        bottom_sec_limit*np.ones(num_assets+1) <= w,
        w <= upper_sec_limit*np.ones(num_assets+1),
    ]
    prob_sm = cp.Problem(cp.Minimize(obj), cons)
    
    w_prev.value = np.zeros(num_assets+1)
    w_prev.value[-1] = 1
    
    W = [w_prev.value.reshape(-1,1)]

    for date in range(1,T):
        node = tuple(Z_returns.iloc[date])
        Zs_time[date-1, :] = [*node]

        #adding cash asset into returns and covariances
        cov_placeholder = np.vstack([covs[node], np.zeros(num_assets)])
        SIGMA.value = np.hstack([cov_placeholder, np.zeros(num_assets+1).reshape(-1,1)])
        MU.value = np.concatenate([means[node], np.zeros(1)])

        prob_sm.solve(verbose=False)    
        
        returns_date = (1+np.concatenate([returns[date, :], np.zeros(1)]))     
        value_strat *= returns_date@w.value - (kappa@cp.neg(w)).value - (cp.abs(w-w_prev)@tau).value
        vals_strat += [value_strat]
        W += [w.value.reshape(-1,1)]

        value_benchmark *= 1+benchmark_returns[date]
        vals_benchmark += [value_benchmark]

        w_prev.value = w.value.copy() * returns_date
        
    vals = pd.DataFrame(data=np.vstack([vals_strat, vals_benchmark]).T,
                        columns=["Stratified model policy", "benchmark"],
                        index=Z_returns.index.rename("Date"))
    
    Zs_time = pd.DataFrame(data=Zs_time,
                           index=Z_returns.index[1:],
                           columns=Z_returns.columns)
    
    #calculate sharpe
    rr = vals.pct_change()
    sharpes = np.sqrt(250)*rr.mean()/rr.std()
    returns = 250*rr.mean()

    return vals, Zs_time, W, sharpes, returns

def backtest_common(returns, Z_returns, benchmark, mean, cov,
                    lev_lim, bottom_sec_limit, 
                    upper_sec_limit, shorting_cost, tcost, 
                    MAXRISK, tau=None, kappa=None):


    """
    GENERATE COMMON MARKOWITZ MV PORTFOLIO
    COMMON MEAN, COMMON COVARIANCE
    """
    _, num_assets = returns.shape
    T, K = Z_returns.shape

    value_common, value_benchmark = 1, 1
    vals_common, vals_benchmark = [value_common], [value_benchmark]
    
    benchmark_returns = benchmark.loc[Z_returns.index].copy().values.flatten()

    Wcommon = []
    mu_common = np.concatenate([mean, np.zeros(1)])
    Sigma_common = np.vstack([cov, np.zeros(num_assets)])
    Sigma_common = np.hstack([Sigma_common, np.zeros(num_assets+1).reshape(-1,1)])

    w_common = cp.Variable(num_assets+1)
    w_common_prev = cp.Parameter(num_assets+1)
    w_common_prev.value = np.zeros(num_assets+1)
    w_common_prev.value[-1] = 1
    Wcommon += [w_common_prev.value.reshape(-1,1)]
    
    obj_common = - w_common@(mu_common+1) + shorting_cost*(kappa@cp.neg(w_common)) + tcost*(cp.abs(w_common-w_common_prev)@tau)

    cons_common = [
        cp.quad_form(w_common, Sigma_common) <= MAXRISK*100*100,
            sum(w_common) == 1,
            cp.norm1(w_common) <= lev_lim,
            bottom_sec_limit*np.ones(num_assets+1) <= w_common,
            w_common <= upper_sec_limit*np.ones(num_assets+1)
        ]

    prob_common = cp.Problem(cp.Minimize(obj_common), cons_common)

    for date in range(1,T):        
        prob_common.solve(verbose=False)
        returns_date = (1+np.concatenate([returns[date, :], np.zeros(1)]))

        value_common *= returns_date@w_common.value - (kappa@cp.neg(w_common)).value - (cp.abs(w_common-w_common_prev)@tau).value
        vals_common += [value_common]
        Wcommon += [w_common.value.reshape(-1,1)]

        value_benchmark *= 1+benchmark_returns[date]
        vals_benchmark += [value_benchmark]

        w_common_prev.value = w_common.value.copy() * returns_date
        
    vals = pd.DataFrame(data=np.vstack([vals_common, vals_benchmark]).T,
                        columns=["Common model policy", "benchmark"],
                        index=Z_returns.index.rename("Date"))
    
    #calculate sharpe
    rr = vals.pct_change()
    sharpes = np.sqrt(250)*rr.mean()/rr.std()
    return vals, Wcommon, sharpes