
# coding: utf-8

# In[1]:

import pandas as pd
pd.set_eng_float_format(accuracy=3, use_eng_prefix=False)
import numpy as np


# In[2]:

data = pd.read_csv('raw_data.csv', index_col=0, parse_dates=[0])


# In[3]:

class covariance(object):
    
    def __init__(self, returns):
        self.returns = returns
    
    @property
    def names(self):
        return list(self.returns.columns.values)
    
    @property
    def num_obs(self):
        return len(self.returns)
    
    @property
    def index(self):
        return self.returns.index
    
    def _covariance(self, returns, corr_decay=0, vol_decay=0, horizon=1):
        X = np.column_stack([returns])
        X -= X.mean(axis=0)
        
        time = np.arange(len(returns))[::-1]
        decay_vol = np.power((1 - vol_decay), time)
        decay_corr = np.power((1 - corr_decay), time)
        
        cov = np.dot((X.T * decay_corr), X.conj()) / decay_corr.sum()
        cov_vol = np.dot((X.T * decay_vol), X.conj()) / decay_vol.sum()
        np.fill_diagonal(cov, np.diagonal(cov_vol))
        
        if horizon == 260:
            cov = cov * 260
            
        names = list(returns.columns.values)
        cov = pd.DataFrame(cov, index=names, columns=names)
        return cov
    
    def covariance(self, corr_decay=0, vol_decay=0, horizon=1):
        return self._covariance(self.returns, corr_decay, vol_decay, horizon)

    def rolling_covariance(self, corr_decay=0, vol_decay=0, min_day=260, max_day=2600, horizon=1):
        if self.num_obs < min_day:
            raise ValueError('Not enough data')
        else:
            cov = []
            for i in range(self.num_obs):
                if i < (max_day - min_day):
                    new = self._covariance(self.returns.ix[:i+min_day], corr_decay, vol_decay, horizon)
                else:
                    new = self._covariance(self.returns.ix[(i+min_day-max_day):i+min_day], corr_decay, vol_decay, horizon)
                cov.append(new)
            cov = pd.Series(cov, index = self.index)
        return cov


# In[4]:

class variance_analysis(object):
    def __init__(self, N_cov_mat_copy, portfolio_weight):
        self.N_cov_mat = N_cov_mat_copy
        self.portfolio_weight = portfolio_weight
        self.names = list(portfolio_weight.index)
    
    def cov_to_corr(self, cov_mat):
        D = np.sqrt(np.diag(cov_mat))
        corr = cov_mat/(np.asmatrix(D).T * np.asmatrix(D))
        return corr
    
    def decomp(self):
        corr = []
        vol = []
        for cov_mat in self.N_cov_mat:
            i_vol = np.diag(cov_mat)
            i_vol = pd.Series(i_vol, index = self.names)
            vol.append(i_vol)
            corr.append(self.cov_to_corr(cov_mat))
        corr = pd.Series(corr, index = self.N_cov_mat.index)
        vol = pd.Series(vol, index = self.N_cov_mat.index)
        return corr, vol
    
    def vol_floor(self, vol_floor = 0.00003):
        N_corr, N_vol = self.decomp()
        N_vol_copy = N_vol.copy()
        New_N_cov = []
        for i in N_vol_copy.index:
            vol = N_vol_copy.ix[i].copy()
            vol.flags.writeable = True
            vol[vol<vol_floor] = vol_floor
            
            cov = np.dot(np.dot(np.diag(np.sqrt(vol)), N_corr.ix[i]), np.diag(np.sqrt(vol)))
            cov = pd.DataFrame(cov, index=self.names, columns = self.names)
            New_N_cov.append(cov)
        New_N_cov = pd.Series(New_N_cov, index = self.N_cov_mat.index)
        self.N_cov_mat = New_N_cov.copy() #change the global N_cov_mat if using floor
        return New_N_cov
    
    def ex_ante_vol(self):
        '''
        Input parameters: covariance matrix (NxNxT), portfolio (1xN or TxN)
        Output: Tx1 vector of ex-ante estimated volatility
        '''
        weight = np.asmatrix(self.portfolio_weight)
        N_ex_ante_ext_vol = []
        for cov_mat in self.N_cov_mat:
            ex_ante_ext_vol = weight * np.asmatrix(cov_mat) * weight.T
            N_ex_ante_ext_vol.append(ex_ante_ext_vol.item())
        N_ex_ante_ext_vol = pd.Series(N_ex_ante_ext_vol,index = self.N_cov_mat.index)
        return N_ex_ante_ext_vol
    
    def asset_betas(self):
        corr, vol = self.decomp()
        N_ex_ante_ext_vol = self.ex_ante_vol()
        beta = []
        for i in self.N_cov_mat.index:
            beta_ = np.asmatrix(self.portfolio_weight) * np.asmatrix(self.N_cov_mat.ix[i]) / N_ex_ante_ext_vol.ix[i]
            tmp_beta = pd.Series(beta_.tolist()[0], index = self.names)
            beta.append(tmp_beta)
        beta = pd.Series(beta, index=self.N_cov_mat.index)
        return beta
    
    def mcr(self):
        '''
        Input parameters: covariance matrix (NxNxT), portfolio (1xN or TxN)
        Output: TxN matrix of ex-ante estimated marginal contribution to risk of each asset in the portfolio
        '''
        MCR = []
        for i in self.N_cov_mat.index:
            numerator = np.asmatrix(self.N_cov_mat.ix[i]) * np.asmatrix(self.portfolio_weight).T
            denominator = np.asmatrix(self.portfolio_weight) * numerator 
            MCR_ = np.array(self.portfolio_weight) * np.array(numerator.T) * denominator.item()
            tem_MCR = pd.Series(MCR_.tolist()[0], index= self.names)
            MCR.append(tem_MCR)
        MCR = pd.Series(MCR, index=self.N_cov_mat.index)
        return MCR


# In[5]:

if __name__ == '__main__':
    test = covariance(data.dropna())
    N_cov_mat = test.rolling_covariance()
    N_cov_mat_copy = N_cov_mat.copy()
    portfolio_weight = pd.Series([0.3, 0.2, 0.1, 0.1, -0.2, -0.3], index = N_cov_mat.ix[0].columns)

    variance_analysis_test = variance_analysis(N_cov_mat, portfolio_weight)

