import pandas as pd
import numpy as np
import math
import functools
import time
pd.set_eng_float_format(accuracy=3, use_eng_prefix=False)
import multiprocessing as mp


class Covariance(object):
    '''
    Class of covariance calculation
    Input return data, a pandas DataFrame indexed by data and assets
    '''

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
        ''' Input correlation decay, volatility decay, return covariance of input data.
        '''
        return self._covariance(self.returns, corr_decay, vol_decay, horizon)

    def generator_matrix(self, max_day, min_day):
        for i in range(len(self.returns)):
            if i < (max_day - min_day):
                yield self.returns.ix[:i + min_day]
            else:
                yield self.returns.ix[(i + min_day - max_day):i + min_day]

    def rolling_covariance(self, corr_decay=0, vol_decay=0, min_day=260, max_day=2600, horizon=1):
        '''Input correlation decay, volatility decay, return a N*N*T covariance matrix. Indexed by data and assets.
        '''
        if self.num_obs < min_day:
            raise ValueError('Not enough data')
        else:
            p_covariance = functools.partial(self._covariance, corr_decay=corr_decay ,vol_decay=vol_decay ,horizon = horizon)
            p = mp.Pool(31)
            c_g_matrix = self.generator_matrix(max_day, min_day)
            cov = p.map(p_covariance, c_g_matrix)
            p.close()
            p.join()


            # cov = []
            # for i in range(self.num_obs):
            #     if i < (max_day - min_day):
            #         new = p_covariance(self.returns.ix[:i + min_day])
            #     else:
            #         new = p_covariance(self.returns.ix[(i + min_day - max_day):i + min_day])
            #     cov.append(new)
            cov = pd.Series(cov, index=self.index)
        return cov

    def log_likehood(self, corr_decay=0, vol_decay=0, min_day=260, max_day=2600, horizon=1, lag=0):
        cul_val = 0
        data = self.returns
        N_cov_mat = self.rolling_covariance(corr_decay, vol_decay, min_day, max_day, horizon)
        u = data.mean(axis=0)
        for x in range(lag, len(data)):
            pa = np.log(np.linalg.det(N_cov_mat.ix[x - lag]))
            pb = (data.ix[x] - u)
            pc = np.mat(N_cov_mat.ix[x - lag])
            temp = -0.5 * (pa + np.dot(np.dot(pb.T, pc.I), pb) + 6 * np.log(2 * np.pi))
            if not math.isnan(temp):
                cul_val += temp.item()
        return cul_val / (len(data) + lag)


if __name__ == '__main__':
    data = pd.read_csv('C:\\Users\\ll299\\PycharmProjects\\covariance_estimate\\data\\raw_data.csv', index_col=0)
    start = time.time()

    test = Covariance(data.dropna())

    # print(test.covariance())
    # N_cov_mat = test.rolling_covariance(corr_decay=0.1, vol_decay=0.1)
    # print(N_cov_mat)



    decay = list(np.arange(0,0.1,0.01))
    lag = list(np.arange(0, 3))
    value_likehood = []
    for i in decay:
        temp = []
        for j in lag:
            likehood = test.log_likehood(corr_decay=i, vol_decay=i, lag=j)
            temp.append(likehood)
        value_likehood.append(temp)
    print(value_likehood)

    end = time.time()
    print('finished, time = ', end - start)
