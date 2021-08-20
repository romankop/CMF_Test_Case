from filterpy.kalman import KalmanFilter
import numpy as np
import pandas as pd
import statsmodels.api as sm
import os


class KalmanFilterStrategyETF(KalmanFilter):

    def __init__(self, dim_x, dim_z, dim_u=0,
                 x=None, P=None, R=None, Q=None, F=None, H=None):
        super().__init__(dim_x, dim_z, dim_u=dim_u)

        if x is not None:
            self.x = x
        if P is not None:
            self.P = P
        if R is not None:
            self.R = R
        if x is not None:
            self.Q = Q
        if x is not None:
            self.F = F
        if x is not None:
            self.H = H

    def update(self, z, R=None, H=None):

        super().update(z, R=R, H=H)
        self.H = H
        return self.y[0, 0], np.sqrt(self.S[0, 0])

    def fit_filter(self, path, tickers, n,
                   start_date, use_ols, returns):

        H_s = pd.read_csv(os.path.join(path, "fit_%s" % n +
                                       "_%s" % start_date +
                                       "_%s.csv" % tickers[0]))['Adj Close']

        z_s = pd.read_csv(os.path.join(path, "fit_%s" % n +
                                       "_%s" % start_date +
                                       "_%s.csv" % tickers[1]))['Adj Close']

        if returns:
            H_s = H_s.pct_change().dropna()
            z_s = z_s.pct_change().dropna()

        H_s = np.array([[value, 1.0] for value in H_s.tolist()])
        z_s = z_s.values

        z_reg = sm.OLS(z_s, H_s).fit()

        self.R *= z_reg.scale

        if use_ols:
            self.x = z_reg.params.reshape(-1, 1)
            self.P = z_reg.cov_params()

        x_s = []

        for H, z in zip(H_s, z_s):
            self.update(z, H=np.array([H]))
            x_s.append([self.x[0, 0], 1])

        x_s = np.array(x_s)

        x_autoreg = sm.OLS(x_s[1:, 0], x_s[:-1]).fit()

        self.F[0] = x_autoreg.params
        self.Q *= x_autoreg.scale







