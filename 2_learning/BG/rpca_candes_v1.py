from __future__ import division, print_function

import numpy as np

try:
    from pylab import plt
except ImportError:
    print('Unable to import pylab. R_pca.plot_fit() will not work.')

try:
    # Python 2: 'xrange' is the iterative version
    range = xrange
except NameError:
    # Python 3: 'range' is iterative - no need for 'xrange'
    pass


# D is in shape: [d,N]
class R_pca:

    def __init__(self, D, mu=None, lmbda=None):
        print('D: ', D.shape)
        self.D = D
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)

        if mu:
            self.mu = mu
        else:
            self.mu = np.prod(self.D.shape) / (4 * self.l1_norm(self.D))  # CHANGED FROM: self.mu = np.prod(self.D.shape) / (4 * self.frobenius_norm(self.D)) , 0.0694

        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))  #, 0.0069

        # # trying other parameters:
        # lmbda_ = 1 / np.sqrt(max(D.shape))
        # self.lmbda = lmbda_ / 3
        # self.mu = 10 * lmbda_ / 3

        # # DEBUG:
        self.mu = 1.0   #0.03849001794597506
        self.lmbda = 0.0009  #0.0038490017945975053

        self.mu_inv = 1 / self.mu

        print('mu: ', self.mu)
        print('lambda: ', self.lmbda)

    @staticmethod
    def frobenius_norm(M):
        return np.linalg.norm(M, ord='fro')

    @staticmethod
    def l1_norm(M):
        return np.linalg.norm(M, ord=1, axis=0).sum()

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        tmp = np.diag(self.shrink(S, tau)) @ V
        res = U @ tmp
        return res    # CHANGED FROM: np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def fit(self, tol=None, max_iter=1000, iter_print=100):
        iter = 0
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)
        _dlta = 1E-7

        if tol:
            _tol = tol
        else:
            ####### can change range here
            _tol = _dlta * self.frobenius_norm(self.D)
            _tol = 1E-4

        print('_tol: ', _tol)

        Dnorm = self.frobenius_norm(self.D)

        while iter < max_iter:
            Lk = self.svd_threshold(
                self.D - Sk + (self.mu_inv * Yk), self.mu_inv)
            Sk = self.shrink(
                self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)
            Yk = Yk + self.mu * (self.D - Lk - Sk)
            err = self.frobenius_norm(self.D - Lk - Sk) / Dnorm
            iter += 1
            if (iter % iter_print) == 0 or iter == 1 or iter > max_iter or err <= _tol:
                print('iteration: {0}, error: {1}'.format(iter, err))
                L_rank = np.linalg.matrix_rank(Lk)
                S_rank = np.linalg.matrix_rank(Sk)
                print('L_rank:', L_rank, ' S_rank:', S_rank)
                # print('L small values: ', (Lk < 1E-5).sum(), ' S small values: ', (Sk < 1E-5).sum())
            if err < _tol:
                break

        self.L = Lk
        self.S = Sk
        return Lk, Sk

    def plot_fit(self, size=None, tol=0.1, axis_on=True):

        n, d = self.D.shape

        if size:
            nrows, ncols = size
        else:
            sq = np.ceil(np.sqrt(n))
            nrows = int(sq)
            ncols = int(sq)

        ymin = np.nanmin(self.D)
        ymax = np.nanmax(self.D)
        print('ymin: {0}, ymax: {1}'.format(ymin, ymax))

        numplots = np.min([n, nrows * ncols])
        plt.figure()

        for n in range(numplots):
            plt.subplot(nrows, ncols, n + 1)
            plt.ylim((ymin - tol, ymax + tol))
            plt.plot(self.L[n, :] + self.S[n, :], 'r')
            plt.plot(self.L[n, :], 'b')
            if not axis_on:
                plt.axis('off')
