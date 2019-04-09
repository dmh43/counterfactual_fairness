import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from .ds_helpers import normalize

class NaivePolicy:
  def __init__(self):
    pass

  def evaluate(self, G, L, nb_seats=None):
    assert G.shape == L.shape
    nb_obs = G.shape[0]
    if nb_seats is None:
      nb_seats = nb_obs
    else:
      assert isinstance(nb_seats, int) and (nb_seats > 0)
      nb_seats = min(nb_obs, nb_seats)
    ind = (normalize(G) + normalize(L)).argsort(axis=0)[-nb_seats:][::-1]
    P = np.zeros([nb_obs, 1]).astype(bool)
    P[ind] = True
    return P

class UnawarePolicy:
  def __init__(self):
    self.F_reg = None

  def train(self, G, L, F):
    self.F_reg = LinearRegression().fit(np.hstack([G, L]), F)

  def evaluate(self, G, L, nb_seats=None):
    assert G.shape == L.shape
    nb_obs = G.shape[0]
    if nb_seats is None:
      nb_seats = nb_obs
    else:
      assert isinstance(nb_seats, int) and (nb_seats > 0)
      nb_seats = min(nb_obs, nb_seats)
    F_hat = self.F_reg.predict(np.hstack([G, L]))
    ind = F_hat.argsort(axis=0)[-nb_seats:][::-1]
    P = np.zeros([nb_obs, 1]).astype(bool)
    P[ind] = True
    return P

class FairPolicy:
  def __init__(self):
    self.G_reg, self.L_reg, self.A_reg, self.sgn = None, None, None, None

  def train(self, R, S, G, L):
    self.G_reg = LinearRegression().fit(np.hstack([R, S]), G)
    self.L_reg = LinearRegression().fit(np.hstack([R, S]), L)
    G_err = G - self.G_reg.predict(np.hstack([R, S]))
    L_err = L - self.L_reg.predict(np.hstack([R, S]))
    self.A_reg = PCA(whiten=True, n_components=1).fit(np.hstack([G_err, L_err]))
    self.sgn = np.sign(np.corrcoef(self.A_reg.transform(np.hstack([G_err, L_err])).T, G.T)[0, 1])

  def evaluate(self, R, S, G, L, nb_seats=None):
    assert R.shape == S.shape == G.shape == L.shape
    nb_obs = R.shape[0]
    if nb_seats is None:
      nb_seats = nb_obs
    else:
      assert isinstance(nb_seats, int) and (nb_seats > 0)
      nb_seats = min(nb_obs, nb_seats)
    G_err = G - self.G_reg.predict(np.hstack([R, S]))
    L_err = L - self.L_reg.predict(np.hstack([R, S]))
    A_hat = self.sgn * self.A_reg.transform(np.hstack([G_err, L_err]))
    ind = A_hat.argsort(axis=0)[-nb_seats:][::-1]
    P = np.zeros([nb_obs, 1]).astype(bool)
    P[ind] = True
    return P
