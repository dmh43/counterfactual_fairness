import numpy as np

def simulate_exogenous_vars(nb_obs, R_pct=0.8, S_pct=0.5):
  assert isinstance(R_pct, float) and (0 <= R_pct <= 1)
  assert isinstance(S_pct, float) and (0 <= S_pct <= 1)
  R = 1.0 * (np.random.uniform(low=0, high=1, size=[nb_obs, 1]) < R_pct)
  S = 1.0 * (np.random.uniform(low=0, high=1, size=[nb_obs, 1]) < S_pct)
  A = np.random.randn(nb_obs, 1)
  return R, S, A

def simulate_endogenous_vars(A, R, S):
  assert A.shape == R.shape == S.shape
  nb_obs = A.shape[0]
  G = np.floor(A + 5.0 * R + 1.0 * S + 0.5 * np.random.randn(nb_obs, 1))
  L = 0.3 * A + 1.0 * R + 0.1 * S + 1.0 * np.random.randn(nb_obs, 1)
  F = A + 0.1 * R + 1.0 * S + 0.3 * np.random.randn(nb_obs, 1)
  return G, L, F
