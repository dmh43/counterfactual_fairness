import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cff.policies import NaivePolicy, UnawarePolicy, FairPolicy
from cff.sim import simulate_exogenous_vars, simulate_endogenous_vars
from cff.ds_helpers import build_plot

def main():
  # generate some data
  nb_obs = 100000
  nb_seats = 20000
  R, S, A = simulate_exogenous_vars(nb_obs, R_pct=0.75, S_pct=0.6) # simulate exogeous variables
  G, L, F = simulate_endogenous_vars(A, R, S) # simulate endogenous variables

  # set up naive policy
  naivePolicy = NaivePolicy()

  # set up and train unaware policy
  unawarePolicy = UnawarePolicy()
  unawarePolicy.train(G, L, F)

  # set up and train fair policy
  fairPolicy = FairPolicy()
  fairPolicy.train(R, S, G, L)

  # form policy dictionary
  P = {'naive': naivePolicy.evaluate(G, L, nb_seats),
       'unaware': unawarePolicy.evaluate(G, L, nb_seats),
       'fair': fairPolicy.evaluate(R, S, G, L, nb_seats)}

  build_plot(P['naive'], A, R, S, G, L, F)
  build_plot(P['unaware'], A, R, S, G, L, F)
  build_plot(P['fair'], A, R, S, G, L, F)

if __name__ == "__main__": main()
