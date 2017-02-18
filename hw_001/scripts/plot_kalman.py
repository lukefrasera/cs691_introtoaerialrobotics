#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def main():

  #  Input data
  data = np.genfromtxt('result.txt', delimiter=",")

  z = data[0]
  xhat = data[1]
  Pminus = data[2]
  x = -0.37727
  n_iter = 50

  plt.figure()
  plt.plot(z,'k+',label='noisy measurements')
  plt.plot(xhat,'b-',label='a posteri estimate')
  plt.axhline(x,color='g',label='truth value')
  plt.legend()
  plt.title('Estimate vs. iteration step', fontweight='bold')
  plt.xlabel('Iteration')
  plt.ylabel('Voltage')

  plt.figure()
  valid_iter = range(1,n_iter) # Pminus not valid at step 0
  plt.plot(valid_iter,Pminus[valid_iter],label='a priori error estimate')
  plt.title('Estimated $\it{\mathbf{a \ priori}}$ error vs. iteration step', fontweight='bold')
  plt.xlabel('Iteration')
  plt.ylabel('$(Voltage)^2$')
  plt.setp(plt.gca(),'ylim',[0,.01])
  plt.show()

if __name__ == "__main__":
  main()