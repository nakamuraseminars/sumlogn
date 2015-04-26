from numpy import log, exp, newaxis, sum, sqrt, matrix
from numpy import linspace, sinh, cosh, hstack, vstack, trapz, pi
from scipy.special import erfc

def mmFentonWilkinson(m, s):
  """
  Implements Fenton-Wilkinson approach for sum of lognormals. 
  Assume L = sum L_i, with L_i = exp(Y_i).
  Inputs are m and s, which are the vector of means and
  the covariance matrix respectively, such that Y ~ N(m, s).
  Outputs mean and stdev for the single lognormal that approximates L,
  that is exp(Z) with Z ~ N(mz, sz).
  """
  # Based on Fenton-Wilkinson Approximation
  # Fenton, L.F. (1960). The sum of log-normal probability distibutions in 
  # scattered transmission systems. IEEE Trans. Commun. Systems 8: 57-67.
  # Correlated version from Pekka Pirinen (2003). Statistical power sum analysis 
  # for nonidentically distributed correlated lognormal signals
  #
  # The Python code is based on original MatLab code by Takaki Makino (2011)
  #
  # Copyright (c) 2011 Takaki Makino. [http://www.snowelm.com/~t/]
  # Copyright (c) 2015 Nakamura Seminars. [http://nakamuraseminars.org/]
  #
  # This program is free software; you can redistribute it and/or modify
  # it under the terms of the GNU General Public License as published by
  # the Free Software Foundation; either version 3 of the License, or
  # (at your option) any later version.
  # This program is distributed in the hope that it will be useful,
  # but WITHOUT ANY WARRANTY; without even the implied warranty of
  # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  # GNU General Public License http://www.gnu.org/licenses/gpl.html 
  # for more details.  
  
  ds = s.diagonal()
  u1 = sum(exp(m + 0.5*ds))
  # u1 = sum([m[i]+0.5*s[i,i] for i in range(m.size)])
  u2 = sum(exp(m + m[:,newaxis] + 0.5*(ds + ds[:,newaxis])+s))
  # u2 = 2*sum([sum([exp(m[i]+m[j]+0.5*(s[i,i]+s[j,j])+s[i,j]) 
  #      for j in range(i+1,m.size)]) for i in range(m.size-1)])   
  # u2 = u2 + sum([exp(2*m[i]+2*s[i,i]) for i in range(m.size)])
  mz = 2*log(u1)-0.5*log(u2)
  sz = log(u2)-2*log(u1)
  return mz, sz

def mmSchwartzYehHo(m, s):
  """
  Implements Schwartz-Yeh-Ho approach for sum of lognormals. 
  Assume L = sum L_i, with L_i = exp(Y_i).
  Inputs are m and s, which are the vector of means and
  the covariance matrix respectively, such that Y ~ N(m, s).
  Outputs mean and stdev for the single lognormal that approximates L,
  that is exp(Z) with Z ~ N(mz, sz).
  """
  # Based on Ho's Approximation for Schwartz-Yeh Approximation
  # IEEE TRANSACTIONS ON VEHICULAR TECHNOLOGY, VOL. 44. NO. 4, NOVEMBER 1995 
  # Calculating the Mean and Variance of Power Sums with Two Log-Normal 
  # Components by Chia-Lu Ho
  #
  # The Python code is based on original MatLab code by Takaki Makino (2011)
  #
  # Copyright (c) 2011 Takaki Makino. [http://www.snowelm.com/~t/]
  # Copyright (c) 2015 Nakamura Seminars. [http://nakamuraseminars.org/]
  #
  # This program is free software; you can redistribute it and/or modify
  # it under the terms of the GNU General Public License as published by
  # the Free Software Foundation; either version 3 of the License, or
  # (at your option) any later version.
  # This program is distributed in the hope that it will be useful,
  # but WITHOUT ANY WARRANTY; without even the implied warranty of
  # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  # GNU General Public License http://www.gnu.org/licenses/gpl.html 
  # for more details.  
  
  # setup constants for the numerical integration  
  xval = linspace(-3.6,3.6,25)  # this controls interval
  delta = (xval[-1] - xval[0])/(xval.size-1)
  uval = exp( -2 * sinh(xval) );
  vval = 1 / (1+uval); 
  zval = (2 * delta) * vval * uval * cosh(xval);

  idx = np.argsort(m)[::-1]
  sortmu = m[idx]
  sortsigma = s[idx][:,idx]

  # merge log-normals two-by-two
  while sortmu.size >= 2: 
    mw = sortmu[1] - sortmu[0];
    sigy12 = sortsigma[0,0] ;
    sigy22 = sortsigma[1,1] ;
    rhoy1y2sigy1sigy2 = sortsigma[0,1] ;
    sigw2 = sigy12 + sigy22 - 2*rhoy1y2sigy1sigy2;
    sigw = sqrt( sigw2 );
    eta = - mw / sigw;

    def fw(w): 
      return ( exp(-(w - mw)**2/(2*sigw2) ) / sqrt(2*pi*sigw2) )
    
    def h1234(v):
      logv = log(v);
      fwlogv = fw(logv);
      fwmlogv = fw(-logv);
      log1v = log(1+v);
      h1 = (fwlogv+fwmlogv)*log1v;
      return [h1, (fwlogv-fwmlogv)/(1+1.0/v), h1*log1v, -fwmlogv*log1v*logv]

    I = hstack(([erfc(sqrt(eta))], sum(h1234(vval) * zval, axis=1)))
    A0 = sigw / sqrt(2 * pi) * exp( - eta**2 / 2 ) + mw * I[0];
    G1 = A0 + I[1]
    G2 = I[3] + 2*I[4] + sigw2*I[0] + mw*A0
    G3 = sigw2*(I[2]+I[0])

    mz = sortmu[0] + G1; 
    sigz2 =  sigy12 - G1**2 + G2 + 2*(rhoy1y2sigy1sigy2 - sigy12)*(G3 / sigw2);
    
    newmu = hstack(([mz], sortmu[2:])); 
    diag3 = sqrt(sortsigma.diagonal()[2:]); 
    rhos = sortsigma[2:,0] / diag3; 
    rhosy = sortsigma[2:,1] / diag3; 
    newrhos = rhos + (rhosy - rhos) * (G3 / sigw2); 
    newsig = np.array([newrhos * diag3]); 
    newsigma = vstack(( hstack(( [[sigz2]], newsig )),
                        hstack(( newsig.transpose(), sortsigma[2:,2:] )) ))
    sortmu = newmu;
    sortsigma = newsigma;

  return sortmu[0], sortsigma[0][0]

# Examples that test out the code  
if __name__ == '__main__':
  import numpy as np
  import matplotlib.pyplot as plt
  from scipy.stats import norm
  
  mY = np.array([0.1, 0.2, 0.4, 0.3])
  sY = np.diag([0.3, 0.2, 0.6, 0.5])
  sY[[3,0],[0,3]] = 0.1  
  k = 4 # try k 1, 2, 3 or 4
  mY = mY[:k]
  sY = sY[:k][:,:k]
  
  for mZ, sZ in [mmFentonWilkinson(mY, sY), mmSchwartzYehHo(mY, sY)]:
    print "mZ, sZ: ", mZ, sZ 
    rZ = exp(np.random.normal(mZ, sqrt(mZ), 10000))
    rY = sum(exp(np.random.multivariate_normal(mY, sY, rZ.size)), axis=1)
    plt.figure()
    x = np.linspace(-2,2,100)
    plt.hist(log(rY), bins=30, normed = 1)
    plt.plot(x,norm.pdf(x,loc=mZ,scale=sqrt(sZ)))
  plt.show()

