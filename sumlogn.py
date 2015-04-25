from numpy import log, exp, newaxis, sum, sqrt, matrix
from numpy import linspace, sinh, cosh, hstack, vstack, trapz, pi
from scipy.special import erfc

def mmFentonWilkinson(m, s):
  """
  Implements Fenton-Wilkinson approach for sum of lognormals
  assume Z = sum Y_i, with Y_i = exp(m_i + X_i * s_i)  
  m is the vector of means
  s is the covariance matrix
  of X ~ N(m, s)
  output mean and stdev for lognormal that approximates Z
  Z approx exp(mz + W * sz), with W ~ N(mz, sz)
  """
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
#%%if True:
  xval = linspace(-3.6,3.6,25)  # this controls the integration interval
  delta = (xval[-1] - xval[0])/(xval.size-1)
  uval = exp( -2 * sinh(xval) );
  vval = 1 / (1+uval); 
  zval = (2 * delta) * vval * uval * cosh(xval);

  idx = np.argsort(m)[::-1]
  sortmu = m[idx]
  sortsigma = s[idx][:,idx]
#%%
  # merge two-by-two
  while sortmu.size >= 2: 

#%%if True:
    mw = sortmu[1] - sortmu[0];
    sigy12 = sortsigma[0,0] ;
    sigy22 = sortsigma[1,1] ;
    rhoy1y2sigy1sigy2 = sortsigma[0,1] ;
    sigw2 = sigy12 + sigy22 - 2*rhoy1y2sigy1sigy2;
    sigw = sqrt( sigw2 );

#%%if True:
    #print "mw,sigw: ", mw, sigw
    eta = - mw / sigw; #print eta
    I0 = erfc( sqrt(eta) ) 

    def fw(w): 
      return ( exp(-(w - mw)**2/(2*sigw2) ) / sqrt(2*pi*sigw2) )
    
    def h1234(v):
      logv = log(v);
      fwlogv = fw(logv);
      fwmlogv = fw(-logv);
      log1v = log(1+v);
      h1 = (fwlogv+fwmlogv)*log1v;
      return [h1, (fwlogv-fwmlogv)/(1+1.0/v), h1*log1v, -fwmlogv*log1v*logv]

    #print I0, sum(h1234(vval) * zval, axis=1)   
    I = hstack(([I0], sum(h1234(vval) * zval, axis=1)))
    
#%%if True:
    A0 = sigw / sqrt(2 * pi) * exp( - eta**2 / 2 ) + mw * I[0];
    G1 = A0 + I[1]
    G2 = I[3] + 2*I[4] + sigw2*I[0] + mw*A0
    G3 = sigw2*(I[2]+I[0])

#%%if True:  
    mz = sortmu[0] + G1; 
    sigz2 =  sigy12 - G1**2 + G2 + 2*(rhoy1y2sigy1sigy2 - sigy12)*(G3 / sigw2);

    #print "mz, sortmu: ", mz, sortmu
    newmu = hstack(([mz], sortmu[2:])); print newmu
    if sortmu.size > 2:
      diag3 = sqrt(sortsigma.diagonal()[2:]); #print diag3
      rhos = sortsigma[2:,0] / diag3; #print rhos
      rhosy = sortsigma[2:,1] / diag3; #print rhosy
      newrhos = rhos + (rhosy - rhos) * (G3 / sigw2); #print newrhos
      newsig = np.array([newrhos * diag3]); #print newsig

      #newsig0 = sortsigma[2:,2:]; print "newsig0", newsig0
      #newsig1 = hstack(([[sigz2]], newsig)); print newsig1
      #newsig2 = hstack((newsig.transpose(), newsig0)); print newsig2
      #newsigma = vstack((newsig1, newsig2)); print newsigma
      newsigma = vstack(( hstack(( [[sigz2]], newsig )),
                          hstack(( newsig.transpose(), sortsigma[2:,2:] )) ))
    else:
      newsigma = [[sigz2]]
    sortmu = newmu;
    sortsigma = newsigma;
#%%
  return sortmu[0], sortsigma[0][0]
#%%
  
if __name__ == '__main__':
  import numpy as np
  import matplotlib.pyplot as plt
  from scipy.stats import norm
  
  mY = np.array([0.1, 0.2, 0.4, 0.3])
  sY = np.diag([0.3, 0.2, 0.6, 0.5])
  sY[[3,0],[0,3]] = 0.1  
  
  #mZ, sZ = mmFentonWilkinson(mY, sY)
  mZ, sZ = mmSchwartzYehHo(mY, sY)
  print "mZ, sZ: ", mZ, sZ 
  rZ = exp(np.random.normal(mZ, sqrt(mZ), 10000))
  rY = sum(exp(np.random.multivariate_normal(mY, sY, rZ.size)), axis=1)
  
  x = np.linspace(-2,2,100)
  plt.hist(log(rY), bins=30, normed = 1)
  plt.plot(x,norm.pdf(x,loc=mZ,scale=sqrt(sZ)))

  plt.show()

