"""
Collects some simple classes for estimation in Python, including: Tobit,....
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy import optimize
# Plotting features require:
import matplotlib as mpl
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
plt.style.use('seaborn-whitegrid')
mpl.style.use('seaborn')
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

class Tobit():
  """
  """
  def __init__(self,x=None,y=None,**kwargs):
    self.x = x
    self.y = y
    self.base_par()
    self.upd_par(kwargs)

  def base_par(self):
    # Options to pass to minimizer
    self.intercept=True
    self.method = 'Nelder-Mead'
    self.jac = None
    self.hess = None
    self.hessp = None
    self.bounds=None
    self.constraints=None
    self.tol=None
    self.callback=None
    self.options=None

  def upd_par(self,kwargs):
  	for key,value in kwargs.items():
  		setattr(self,key,value)

  @staticmethod
  def tobit_ll(y,x,theta,intercept=True):
  	if intercept:
  		X = np.append(np.ones([x.shape[0],1]),x,axis=1)
  	else:
  		X = x
  	return -sum((y==0) * np.log(1-stats.norm.cdf(np.matmul(X,theta[:-1])/theta[-1])) + (y>0) * np.log(stats.norm.pdf((y-np.matmul(X,theta[:-1]))/theta[-1])/theta[-1]))

  @staticmethod
  def tobit_sim(theta,X=None,N=100):
    if X:
      y = np.maximum(np.matmul(X,theta[:-1])+np.random.normal(0,theta[-1],X.shape[0]),0)
    else:
      y = np.maximum(np.matmul(np.append(np.ones([N,1]), np.random.normal(0,1,[N,theta.shape-1]),axis=1),theta[:-1])+np.random.normal(0,theta[-1],N),0)

  @staticmethod
  def tobit_novariance(theta,x,intercept=True):
    if intercept:
      return np.maximum(np.matmul(np.append(np.ones([x.shape[0],1]), x,axis=1), theta[:-1]),0)
    else:
      return np.maximum(np.matmul(x,theta[:-1]),0)

  @staticmethod
  def tobit_censor_prob(theta,x,intercept=True):
    if intercept:
      return 1-stats.norm.cdf(np.matmul(np.append(np.ones([x.shape[0],1]),x,axis=1),theta[:-1])/theta[-1])
    else:
      return 1-stats.norm.cdf(np.matmul(x,theta[:-1])/theta[-1])

  @staticmethod
  def tobit_cond_exp(theta,x,intercept=True):
    if intercept:
      X = np.append(np.ones([x.shape[0],1]),x,axis=1)
    else:
      X = x
    return np.matmul(X,theta[:-1])*stats.norm.cdf(np.matmul(X,theta[:-1])/theta[-1])+theta[-1]*stats.norm.pdf(np.matmul(X,theta[:-1])/theta[-1])

  @staticmethod
  def tobit_cond_exp_pos(theta,x,intercept=True):
    if intercept:
      X = np.append(np.ones([x.shape[0],1]),x,axis=1)
    else:
      X = x
    return np.matmul(X,theta[:-1])+theta[-1]*(stats.norm.pdf(np.matmul(X,theta[:-1])/theta[-1])/stats.norm.cdf(np.matmul(X,theta[:-1])/theta[-1]))

  def max_ll(self,theta0):
    sol = optimize.minimize(lambda theta: self.tobit_ll(self.y,self.x,theta,self.intercept),theta0,
                            method=self.method,
                            jac=self.jac,
                            hess=self.hess,
                            hessp=self.hessp,
                            bounds=self.bounds,
                            constraints=self.constraints,
                            tol = self.tol,
                            callback=self.callback,
                            options = self.options);
    if not sol['success']:
      print(sol['message'])
      return sol
    else:
      self.theta = sol['x']

  def predict(self):
    self.predicted = Tobit.tobit_cond_exp(self.theta,self.x,self.intercept)
    self.predicted_pos = Tobit.tobit_cond_exp(self.theta,self.x,self.intercept)

  def plot_fit(self):
    if self.intercept:
      X = np.append(np.ones([self.x.shape[0],1]),self.x,axis=1)
    else:
      X = self.x
    y_not_censored = np.matmul(X,self.theta[:-1])
    Data = pd.DataFrame(np.transpose(np.array([y_not_censored,self.y])),columns=['$x\\beta$','Data'])
    Predicted = pd.DataFrame(np.transpose(np.array([y_not_censored,self.predicted,self.predicted_pos])), columns=['$x\\beta$','$E[y|x]$','$E[y|x,y>0]$']).sort_values(by='$x\\beta$')
    fig = plt.figure(frameon=False,figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    ax.scatter(y_not_censored,self.y,s=5,alpha=0.9)
    ax.plot(Predicted['$x\\beta$'],Predicted['$E[y|x]$'],c=colors[1],linewidth=2)
    ax.plot(Predicted['$x\\beta$'],Predicted['$E[y|x,y>0]$'],c=colors[2],linewidth=2)
    ax.set_ylabel('Data, predicted', fontsize=13)
    ax.set_xlabel('$x\\beta$',fontsize=13)
    plt.legend(('$E[y|x]$','$E[y|x,y>0]$','Data'),fontsize=12,frameon=True)
    plt.title('Tobit model fit', fontsize=14)
    fig.tight_layout()

  def plot_fit_scatter(self):
    return pd.DataFrame(np.transpose(np.array([self.predicted, self.y])), columns=['Predicted','Data']).sort_values(by='Predicted').plot.scatter(x='Data',y='Predicted')

  def plot_fit_pos_scatter(self):
    return pd.DataFrame(np.transpose(np.array([self.predicted_pos, self.y])), columns=['Predicted, given y>0','Data']).sort_values(by='Predicted, given y>0').plot.scatter(x='Data',y='Predicted, given y>0')


class D_Tobit():
  """
  Double-censored tobit estimation, with outer-discrete variable on lower and/orupper.
  Note that we nest (1) standard OLS, (2) Standard tobit, (3) double-censored tobit,
  and combinations of the above with an unrelated discrete shock added.
  """

  def __init__(self,x=None,y=None,lower=None,upper=None,**kwargs):
    self.x = x
    self.y = y
    self.lower = lower
    self.upper = upper
    self.base_par()
    self.upd_par(kwargs)

  def base_par(self):
    # Options to pass to minimizer
    self.intercept=True
    self.method = None
    self.jac = None
    self.hess = None
    self.hessp = None
    self.bounds=None
    self.constraints=None
    self.tol=None
    self.callback=None
    self.options=None

  @property
  def args(self):
    """Dictionary of estimated parameters"""
    return {'theta': self.theta, 'beta': self.theta[:-3], 'sigma': self.theta[-3], 'pl': self.theta[-2], 'ph': self.theta[-1]}

  @staticmethod
  def theta2args(theta,lower,upper):
    if lower==None and upper==None:
      return {'beta': theta[:-1], 'sigma': theta[-1]}
    if lower!=None and upper==None:
      return {'beta': theta[:-2], 'sigma': theta[-2], 'pl': theta[-1]}
    if lower==None and upper!=None:
      return {'beta': theta[:-2], 'sigma': theta[-2], 'ph': theta[-1]}
    if lower!=None and upper!=None:
      return {'beta': theta[:-3], 'sigma': theta[-3], 'pl': theta[-2], 'ph': theta[-1]}

  @staticmethod
  def discrete_sample(theta,n,lower,upper):
    if lower!=None and upper==None:
      return stats.rv_discrete(values=((0,1), (1-D_Tobit.theta2args(theta,lower,upper)['pl'],D_Tobit.theta2args(theta,lower,upper)['pl']))).rvs(size=n)
    if lower==None and upper!=None:
      return stats.rv_discrete(values=((0,1), (1-D_Tobit.theta2args(theta,lower,upper)['ph'],D_Tobit.theta2args(theta,lower,upper)['ph']))).rvs(size=n)
    if lower!=None and upper!=None:
      return stats.rv_discrete(values=((0,1,2), (1-D_Tobit.theta2args(theta,lower,upper)['pl']-D_Tobit.theta2args(theta,lower,upper)['ph'],D_Tobit.theta2args(theta,lower,upper)['pl'],D_Tobit.theta2args(theta,lower,upper)['ph']))).rvs(size=n)

  @staticmethod
  def x2X(x,intercept):
    if intercept:
      return np.append(np.ones([x.shape[0],1]),x,axis=1)
    else:
      return x 

  @staticmethod
  def args2theta(beta,sigma,pl,ph,lower,upper):
    if lower==None and upper==None:
      return np.append(beta,sigma)
    if lower!=None and upper==None:
      return np.append(beta,[sigma,pl])
    if lower==None and upper!=None:
      return np.append(beta,[sigma,ph])
    if lower!=None and upper!=None:
      return np.append(beta,[sigma,pl,ph])

  def upd_par(self,kwargs):
    for key,value in kwargs.items():
      setattr(self,key,value)

  @staticmethod
  def ll(y,x,theta,lower,upper,intercept=True):
    X = D_Tobit.x2X(x,intercept)
    if lower==None and upper==None:
      return -sum( np.log(stats.norm.pdf((y-np.matmul(X,theta[:-1]))/theta[-1])/theta[-1]) )
    if lower!=None and upper==None:
      return -sum((y==lower)*np.log(theta[-1]+(1-theta[-1])*(1-stats.norm.cdf((np.matmul(X,theta[:-2])-lower)/theta[-2]))) + 
                (y>lower) * np.log((1-theta[-1])* stats.norm.pdf((y-np.matmul(X,theta[:-2]))/theta[-2])/theta[-2]))
    if lower==None and upper!=None:
      return -sum((y<upper) * np.log((1-theta[-1])* stats.norm.pdf((y-np.matmul(X,theta[:-2]))/theta[-2])/theta[-2]) +
                (y==upper) * np.log(theta[-1]+(1-theta[-1])*stats.norm.cdf((upper-np.matmul(X,theta[:-2]))/theta[-2])))
    if lower!=None and upper!=None:
      return -sum((y==lower)*np.log(theta[-2]+(1-theta[-2]-theta[-1])*(1-stats.norm.cdf((np.matmul(X,theta[:-3])-lower)/theta[-3]))) + 
                (y>lower)*(y<upper) * np.log((1-theta[-2]-theta[-1])* stats.norm.pdf((y-np.matmul(X,theta[:-3]))/theta[-3])/theta[-3]) +
                (y==upper) * np.log(theta[-1]+(1-theta[-2]-theta[-1])*stats.norm.cdf((upper-np.matmul(X,theta[:-3]))/theta[-3])))


  @staticmethod
  def sim_static(theta,lower,upper,x=None,N=100,intercept=True):
    if x is not None:
      X = D_Tobit.x2X(x,intercept)
      N = X.shape[0]
    else:
      X = np.append(np.ones([N,1]),np.random.normal(0,1,[N,D_Tobit.theta2args(theta,lower,upper)['beta'].shape-1]))
    d = D_Tobit.discrete_sample(theta,X.shape[0],lower,upper)
    if lower==None and upper==None:
      y = np.matmul(X,D_Tobit.theta2args(theta,lower,upper)['beta'])+np.random.normal(0,D_Tobit.theta2args(theta,lower,upper)['sigma'],N)
    if lower!=None and upper==None:
      y = (d==1) * lower + (d==0) * np.maximum(np.matmul(X,D_Tobit.theta2args(theta,lower,upper)['beta'])+np.random.normal(0,D_Tobit.theta2args(theta,lower,upper)['sigma'],N),lower)
    if lower==None and upper!=None:
      y = (d==1) * upper + (d==0) * np.minimum(np.matmul(X,D_Tobit.theta2args(theta,lower,upper)['beta'])+np.random.normal(0,D_Tobit.theta2args(theta,lower,upper)['sigma'],N),upper)
    if lower!=None and upper!=None:
      y = (d==2) * upper + (d==1) * lower + (d==0) * np.maximum(np.minimum(np.matmul(X,D_Tobit.theta2args(theta,lower,upper)['beta'])+np.random.normal(0,D_Tobit.theta2args(theta,lower,upper)['sigma'],N),lower),upper)
    return y
    
  def max_ll(self,beta,sigma=1,pl=0,ph=0):
    sol = optimize.minimize(lambda theta: self.ll(self.y,self.x,theta,self.lower,self.upper,self.intercept),
                            self.args2theta(beta,sigma,pl,ph,self.lower,self.upper),
                            method=self.method,
                            jac=self.jac,
                            hess=self.hess, 
                            hessp=self.hessp,
                            bounds=self.bounds,
                            constraints=self.constraints,
                            tol = self.tol,
                            callback=self.callback,
                            options = self.options);
    if not sol['success']:
      print(sol['message'])
      return sol
    else:
      self.theta = sol['x']
      return sol

  def sim(self):
    return D_Tobit.sim_static(self.theta,self.lower,self.upper,self.x,self.x.shape[0],self.intercept)

  def sim_(self,x):
    return D_Tobit.sim_static(self.theta,self.lower,self.upper,x,x.shape[0],self.intercept)

  @property
  def view_theta(self):
    """Dictionary of estimated parameters"""
    return {'theta': self.theta, 'beta': self.theta[:-3], 'sigma': self.theta[-3], 'pl': self.theta[-2], 'ph': self.theta[-1]}
