# -*- coding: utf-8 -*-
"""
Created on Sun May 10 02:00:18 2020

@author: Lionel
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import itertools
from numba import jit
import hdbscan


def updateomega():
    '''
    

    Returns
    -------
    None.

    '''
    psx = np.random.normal(0,1,N)
    psy = np.random.normal(0,1,N)
    for i in range(N):
        vidx = rmat[i] <= re
        fidx = rmat[i] <= ro
        y = alpha*sum(vy[vidx,nxt-1][:3])+ beta*sum(fmat[i,fidx][:3])+ eta*psy[i]
        x = alpha*sum(vx[vidx,nxt-1][:3])+ beta*sum(fmat[i,fidx][:3])+ eta*psx[i]
        omega[i,nxt] = np.arctan(y/x)

    
def updatevmat():
    '''
    

    Returns
    -------
    None.

    '''
    vx[:,nxt] = vo*np.cos(omega[:,nxt])
    vy[:,nxt] = vo*np.sin(omega[:,nxt])
    
def updatexymat():
    '''
    

    Returns
    -------
    None.

    '''
    xmat[:,nxt] = xmat[:,nxt-1] + vx[:,nxt]
    ymat[:,nxt] = ymat[:,nxt-1] + vy[:,nxt]

def creatermat():
    '''
    

    Returns
    -------
    rmat : euclidean distance matrix between birds.

    '''
    data = np.array([xmat[:,nxt-1],ymat[:,nxt-1]]).T
    rmat = euclidean_distances(data)
    np.fill_diagonal(rmat,np.inf)
    return rmat
    
def createfmat():
    '''
    Input: ra, ro, re, and the distance matrix rmat

    Returns
    -------
    fmat : stores the pairwise repelling force between two birds.

    '''
    fmat= (rmat-re)/(4*(ra-re))
    # idx = rmat <= rc
    # fmat[idx] = -np.exp(100)
    idx = (rmat > ra)*(rmat<ro)
    fmat[idx] = 1
    idx = rmat > ro
    fmat[idx]=0
    np.fill_diagonal(fmat,0)
    return fmat

@jit(nopython=True)
def comato(i,j):
    return (vx[i,nxt-dta:nxt].mean()*vx[j,nxt-dta:nxt].mean()+vy[i,nxt-dta:nxt].mean()*vy[j,nxt-dta:nxt].mean())/np.sqrt((vx[i,nxt-dta:nxt].mean()**2+vy[i,nxt-dta:nxt].mean()**2)*(vx[j,nxt-dta:nxt].mean()**2+vy[j,nxt-dta:nxt].mean()**2))

        
T = 500 # number of time-steps

N = 2000 # number of agents

indices = list(itertools.combinations(range(N),2))
dta=1
omega = np.zeros((N,T))
xmat = np.zeros((N,T))
ymat = np.zeros((N,T))
vx = np.zeros((N,T))
vy = np.zeros((N,T))

''' initialization of system '''
alpha = 1
beta = 1
eta = 1
ra = 0.8
rc = .2
re = .5
ro = 1
vo=0.05

omega[:,0] = np.random.uniform(-np.pi,np.pi,N)
vx[:,0] = vo*np.cos(omega[:,0])
vy[:,0] = vo*np.sin(omega[:,0])
xmat[:,0] = np.random.uniform(0,1, N) + vx[:,0]
ymat[:,0] = np.random.uniform(0,1, N) + vy[:,0]
cor = np.zeros((N,N))
np.fill_diagonal(cor,1)

for nxt in range(1,T):
    
    ''' create the distance and force matrices'''
    rmat = creatermat()
    fmat = createfmat()
    
    ''' update the angular position'''
    updateomega()
    ''' update the velocity with the angular position '''
    updatevmat()
    ''' update the position vector '''
    updatexymat()

    data = np.array([xmat[:,nxt],ymat[:,nxt]]).T
    
    
    ''' save figures at every time step'''
    plt.figure()
    plt.scatter(xmat[:,nxt],ymat[:,nxt], s=10)
    plt.tight_layout()
    plt.savefig(f'gif/flock_{nxt}.png')

    plt.close()
    print(nxt)