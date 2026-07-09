# -*- coding: utf-8 -*-
#
# This program tests the generalised Schur algorithm to find the 
# Cholesky decomposition of the inverse covariance matrix.
#
#  This script is part of Hector 3.0.
#
#  Hector is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  any later version.
#
#  Hector is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with Hector. If not, see <http://www.gnu.org/licenses/>
#
# 28/6/2026 Machiel Bos
#===============================================================================

import os
import math
import cmath
import time
import sys
import re
import argparse
import numpy as np
from operator import add
from hector.powerlaw import Powerlaw
from numpy.linalg import inv
from numpy.polynomial import Polynomial
from math import sin,cos,pi
from itertools import zip_longest
import numpy.fft as npfft

#===============================================================================
# Subroutines
#===============================================================================


def multiply (a,b):

    #--- Determine which polynomial is longer
    if len(a)>len(b):
        m = len(a)
    else:
        m = len(b)


    #--- Algorithm only works for powers of 2 and zero padding
    n=1
    while n<m:
        n *= 2
    n *= 2

    #--- forward transform to point value
    Fa = npfft.rfft(a, n=n)
    Fb = npfft.rfft(b, n=n)

    # convolution
    #Fc = [ Fa [ i ] * Fb [ i ] for i in range ( n ) ]
    Fc = Fa* Fb 

    # reverse transform to get back coefficients
    #c = np.fft.irfft ( Fc )
    c = npfft.irfft(Fc)

    m = len(a)+len(b)-1
   
    #c = np.zeros(m)
    #for i in range(0,m):
    #    c[i] = final[i].real
        
    return c[0:m]

   
 
def levinson(t):
    ''' Use Durbin-Levinson algorithm to compute l1 and l2

    Args:
        t : array containing first column of Toeplitz matrix

    Returns:
        l1,l2 : arrays containing Schur polynomials
        delta : scale factor
    '''

    #--- Durbin-Levinson to compute l1 and l2
    m = len(t)
    r = np.zeros(m-1)
    delta = t[0]
    for i in range(0,m-1):
        if i==0:
            gamma = -t[i+1]/delta
        else:
            gamma = -(t[i+1] + np.dot(t[1:i+1],r[0:i]))/delta
            r[1:i+1] = r[0:i] + gamma*r[i-1::-1]

        r[0] = gamma
        delta = t[0] + np.dot(t[1:i+2],r[i::-1])

    #--- create l1 & l2 using r
    l1 = np.zeros(m)
    l2 = np.zeros(m)

    l1[0]   = 1.0
    l1[1:m] = r[m-2::-1]
    l2[1:m] = r[0:m-1]

    return [l1,l2,delta]


def schur(m):
    ''' Basic Fast Schur
    '''

    n = len(m)
    alpha = np.zeros((n,n))
    beta  = np.zeros((n,n))
    gamma = [1.0]*n

    for k in range(1,n):

        alpha[0,k-1] = m[k]
        beta[0,k-1]  = m[k-1]
        
        for j in range(1,k):
            alpha[j,k-j-1] = alpha[j-1,k-j] - gamma[j]*beta[j-1,k-j]
            beta[j,k-j]    = beta[j-1,k-j]  - gamma[j]*alpha[j-1,k-j]
 
        gamma[k] = alpha[k-1,0]/beta[k-1,0]
        beta[k,0] = beta[k-1,0]*(1.0 - pow(gamma[k],2.0))

    beta[0,n-1] = m[n-1]

    return [alpha,beta,gamma]



def schur2(m):
    '''
    '''
    n = len(m)
    t = np.zeros((n,n))
    gamma = [1.0]*n

    t[0,0] = m[0]
    for j in range(1,n):
        t[j,0] = m[j]
        s = m[j] 
        for k in range(1,j):
            t[j,k] = t[j-1,k-1] + gamma[k]*s
            s = s + gamma[k]*t[j-1,k-1]
        gamma[j] = -s/t[j-1,j-1]
        t[j,j] = t[j-1,j-1] * (1 - pow(gamma[j],2.0))

    #--- scale
    #for i in range(0,n):
    #    t[:,i] /= math.sqrt(t[i,i])

    return [t,gamma]


def schur_polynomials2(gamma):
    ''' Compute Schur polynomials of degree n
    '''

    n = len(gamma)

    xi    = Polynomial([0])
    eta   = Polynomial([1])
    xi_p  = Polynomial([0])
    eta_p = Polynomial([1])

    x   = Polynomial([0,1])
    one = Polynomial([1])

    S     = np.matrix([[eta_p,xi],[xi_p,eta]])
    for i in range(1,n):
        T = np.matrix([[x         ,gamma[i]*one],\
                       [x*gamma[i],one]])

        S_new = S @ T

        S = S_new

    eta_p = S[0,0]
    xi_p  = S[1,0]
    print('eta_p=',eta_p.convert().coef)
    print('xi_p=',xi_p.convert().coef)
    print(eta_p.convert().coef[0:-1] + xi_p.convert().coef[1:])


def schur_polynomials(gamma):
    ''' Compute Schur polynomials of degree n
    '''

    n     = len(gamma)
    S     = [[[1],[0]],[[0],[1]]]
    S_new = [[None,None],[None,None]]
    for i in range(1,n):
        T = [[[0,1],[gamma[i]]],\
             [[0,gamma[i]],[1.0]]]

        S_new[0][0] = [sum(j) for j in zip_longest( \
				multiply(S[0][0],T[0][0]), \
				multiply(S[0][1],T[1][0]), fillvalue=0.0)]
        S_new[1][0] = [sum(j) for j in zip_longest( \
				multiply(S[1][0],T[0][0]), \
				multiply(S[1][1],T[1][0]), fillvalue=0.0)]

        #--- Stupid copy() operator needs to be applied to each element
        S[0][0] = S_new[0][0].copy()
        S[1][1] = S_new[0][0][-1:0:-1].copy()
        S[1][0] = S_new[1][0].copy()
        S[0][1] = S_new[1][0][-1:0:-1].copy()
        #print('eta   = ',S[1][1])
        #print('eta_p = ',S[0][0])
        #print('xi    = ',S[0][1])
        #print('xi_p  = ',S[1][0])

    eta_p = S[0][0]
    xi_p  = S[1][0]
    print('psi_n = ',list( map(add,eta_p[0:-1],xi_p[1:]))) 



def generalised_schur(tm,p_0,q_0):
    ''' Generalised Schur Algorithm

    '''

    if tm==1:
        gamma = -p_0[0]/q_0[0]
        delta_tm = 1.0 - pow(gamma,2.0) 
        a_tm = np.array([0.0,1.0])
        c_tm = np.array([0.0,-gamma])
        return [a_tm,c_tm,delta_tm]
    else:
        m = tm//2
        [a_m,c_m,delta_m] = generalised_schur(m,p_0[0:m],q_0[0:m])
 
        #--- Construct b_m and d_m
        d_m = a_m[-1:0:-1].copy() 
        b_m = c_m[-1:0:-1].copy() 

        #--- Multiply polynomials
        #start_time = time.time()
        part1 =       multiply(d_m,p_0) 
        part2 =  -1 * multiply(b_m,q_0)
        #if m>3200:
        #    print("1 - multiply m=",m,', : {0:8.3f} s n'.\
	#			format(float(time.time() - start_time)))
        #start_time = time.time()
        p_m   = [sum(j) for j in zip_longest(part1,part2, fillvalue=0.0)]
        #if m>3200:
        #    print("1 - sum m=",m,', : {0:8.3f} s n'.\
	#			format(float(time.time() - start_time)))
        
        #start_time = time.time()
        part1 = -1 * multiply(c_m,p_0)
        part2 =      multiply(a_m,q_0)
        #if m>3200:
        #    print("2 - multiply m=",m,', : {0:8.3f} s n'.\
	#			format(float(time.time() - start_time)))
        q_m   = [sum(j) for j in zip_longest(part1,part2, fillvalue=0.0)]
        
        [a_mtm,c_mtm,delta_mtm] = generalised_schur(m,p_m[m:tm],q_m[m:tm])

        #--- Multiply polynomials
        part1 = multiply(a_m,a_mtm)
        part2 = multiply(b_m,c_mtm)
        a_tm  = [sum(j) for j in zip_longest(part1,part2, fillvalue=0.0)]
        
        part1 = multiply(c_m,a_mtm)
        part2 = multiply(d_m,c_mtm)
        c_tm  = [sum(j) for j in zip_longest(part1,part2, fillvalue=0.0)]

        delta_tm = delta_m * delta_mtm
  
        return [a_tm,c_tm,delta_tm]


#===============================================================================
# Main program
#===============================================================================

def main():

    nn = 4*1024 + 1
    noise_model = Powerlaw()
    [t,k_new] = noise_model.create_t(nn,0,[-0.8])
    #C = np.zeros((nn,nn))
    #for i in range(0,nn):
    #    for j in range(0,nn-i):
    #        C[i+j,i] = t[j]
    #        C[i,i+j] = t[j]

    #print('\nmatrix C:\n',C)
    #print('\ninv(C):\n',inv(C))
    #U = np.linalg.cholesky(C)
    #print('\nU:\n',U)
    #print('\ninv(U):\n',inv(U))

    #--- Levinson-Durbin
    print('\nLevingson-Durbin')
    start_time = time.time()
    [l1,l2,delta] = levinson(t)
    print("Levinson : {0:12.6f} s n".format(float(time.time() - start_time)))
    print("l1 = ",l1)

    #L1 = np.zeros((nn,nn))
    #L2 = np.zeros((nn,nn))
    #for i in range(0,nn):
    #    for j in range(0,nn-i):
    #        L1[i+j,i] = l1[j]
    #        L2[i+j,i] = l2[j]
   
    #print("delta = ",delta)
    #print('\nL1 = \n',L1) 
    #print('\nL2 = \n',L2) 
    #print("1/delta * (L1*L1' - L2*L2')\n",1/delta*(L1@L1.T-L2@L2.T))


    #--- Schur
    #[alpha,beta,gamma] = schur(t)
    #print('alpha = \n',alpha)
    #print('beta  = \n',beta)
    #print('gamma = \n',gamma)

    #---
    #[tt,gamma] = schur2(t)
    #print('\nU\n',U,'\n')
    #print(tt,gamma)
    #detC = 1.0
    #for i in range(0,len(gamma)):
    #    detC *= beta[i][0]
    #print('detC=',detC,', ',np.linalg.det(C))

    #--- Schur polynomials
    #schur_polynomials(gamma)
    #schur_polynomials2(gamma)

    #--- Generalised Schur
    #p_0 = [0.0]*nn
    #p_0[0:nn-1] = t[1:nn]
    #q_0 = np.array(t)

    #--- remember this one... gamma's are good
    xi  = np.zeros(nn-1)
    p_0 = np.zeros(nn)
    p_0[0:nn-1] = np.array(t[1:])
    q_0 = np.array(t)
    tm  = len(t)
    #print('\n\n\nt = ',t)
    #print('p_0 = ',p_0)
    #print('q_0 = ',q_0)
    start_time = time.time()
    [a_tm,c_tm,delta_tm] = generalised_schur(tm-1,p_0,q_0)
    print("GSA : {0:12.6f} s n".format(float(time.time() - start_time)))
    print('delta_tm = ',delta_tm)
    for i in range(0,nn-1):
        xi[nn-2-i]=a_tm[i] - c_tm[i+1]
    print(xi)

    #--- Show time lapsed


#--- Call main program
if __name__ == "__main__":
    main()


