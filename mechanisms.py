#MIT License

#Copyright (c) 2022 Michael Shekelyan, King's College London

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

# This code has been tested with Python 3.9.9

# Some comments in the code may be outdated

from mpmath import mp # code was not tested without this library

import json
import sys
import random
from array import array
import argparse
import itertools

from math import sqrt,floor, ceil, log, exp,inf
import math
from random import uniform, sample
import heapq

import numpy as np
import os
from io import StringIO
import time
from lib import Arithmetic, Top, strformat, between

num = Arithmetic(256) # code was note tested with smaller than 256 bit precision

# generate random subset from class C_(h,t)

def random_ht(k,ht):

	(h,t) = ht

	assert (h >= 0)
	assert (h < k)
	
	assert (t >= k)

	ret = []

	if t == k:
		ret = list(between(1, k))
		
	else:

		if h >= 1:
			ret = list(between(1, h))
		
		if t > (h+2):
			ret.extend( sorted(random.sample(between(h+2, t-1), k-h-1 )) )
		
		ret.append(t)
	
	assert (len(ret) == k)
	
	return ret

############# PEELING #############

# Select subset with Peeling Lipschitz Mechanism in O(dk)

def peeling(x, k, eps, Finv):

	assert (x[0] == -inf)
	d = len(x)-1
	keps = eps/float(k)
	
	topk = []
	
	for ii in between(1,k):
		
		i = 0; itop = -1; qtop = -inf
		
		for t in between(1,d):
		
			if i < len(topk) and t == topk[i]:
				
				i += 1
				continue
			
			q = x[t]*keps + Finv(num, num.unif(0,1))
			
			if q > qtop:
			
				itop = t
				qtop = q
		
		topk.append( itop )
		topk.sort()
	
	return topk
	
def peelingExponentialMechanism(x, k, eps = 1):

	return peeling(x,k,eps, lambda num, p : num.qgumbel(p) )

def peelingNoisyReportMax(x, k, eps = 1):

	return peeling(x, k, eps, lambda num, p : num.qlaplace(p) )

def peelingPermuteAndFlip(x, k, eps = 1):

	return peeling(x, k, eps, lambda num, p : num.qexp(p) )

############# ONESHOT #############

# Select subset with (Oneshot) Lipschitz Mechanism in O(d log k)

def lipschitz(x, k, eps, Finv):

	assert (x[0] == -inf) # x[1] is first index
	d = len(x)-1
	
	keps = eps/float(k)

	topk = Top(k)
				
	for t in between(1,d):
		q = x[t]*keps + Finv(num, num.unif(0,1))
		topk.digest(q, t)		
	
	ret = topk.argmaxs()
	
	return ret

def gumbelLipschitz(x, k, eps = 1):

	return lipschitz(x,k,eps,lambda num, p : num.qgumbel(p))

def laplaceLipschitz(x, k, eps = 1):
	
	return lipschitz(x, k, eps, lambda num,p : num.qlaplace(p))

def expLipschitz(x, k, eps = 1):
	
	return lipschitz(x, k, eps, lambda num,p : num.qexp(p))



############# CANONICAL WITH GAMMA = 1 #############

# O(dk) for Pr(h,t) distribution of Canonical Lipschitz with gamma = 1
def probsCanonicalGamma1Gumbel(x, k, eps, rho): 

	assert (rho >= 0.0)
	assert (rho <= 1.0)
	assert (x[0] == -inf)
	d = len(x)-1
	
	#lim = 0.00001/(d*k)
	
	A = [[mp.mpf(0.0) for t in between(0, d)] for h in between(0, k-1)]
	T = [mp.mpf(0.0) for t in between(0, d)]
	
	eps1 = (1.0-rho)*eps
	eps2 = rho*eps
	
	s = mp.mpf(0.0)
	
	T[k] = mp.exp(x[k]*eps2)
	s = s+T[k]
	mlog = 0.0	# t = k => choose(t-1, k-1) = 1

	for t in between(k+1, d):
	
		nn = t-1 # t -= 1 => nn += 1
		kk = k-1 #
		mlog += log(nn)-log(nn-kk) # choose(nn,kk) = nn/(nn-kk) * choose(nn-1, kk)
	
		T[t] = mp.exp(mlog+x[t]*eps2)
		s = s+T[t]
	
	# T[h] propto m * exp( x[t]*rho*eps )
	for t in between(k, d):
		T[t] = T[t] / s
			
	A[k-1][k] = T[k]
	
	for t in between(k+1,d):
	
		H = [mp.mpf(0.0) for h in between(0, k-1)]
		s = mp.mpf(0.0)
		
		h = k-1
		H[h] = mp.exp(-x[h+1]*eps1)
		s = s+H[h]
		
		mlog = 0.0 # h = k-1 => choose(t-(h+2),k-(h+1)) = choose(...,0) = 1
		for h in between(k-2, 0):
	
			nn = t-(h+2) # h -= 1 => nn += 1
			kk = k-(h+1) # h -= 1 => kk += 1
			mlog += log(nn)-log(kk) # choose(nn,kk) = nn/kk * choose(nn-1, kk-1)
			
			H[h] = mp.exp(mlog-x[h+1]*eps1)
			s = s+H[h]
		
		# H[h] propto m * exp( -x[h+1]*(1-rho)*eps )
		for h in between(k-1, 0):
			A[h][t] = (H[h]/s)*T[t]
	
	return A
	

# Select subset with CANONICAL(gamma=1) in O(d) for pre-sorted input (sometimes called tail)

def canonicalGamma1Lipschitz(x, k, eps, Finv, rho):
	
	assert (x[0] == -inf) # x[1] is first index
	d = len(x)-1
	
	assert (rho >= 0)
	assert (rho <= 1)
	
	
	eps1 = (1.0-rho)*eps
	eps2 = rho*eps
	
	top = k
	qtop = x[k]*eps2+Finv(num, num.unif(0,1), 0)
	
	mlog = 0.0 # t == k => choose(t-1,k-1) = choose(k-1,k-1) = 1
	
	for t in between(k+1,d):	
	
		nn = t-1
		kk = k-1
		
		mlog += log(nn)-log(nn-kk)
		
		if nn < kk:
			continue
		
		q = x[t]*eps2 + Finv(num, num.unif(0,1), mlog)
		
		if q > qtop:
			top = t
			qtop = q
	
	t = top
	
	if t == k:
		
		h = k-1
		return random_ht(k,(h,t))
		
	else:
		
		h = k-1
		top = h
		qtop = ( -x[h+1] )*eps1+Finv(num, num.unif(0,1), 0)
		
		mlog = 0.0 # h = k-1 => choose(t-(h+2),k-(h+1)) = choose(...,0) = 1
		for h in between(k-2, 0):

			nn = t-(h+2) # h -= 1 => nn += 1
			kk = k-(h+1) # h -= 1 => kk += 1
			mlog += log(nn)-log(kk) # choose(nn,kk) = nn/kk * choose(nn-1, kk-1)
			
			q = -x[h+1]*eps1 + Finv(num, num.unif(0,1), mlog)
			
			if q > qtop:
				top = h
				qtop = q
		
		h = top
		return random_ht(k,(h,t))

############# CANONICAL #############

# Select subset with CANONICAL in O(dk) for pre-sorted input

def canonicalLipschitz(x, k, eps, Finv, rho):
	
	assert (x[0] == -inf) # x[1] is first index
	d = len(x)-1
	
	assert (rho >= 0)
	assert (rho <= 1)
	
	eps1 = (1.0-rho)*eps
	eps2 = rho*eps
	
	top = (k-1,k)
	qtop = x[k]*(eps2-eps1) +Finv(num, num.unif(0,1), 0)
	
	for t in between(k+1,d):	
	
		mlog = 0.0
	
		for h in between(k-1,0):
		
			nn = t-(h+2)
			kk = k-(h+1)
			
			if kk > 0:
				mlog += log(nn)-log(kk)
			
			if nn < kk:
				continue
			
			q = x[t]*eps2-x[h+1]*eps1 + Finv(num, num.unif(0,1), mlog)
			
			if q > qtop:
				top = (h,t)
				qtop = q
	
	(h,t) = top
	return random_ht(k,(h,t))
	
def canonicalLipschitzGumbel(x, k, eps):
	
	return canonicalLipschitz(x,k,eps,lambda num, p, mlog : num.qgumbelmax(p,mlog))

def canonicalLipschitzExp(x, k, eps):
	
	return canonicalLipschitz(x,k,eps,lambda num, p, mlog : num.qexpmax(p,mlog))


# O(dk) for Pr(h,t) distribution of Canonical Lipschitz

def probsCanonicalGumbel(x, k, eps, rho=0.5):
	
	assert (rho >= 0.0)
	assert (rho <= 1.0)
	assert (x[0] == -inf)
	d = len(x)-1
	
	A = [[mp.mpf(0.0) for t in between(0, d)] for h in between(0, k-1)]
	
	eps1 = (1.0-rho)*eps
	eps2 = rho*eps
	
	# y = OPT, i.e., h+1 = t = k
	A[k-1][k] = mp.exp(-x[k]*(eps1-eps2) )
	s = mp.mpf(0.0)
	s = s+A[k-1][k]
	
	for t in between(k+1, d):
	
		mlog = 0.0 # choose(t-k-1, 0) = 1
		
		h = k-1
		A[h][t] = mp.exp(-(eps1*x[h+1] - eps2*x[t]))
		s = s+A[h][t]
		
		mlog = 0.0 # h = k-1 => choose(t-(h+2),k-(h+1)) = choose(...,0) = 1
		
		for h in between(k-2, 0):
			
			nn = t-(h+2) # h -= 1 => nn += 1
			kk = k-(h+1) # h -= 1 => kk += 1
			mlog += log(nn)-log(kk) # choose(nn,kk) = nn/kk choose(nn-1, kk-1)
			
			A[h][t] = mp.exp(mlog-(eps1*x[h+1] - eps2*x[t]))
			s = s+A[h][t]
	
	#  A[h][t] propto m * exp( ( (1-rho) * x[h+1] - rho * x[t])*eps )
	A[k-1][k] = A[k-1][k] / s
	for t in between(k+1, d):
		for h in between(k-1, 0):
			A[h][t] = A[h][t]/s
	
	return A
