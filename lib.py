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

from mpmath import mp # This has not been tested without this library

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

strformat = lambda s, *args, **kw : s.replace("{","!(!").replace("}","!)!").replace("@","{}").replace("<<<","{").replace(">>>","}").format(*args, **kw).replace("!(!","{").replace("!)!","}")

# between(a,b) is an iterator for values between a and b
between = lambda a,b : range(a,b+1) if b >= a else range(a,b-1,-1)

class Arithmetic:

	def __init__(self, bits=1024):
	
		self.mp = mp.clone()
		self.mp.prec = bits

	val = lambda self, x : float(x)
	log = lambda self, x : math.log(x)
	
	absval = lambda self, x : math.abs(x) 
	exp = lambda self, x : math.exp(x)
	equal = lambda self, x, y: ( self.absval(x-y) == 0 ) or ( self.log(self.absval(x-y)) < self.val(-10) )
	
	# U sim Unif(0,1)
	unif = lambda self, a,b : self.val(uniform(a,b))
	
	# F^-1 for Unif
	qunif = lambda self, p : p
	
	# F^-1 for Max Order Statistic of Unif
	qunifmax = lambda self, p, m=1: p**(1.0/m)
	
	# F^-1 for Exponential
	qexp = lambda self, p, rate=1: -self.log(1.0-p)/rate
	
	# F^-1 for Logistic
	qlog = lambda self, p, rate=1: ( self.log(p)-self.log(1.0-p) )/rate
	
	# F^-1 for Halflogistic
	qhalflog = lambda self, p, rate=1: ( self.log(1.0+p)-self.log(1.0-p) )/rate
	
	# F^-1 for Gumbel
	qgumbel = lambda self, p, mu=0,b=1: mu-b*self.log(-self.log(p))
	
	# F^-1 for Laplace
	qlaplace = lambda self, p, mu=0, b=1 : mu+b*self.log(self.val(2.0)*p) if p <= 0.5 else mu-b*self.log(self.val(2.0)-self.val(2.0)*p)

	def fsum(self, a,b ):
	
		return self.mp.fadd(a,b)

	def flog(self, a):
	
		return self.mp.log(a)

	def fexp(self, a):
	
		return self.mp.exp(a)

	def fnum(self, a):
		
		return self.mp.mpf(a)
	
	def fabs(self, a):
		
		return self.mp.fabs(a)
	
	def froot(self, x, p):
	
		return self.fexp( self.flog(x)/self.fnum(p) )
	
	def runif(n):

		ret = 1
		for k in range(n,0, -1):
			ret = ret * num.unif(0,1)**(1.0/k)
			yield ret
	
	
	# numerically stable computation of Finv(p^(1/exp(logm)) ) for Finv of Laplace distribution
	def qlaplacemax(self, p, logm = 0, mu = 0, b = 1):
		
		if logm == 0:
			return mu+b*self.log(self.val(2.0)*p) if p <= 0.5 else mu-b*self.log(self.val(2.0)-self.val(2.0)*p)
		
		pp_m1 = self.mp.powm1(p, self.mp.exp(-logm))
		
		two = self.mp.mpf(2.0)
		
		if p_pow_m <= 0.5:
			return self.val( mu+b*( self.mp.log(two*p)/self.mp.exp(logm) ) )
		else:
			return self.val( mu-b*self.mp.log(-two*pp_m1) )

	# -log(1 - p**(1/m) )/rate with m = exp(logm)
	def qlogmax(self, p, logm = 0, rate = 1):
	
		if logm == 0:
			return self.qlog(p, rate)
		
		assert (False)
	
	def qhalflogmax(self, p, logm = 0, rate = 1):
	
		if logm == 0:
			return self.qhalflog(p, rate)
		
		assert (False)
	
	# numerically stable computation of - log( 1 - p^(1/exp(logm)) )/rate
	def qexpmax(self, p, logm = 0, rate = 1):
	
		if logm == 0:
			return self.qexp(p, rate)
		
		pp_m1 = self.mp.powm1(p, self.mp.exp(-logm))
		
		return self.val( -self.mp.log(-pp_m1)/rate )
	
	# numerically stable computation of mu - b * log( p^(1/exp(logm)) )
	def qgumbelmax(self, p, logm = 0, mu = 0, b = 1):
		
		if logm == 0:
			return mu-b*self.log(-self.log(p))
		
		return self.val( mu-b*self.log(-self.log(p) )+b * logm )
	
	# efficiently compute binomial coefficients based on existing binomial coefficients
	# e.g. choose(30,10, b = None, given = (29,10, 20030010))
	def choose(self, n,k, b = None, given = None):

		if (n < k) or (k < 0):
			return 0 # choose(n,k) = 0 for k > n or k < 0
	
		if (k == 0) or (n == k):
			return 1 # choose(n,0) = choose(n,n) = choose(k,k) = 1
		
		nn = n; kk = 1; v = n # choose(n,1) = n
		
		if abs( (n-kk)-k) < abs(kk-k):
			kk = n-kk # choose(n,k) = choose(n, n-k)
		
		if given != None:
		
			(nnn, kkk, vv) = given
			
			if nnn > 0 and kkk > 0 and vv > 1:
		
				# move closer to goal
				if abs( (nnn-kkk)-k) < abs(kkk-k):
					kkk = nnn-kkk # choose(n,k) = choose(n, n-k)
		
				# is given closer to goal when choose(n,1) or choose(n,n-1)?
				if abs(nnn-n)+abs(kkk-k) < abs(nn-n)+abs(kk-k):
					nn = nnn
					kk = kkk
					v = vv
		
		if nn > n:
	
			while nn > n and kk > k:
		
				# choose(n-1,k-1) = choose(n, k)*(k)/(n)
				v = v * kk / nn
				nn = nn-1
				kk = kk-1
		
			while nn > n: 
		
				# choose(n-1, k) = choose(n,k)*(n-k)/(n)
				v = v * (nn-kk) / nn
				nn = nn-1
		
		elif nn < n:
		
			while nn < n and kk < k:
		
				# choose(n+1,k+1) = choose(n, k)*(n+1)/(k+1)
				v = v * (nn+1)/(kk+1)
				nn = nn+1
				kk = kk+1
		
			while nn < n:
		
				# choose(n+1, k) = choose(n,k)*(n+1)/(n+1-k)
				v = v * (nn+1)/(nn+1-kk)
				nn = nn+1	
	
		while kk > k:
	
			# choose(n, k-1) = choose(n,k)*(k)/(n+1-k)
			v = v * (kk) / (nn+1-kk)
			kk = kk-1
	
		while kk < k:
		
			# choose(n, k+1) = choose(n,k)*(n-k)/(kk+1)
			v = v * (nn-kk) / (kk+1)
			kk = kk+1
		
		return v
	
	# as choose but in logspace
	def logchoose(self, n,k, given = None):

		if (n < k) or (k < 0):
			return -inf # choose(n,k) = 0 for k > n or k < 0
	
		if (k == 0) or (n == k):
			return 0.0 # choose(n,0) = choose(n,n) = choose(k,k) = 1
		
		nn = n; kk = 1; vv = self.log(n) # choose(n,1) = n
		
		if abs( (n-kk)-k) < abs(kk-k):
			kk = n-kk # choose(n,k) = choose(n, n-k)
		
		if given != None:
		
			(nnn, kkk, vvv) = given
			
			if nnn > 0 and kkk > 0 and vvv > 0:
			
				# move closer to goal
				if abs( (nnn-kkk)-k) < abs(kkk-k):
					kkk = nnn-kkk # choose(n,k) = choose(n, n-k)
				
				# is given closer to goal when choose(n,1) or choose(n,n-1)?
				if abs(nnn-n)+abs(kkk-k) < abs(nn-n)+abs(kk-k):
					nn = nnn
					kk = kkk
					vv = vvv
		
		if nn > n:
	
			while nn > n and kk > k:
		
				# choose(n-1,k-1) = choose(n, k)*(k)/(n)
				vv += self.log(kk)-self.log(nn)
				nn = nn-1
				kk = kk-1
		
			while nn > n: 
		
				# choose(n-1, k) = choose(n,k)*(n-k)/(n)
				vv += self.log(nn-kk)-self.log(nn)
				nn = nn-1
		
		elif nn < n:
		
			while nn < n and kk < k:
		
				# choose(n+1,k+1) = choose(n, k)*(n+1)/(k+1)
				vv += self.log(nn+1)-self.log(kk+1)
				nn = nn+1
				kk = kk+1
		
			while nn < n:
		
				# choose(n+1, k) = choose(n,k)*(n+1)/(n+1-k)
				vv += self.log(nn+1)-self.log(nn+1-kk)
				nn = nn+1	
	
		while kk > k:
	
			# choose(n, k-1) = choose(n,k)*(k)/(n+1-k)
			vv += self.log(kk)-self.log(nn+1-kk)
			kk = kk-1
	
		while kk < k:
		
			# choose(n, k+1) = choose(n,k)*(n-k)/(kk+1)
			vv += self.log(nn-kk)-self.log(kk+1)
			kk = kk+1
		
		return vv
	
# heap for top-k
class Top:

	def __init__(self, k = 1):
	
		self.k = k
		self.h = []
		
		self.minval = None
		self.maxval = None
		self.sumval = None
		
		self._argmin = None
		self._argmax = None
		
		self.count = 0
		
	def __str__(self):
		
		return str( dict(min= self.min(), mean=self.avg(), max = self.max() ) )
	
	def digest(self, val, obj = None):
	
		if self.sumval == None:
			self.sumval = val
		else:
			self.sumval = self.sumval + val
			
		self.count = self.count + 1
		
		if (self.maxval == None) or (val > self.maxval):
			
			self._argmax = obj
			self.maxval = val
			
		if (self.minval == None) or (val < self.minval):
		
			self._argmin = obj
			self.minval = val	
	
		if len(self.h) < self.k:
		
			if self.k == 1:
				self.h = [(val,obj)]
			else:
				heapq.heappush(self.h, (val, obj) )
		else:
			if val > self.h[0][0]:
			
				if self.k == 1:
					self.h[0] = (val,obj)
				else:
					heapq.heapreplace(self.h, (val, obj) )
		
	def argmax(self):
		
		return self._argmax
		
	def argmin(self):
		
		return self._argmin
		
	def avg(self):
		
		return self.sumval / self.count
		
	def sum(self):
		
		return self.sumval
		
	def max(self):
		
		return self.maxval
			
	def min(self):
		
		return self.minval
		
	def count(self):
		
		return self.count
	
	def argmaxs(self):
		
		assert (len(self.h) == self.k)
		return array('l', sorted(x[1] for x in self.h) )
		
	def maxs(self):
		
		return sorted(x[0] for x in self.h)
