# This code has been tested with Python 3.9.9

# This file contains a script to generate tikz latex plot files. Some comments in the code may be outdated.

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
from mechanisms import *

TOPK = 0
GREATK = 1
GOODK = 2
BADK = 3

mp.prec = 256 # code was note tested with smaller than 256 bit precision

class LatexFile:

	def __init__(self, path):
	
		self.path = path
		self.stream = StringIO()
		
		self.dir = "/".join(self.path.split("/")[:-1])+"/"
		self.filename = self.path.split("/")[-1]
		self.name = ".".join( self.filename.split(".")[:-1] )
		
	def write(self, s):
		
		self.stream.write(s)
	
	def savelatex(self):
	
		with open(self.path, 'w') as ftab:
		
			ftab.write(strformat(r"""
\documentclass[crop,tikz]{standalone}
%\documentclass{article}

\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{pgfplots}
\usepackage{colortbl}
\definecolor{teal}{RGB}{50, 125, 175}
\definecolor{orange}{RGB}{255, 165, 0}
\newcommand{\eps}{\varepsilon}
			
\newcommand{\NETFLIX}{\ensuremath{\textsc{netflix}}}
\newcommand{\ZIPFS}{p}
\newcommand{\ZIPF}{\ensuremath{ {\textsc{zipfian}}}}

\usepgfplotslibrary{fillbetween}
\usetikzlibrary{patterns}

\newcommand{\CANONICAL}{\textsc{canonical}}
\newcommand{\TAIL}{\textsc{canonical}_{\gamma=1}}
\newcommand{\PEEL}{\textsc{peeling}}
\newcommand{\ONESHOT}{\textsc{oneshot}}

\newcommand{\EXPONENTIAL}{\text{exponential}}
\newcommand{\HALFLOGISTIC}{\text{half-logistic}}
\newcommand{\LOGISTIC}{\text{logistic}}
\newcommand{\LAPLACE}{\text{laplace}}
\newcommand{\GUMBEL}{\text{gumbel}}
			
\newcommand{\TOP}[1]{\ensuremath{\textsc{top}\text{-}#1}}
\newcommand{\GOOD}[1]{\ensuremath{\textsc{good}\text{-}#1}}
\newcommand{\GREAT}[1]{\ensuremath{\textsc{great}\text{-}#1}}

\newcommand{\PrTOP}[1]{Pr[\TOP{k}]}
\newcommand{\PrGOOD}[1]{Pr[\GOOD{k}]}
\newcommand{\PrGREAT}[1]{Pr[\GREAT{k}]}
\pgfplotsset{
compat=1.11,
legend image code/.code={
\draw[mark repeat=2,mark phase=2]
plot coordinates {
(0cm,0cm)
(0.15cm,0cm)        %% default is (0.3cm,0cm)
(0.3cm,0cm)         %% default is (0.6cm,0cm)
};%
}
}
\usepgfplotslibrary{groupplots}
			\begin{document}
			@
			\end{document}
			""",self.stream.getvalue()) )
	
	def save(self):
	
		with open(self.path, 'w') as f:
		
			f.write(self.stream.getvalue())

	def pdflatex(self, open= True):
	
		print("pdflatex "+self.path)
		os.system("pdflatex -output-directory "+self.dir+" "+self.path+" >/dev/null 2>&1")
		
		print("/pdflatex "+self.path)
		
		os.system("rm "+self.dir+self.name+".log")
		os.system("rm "+self.dir+self.name+".aux")
		
		os.system("open "+self.dir+self.name+".pdf")

def utilityclasses(k):
	
	ret = dict()
	ret[TOPK] = k,k
	ret[GREATK] = int( ceil(k*0.1)  ), int(k*1.1)
	ret[GOODK] =  int( ceil(k*0.01) ), int(k*1.5)
	return ret
	
utilityclassnames = dict([ (TOPK,"TOP"), (GREATK,"GREAT"), (GOODK,"GOOD"), (BADK,"BAD") ])	
	
def getutilityclass(k, xs):

	assert (len(xs) == k)
	assert (xs[0] != -inf)
	
	if k == 1:
		if xs[0] == 1:
			return TOPK
		else:
			return BADK
	
	good_h,good_t = utilityclasses(k)[GOODK]
	great_h,great_t = utilityclasses(k)[GREATK]
	
	for j in range(1, k):
		assert (xs[j] > xs[j-1])
	
	utilityclass = TOPK
	
	if xs[k-1] > k:
		utilityclass = GREATK
	
	if xs[k-1] > great_t:
		utilityclass = GOODK
	
	if xs[k-1] > good_t:
		return BADK
	
	if xs[great_h-1] > great_h:
		utilityclass = GOODK
	
	if (good_h > 0) and (xs[good_h-1] > good_h):
		utilityclass = BADK
	
	return utilityclass

def utilityclassCanonicalLipschitz(x, k, eps, Finv):
	
	assert(False)
	
	assert (x[0] == -inf)
	d = len(x)-1
	
	## (k,k)-group = {1,...,k}
	
	good_h,good_t = utilityclasses(k)[GOODK]
	great_h,great_t = utilityclasses(k)[GREATK]
	
	# (t,h)-group = {1,...,t} U subset[k-(h+2)]{h+2,...,t-1} U {t}
	
	top = (k,k) 
	qtop = Finv(num, num.unif(0,1), 0)
	
	maxh = random.randint(0, k-1)
	maxt = random.randint(good_t+1, d)
	
	maxnn = maxt-(maxh+2)
	maxkk = k-(maxh+1)
	
	maxnoise = Finv(num, maxu, num.logchoose(maxnn,maxkk))
	
	for t in between(k+1, d):
	
		mlog = None
		
		for h in between(k-1, 0):
			
			nn = t-(h+2)
			kk = k-(h+1)
			
			if nn < kk:
				continue
				
			if mlog != None:
				if mlog[2] == 0:
					mlog = None
			
			mlog = (nn, kk, num.logchoose(nn,kk, mlog))
			
			qval = ( x[t]-x[h+1] )*eps
			
			if t <= good_t:
			
				q = qval + Finv(num, num.unif(0,1), mlog[2])
				
				if q > qtop:
					top = (h,t)
					qtop = q
			
				if (t == good_t) and (h == 0):
					
					qval2 = (x[maxt]-x[maxh+1])*eps
					if qval2+maxnoise > qtop:
						return BADK
			else:
				
				if (t == maxt) and (h == maxh):
					
					hh,tt = top
					
					if hh == k:
						return TOPK
					
					elif hh >= great_h and tt <= great_t:
						return GREATK
					
					elif hh >= good_h and tt <= good_t:
						return GOODK
					
					else:
						return BADK
				
				if qval+Finv(num, num.unif(0,maxu), mlog[2]) > qtop:
					return BADK
	
	assert (False)
	

def utilityclassProbs(A,k,d):

	good_h,good_t = utilityclasses(k)[GOODK]
	great_h,great_t = utilityclasses(k)[GREATK]

	ret = [None,None,None]
	
	ret[TOPK] = float(A[k-1][k])
	ret[GREATK] = float(A[k-1][k])
	ret[GOODK] = float(A[k-1][k])
	
	for h in between(great_h, k-1): 
		for t in between(k+1,great_t):
			ret[GREATK] += float(A[h][t])
	
	for h in between(good_h, k-1): 
		for t in between(k+1,good_t):
			ret[GOODK] += float(A[h][t])
	
	return ret

def utilityclassProbsTailLipschitzGumbel(x, k, eps, rho=0.5):
	
	assert (x[0] == -inf)
	d = len(x)-1
	
	A = probsCanonicalGamma1Gumbel(x,k,eps, rho)
	
	return utilityclassProbs(A,k,d)
	
def utilityclassProbsCanonicalLipschitzGumbel(x, k, eps, rho=0.5):
	
	assert (x[0] == -inf)
	d = len(x)-1
	
	A = probsCanonicalGumbel(x,k,eps, rho)
	
	return utilityclassProbs(A,k,d)
	
def utilityclassPeeling(x, k, eps,  Finv):

	keps = eps/float(k)
	
	good_h,good_t = utilityclasses(k)[GOODK]
	great_h,great_t = utilityclasses(k)[GREATK]
	
	assert (x[0] == -inf)
	d = len(x)-1 # d-n items that still qualify for good-k
	n = d-good_t # are followed by n items outside of good-k
	
	assert (n > 0)
	
	topk = []
	
	for ii in between(1, k):
		
		itop = -1
		qtop = -inf
		
		# cursor for items selected in prior rounds
		i = 0
		
		for t in between(1, good_t):
		
			# skip items selected in prior rounds
			if i < len(topk) and t == topk[i]:
				
				i += 1
				continue
			
			q = x[t]*keps + Finv(num, num.unif(0,1))
			
			if q > qtop:
			
				itop = t
				qtop = q
		
		assert (itop <= good_t)
		assert(itop != -1)
		
		maxu = float( num.froot(num.unif(0,1), n) )
		maxnoise = Finv(num, maxu ) # inv transform
		maxnoiseitem = random.randint(good_t+1, d) # each item may receive max noise
		
		if x[maxnoiseitem]*keps+maxnoise > qtop:
			return BADK
		
		for t in between(good_t+1, d):
		
			qval = x[t]*keps 
			
			if qval + maxnoise <= qtop:
				break
			
			if ( qval + Finv(num, num.unif(0,maxu)) ) > qtop:
				return BADK
		
		assert (itop <= good_t)
		
		topk.append( itop )
		topk.sort()
	
	return getutilityclass(k, topk)
	

def utilityclassLipschitz(x, k, eps,  Finv):

	assert (x[0] == -inf)
	d = len(x)-1
	
	keps = eps/float(k)
	
	good_h,good_t = utilityclasses(k)[GOODK]
	great_h,great_t = utilityclasses(k)[GREATK]
	
	topk = Top(k)
	
	for t in between(1,good_t):
		
		q = x[t]*keps + Finv(num, num.unif(0,1))
		topk.digest(q, t)
	
	topksel = topk.argmaxs()
	utilityclass = getutilityclass(k, topksel )
	
	n = d-good_t
	
	if n == 0:
		return utilityclass
	
	maxu = float( num.froot(num.unif(0,1), n) )
	maxnoise = Finv(num, maxu )
	topkmin = min( topk.maxs() )
	
	maxnoiseitem = random.randint(good_t+1, d)
	
	if x[maxnoiseitem]*keps + maxnoise > topkmin:
		return BADK
	
	for t in between(good_t+1,d):
		
		qval = x[t]*keps
			
		if qval + maxnoise <= topkmin:
			return utilityclass
			
		if qval + Finv(num, num.unif(0,maxu)) > topkmin:
			return BADK
	
	assert (False)

pow10floor = lambda x,y = 0 : 10 ** ( int( floor(log(x)/log(10)) )-y )
pow10ceil = lambda x,y = 0 : 10 ** ( int( ceil(log(x)/log(10)) )+y )

class DataPoint:

	def getcopy(self, eps):
	
		ret = DataPoint(None,None,None,None,None)
		ret.eps = eps
		
		ret.verbose = self.verbose
		
		ret.probs = self.probs
		ret.style = self.style
		ret.legend = self.legend
		ret.simtime = self.simtime
		ret.runtime = None
		
		ret.zipfs = self.zipfs
		
		ret.origs = self.origs
		ret.mode = self.mode
		
		return ret

	def __init__(self, x, k, eps, s = None, reps = 100, verbose=False):
	
		self.verbose = verbose
	
		if x == None:
			return
			
		assert (s != None)
			
		self.zipfs = -1
		
		assert (x[0] == -inf)
		d = len(x)-1
	
		s = " "+s+" "
		
		self.eps = eps
		self.legend = s
		self.style = ""
		
		finv = None
		finvMax = None
		
		self.origs = s
		
		self.mode = ""
		
		if "$" in s:
			self.legend = "$"+s.split("$")[1]+"$"
			s = s.split("$")[0]+s.split("$")[-1]
			
		assert("$" not in s)
		
		if " fast " not in s:
			self.style += ",only marks,mark size=1pt,solid, mark=*"
			self.mode += "[slow]"
		
		if " gumbel " in s:
			finv = lambda num,p : num.qgumbel(p)
			finvMax = lambda num,p,mlog = 0 : num.qgumbelmax(p,mlog)
			
			self.mode += "[gumbel]"
			
		if " logistic " in s:
			finv = lambda num,p : num.qlog(p)
			finvMax = lambda num,p,mlog = 0 : num.qlogmax(p,mlog)
			self.mode += "[logistic]"
		
		if " halflogistic " in s:
			finv = lambda num,p : num.qhalflog(p)
			finvMax = lambda num,p,mlog = 0 : num.qhalflogmax(p,mlog)
			self.mode += "[halflogistic]"
			
		if " laplace " in s:
			finv = lambda num,p : num.qlaplace(p)
			finvMax = lambda num,p,mlog = 0 : num.qlaplacemax(p,mlog)
			self.mode += "[laplace]"
			
		if " exponential " in s:
			finv = lambda num,p : num.qexp(p)
			finvMax = lambda num,p,mlog = 0 : num.qexpmax(p,mlog)
			self.mode += "[exponential]"
			
		
		rho = 0.5
		
		if " tail " in s:
			rho = 1.0
		
		if " rho" in s:
			for rhoval in [0.0, 0.5, 0.75, 0.9, 0.99, 1.0]:
		
				if (" rho"+str(rhoval)+" ") in s:
					rho = rhoval
		
		self.mode += "[rho="+str(rho)+"]"
		
		if all( (a in s) for a in [" tail ", " gumbel ", " fast "] ):
		
			t0 = time.monotonic()
			self.probs = utilityclassProbsTailLipschitzGumbel(x, k, eps, rho)	
				
			t1 = time.monotonic()
				
			self.simtime = t1-t0
			self.runtime = None
			
			self.mode += "[tailprob]"
		
		elif all( (a in s) for a in [" canonical ", " gumbel ", " fast "] ):
		
			t0 = time.monotonic()
			
			self.probs = utilityclassProbsCanonicalLipschitzGumbel(x, k, eps, rho)
				
			t1 = time.monotonic()
				
			self.simtime = t1-t0
			self.runtime = None
			
			self.mode += "[canonicalprob]"
			
		else:
			
			self.hist = [0, 0, 0, 0]
			
			t0 = time.monotonic()
			
			if " fast " in s:
				
				if " tail " in s:
					self.mode += "[tailfast]"
					
				elif " canonical " in s:
					self.mode += "[canonicalfast]"
					
				elif (" oneshot " in s) or ((" peeling " in s) and (" gumbel " in s)):
					self.mode += "[oneshotfast]"
					
				elif " peeling " in s:
					self.mode += "[peelingfast]"
					
			else:
				
				if " tail " in s:
					self.mode += "[tailslow]"
					
				elif " canonical " in s:
					self.mode += "[canonicalslow]"
					
				elif " oneshot " in s:
					self.mode += "[oneshotslow]"
					
				elif " peeling " in s:
					self.mode += "[peelingslow]"
			
			for seed in range(0, reps):
		
				random.seed(seed+1000)
				
				if " fast " in s:
				
					if " tail " in s:
						self.hist[getutilityclass(k, canonicalGamma1Lipschitz(x, k, eps, finvMax, rho) ) ] += 1
					
					elif " canonical " in s:
						self.hist[utilityclassCanonicalLipschitz(x, k, eps, finvMax) ] += 1
					
					elif (" oneshot " in s) or ((" peeling " in s) and (" gumbel " in s)):
						self.hist[utilityclassLipschitz(x, k, eps,  finvMax ) ] += 1
					
					elif " peeling " in s:
						self.hist[utilityclassPeeling(x, k, eps,  finvMax ) ] += 1
					
				else:
				
					if " tail " in s:
						self.hist[getutilityclass(k, canonicalGamma1Lipschitz(x, k, eps, finvMax, rho) ) ] += 1
					
					elif " canonical " in s:
						self.hist[getutilityclass(k, canonicalLipschitz(x, k, eps, finvMax, rho) ) ] += 1
					
					elif " oneshot " in s:
						self.hist[getutilityclass(k, lipschitz(x, k, eps, finv) ) ] += 1
					
					elif " peeling " in s:
						self.hist[getutilityclass(k, peeling(x, k, eps, finv) ) ] += 1
					
			t1 = time.monotonic()
			
			self.simtime = t1-t0
			
			if " fast " in s:
				self.runtime = None
			else:
				self.runtime = self.simtime/reps
				
			self.probs = [0.0, 0.0, 0.0]
			
			norm = 1.0/sum(self.hist)
			
			self.probs[TOPK] = ( self.hist[TOPK] ) * norm
			self.probs[GREATK] = ( self.hist[TOPK]+self.hist[GREATK] ) * norm
			self.probs[GOODK] = ( self.hist[TOPK]+self.hist[GREATK]+self.hist[GOODK] ) * norm
		
		# color scheme tested with filters for protanopia, deuteranopia and tritanopia
		
		if " blue " in s:
			self.style +=",blue"
			
		if " red " in s:
			self.style +=",red"
			
		if " teal " in s:
			self.style +=",teal"
			
		if " orange " in s:
			self.style +=",orange"
			
		if " violet " in s:
			self.style +=",violet"
			
		if " black " in s:
			self.style +=",black"
			
		if " gray " in s or " grey " in s:
			self.style +=",gray"
			
		if " olive " in s:
			self.style +=",olive"
			
		if " brown " in s:
			self.style +=",brown"
			
		if " magenta " in s:
			self.style +=",magenta"
		
		if self.verbose:
			print("datapoint collected: ", dict(d=d,s=s,logeps=int(log(eps)*100/log(10.0))/100.0,k=k,reps=reps, probs = self.probs, runtime=self.runtime, simtime=self.simtime,mode=self.mode) )
	
	def __repr__(self):
	
		return "pr "+str(self.probs)+" run "+str(self.runtime)+" sim "+str(self.simtime)

		
def getdatapoint(noneps, args,x,k,eps,key,reps):

	pt = None
	
	slowkey = key.replace(" fast ","")
	
	assert ("fast" not in slowkey)

	# noneps => varying Zipfian distribution parameter
	if noneps == 1:
	
		p = eps
		eps = args.e
		xvec = [-inf]+sorted( zipflaw(10000, 15.0**7, p,verbose=(args.v==1) ), key=lambda x: -x)
					
		pt =  DataPoint(xvec, k, eps, key, args.f , verbose=(args.v==1))
		pt.zipfs = p
		pt2 = DataPoint(xvec, k, eps, slowkey, 1, verbose=(args.v==1))
		
		assert (pt2.runtime != None)
		
		pt.runtime = pt2.runtime
	else:
	
		pt = DataPoint(x, k, eps, key, reps,verbose=(args.v==1))
		
		pt2 = DataPoint(x, k, eps, slowkey, 1,verbose=(args.v==1))
		
		assert (pt2.runtime != None)
		pt.runtime = pt2.runtime
	
	assert (pt.runtime != None)	
	
	if args.v == 1:
		print(pt)
	
	return pt
	
# noneps => varying Zipfian distribution parameter
def getdatapoints(noneps, args,x, k, listeps, key, reps):

	ret = [None for eps in listeps]
	
	# all Pr < (1-0.999999)?
	for i in between(len(listeps)-1, len(listeps)-1):
		
		pt = getdatapoint(noneps, args, x, k, listeps[i], key, reps)
		
		if pt.probs[GOODK] <= 1.0-0.999999:
		
			for j in between(0, len(listeps)-1):
				ret[j] = pt.getcopy(listeps[j])
			
			return ret
			
		else:
			ret[i] = pt
			
	# all Pr > 0.999999?
	for i in between(0, 0):
		
		pt = getdatapoint(noneps, args, x, k, listeps[i], key, reps)
		
		if pt.probs[TOPK] >= 0.999999:
		
			for j in between(0, len(listeps)-2):
				ret[j] = pt.getcopy(listeps[j])
			
			return ret
		else:
			ret[i] = pt
			
			
	mid = int(len(listeps)/2)
	
	for i in between(mid-1, 1):
		
		pt = getdatapoint(noneps, args, x, k, listeps[i], key, reps)
		
		if pt.probs[GOODK] <= 1.0-0.999999:
		
			for j in between(i, 1):
				ret[j] = pt.getcopy(listeps[j])
			break	
		else:
			ret[i] = pt
	
	
	for i in between(mid, len(listeps)-2):
		
		pt = getdatapoint(noneps, args, x, k, listeps[i], key, reps)
		
		if pt.probs[TOPK] >= 0.999999:
		
			for j in between(i, len(listeps)-2):
				ret[j] = pt.getcopy(listeps[j])
			
			break
		else:
			ret[i] = pt
	
	return ret

	
def zipflaw(n, startval, S, verbose=False):

	H = sum(1.0/(K**S) for K in between(1, n))
					
	m = startval/H
					
	q = sorted( m/(K**S) for K in between(1, n) )
					
	if verbose:
		print(dict(p=S,bottom=str(int(q[0])), median=str(int(q[int(n/2)])), top=str(int(q[n-100])), su=sum(q)) )
	
	return q

class Interpolator:
		
	def __init__(self, xs, ys):
	
		self.xs = [x for x in xs]
		self.ys = [y for y in ys]
	
	def __call__(self, x):
		
		return np.interp(x, self.xs, self.ys)


def getname(e, args, utilityclass = None):

	if utilityclass == None:
	
		if args.p == 2:
			return e
		
		return e+str(args.a)+"all"+str(args.k)

	return e+str(args.a)+["top","great","good"][utilityclass]+str(args.k)

def run(experiment, args):

	rootdir = args.r
	subdir = args.u

	algs = None
	
	file = dict(m="MEDCOST.n4096",f="ZIPFIAN.n10000",h="HEPTH.n4096",i="INCOME.n4096",p="PATENT.n4096",s="SEARCHLOGS.n4096",n="NETFLIX.n17770", z="")[experiment]
	
	oneshot_exp = r"""$\ONESHOT$ oneshot exponential red""" # oneshot_exp
	canonical_gumbel = r"""$\CANONICAL$ canonical gumbel blue""" # canonical
	tail_gumbel = r"""$\TAIL$ tail gumbel violet""" # nickname for canonical gamma = 1
	
	peeling_gumbel = r"""$\PEEL$ peeling gumbel orange""" # peeling
	oneshot_laplace = r"""$\LAPLACE$ oneshot laplace teal""" # oneshot_laplace
	oneshot_log = r"""$\LOGISTIC$ oneshot logistic gray""" # oneshot_logistic
	oneshot_halflog = r"""$\HALFLOGISTIC$ oneshot halflogistic gray""" # oneshot_halflogistic
	
	if args.a == 1: # just something to test plots
	
		algs = [oneshot_exp]
	
	elif args.a == 2: # just something faster
	
		algs = [canonical_gumbel,oneshot_exp]
	
	elif args.a == 4: # compare approaches from paper
		
		algs = [ canonical_gumbel,tail_gumbel, oneshot_exp, peeling_gumbel]
	
	elif args.a == 5: # compare noise distributions
	
		algs = [r"""$\EXPONENTIAL$ oneshot exponential red""", oneshot_halflog, oneshot_log, r"""$\GUMBEL$ peeling gumbel orange""", oneshot_laplace]	
	else:
		assert (False)
	
	draft = 1
	
	resx = args.x//draft
	
	noneps = 0 # noneps => varying Zipfian distribution parameter
	
	if experiment == "z":
	
		data = "zipf"
		noneps = 1
	
	data = file.replace(".n4096", "")
	
	x = None
	
	if noneps == 0:
		
		x = [-inf]+sorted( [float(a) for a in np.load("data/{}.npy".format(file)).astype(np.float64)], key=lambda x: -x )
		assert (x[0] == -inf)
	
	else:
		
		x = [-inf for x in between(0, 10000)]
		
	d = len(x)-1
	
	cls = [GOODK,GREATK, TOPK]
	cls2 = [TOPK,GOODK]
	
	k = args.k
	
	ks1 = [k]
	
	while True:
	
		k = k//10
		
		if k <= 1:
			break
			
		ks1.append(k)
	
	ks = []
	
	if noneps == 1:
	
		ks = ks1
	
	else:
	
		for k in ks1:
	
			good_t = utilityclasses(k)[GOODK][1]
			mingap = x[k]-x[good_t+1]	
			maxgap = x[1]-x[d]
		
			xmin = pow10floor( log(d)*k/maxgap, 3)
		
			if xmin < 100:
				ks.append(k)
	
	if args.a == 5:
		ks.append(1)
	
	dictptss = dict()
	eps_to_prs = dict()
	pr_to_epss = dict()
	
	xmins = []
	xmaxs = []
	
	globxmin = 0
	globxmax = 1
	
	if noneps == 0:
	
		maxgap = x[1]-x[d]
	
		good_t = utilityclasses(max(ks))[GOODK][1]
		mingap = x[max(ks)]-x[good_t+1]	
	
		globxmin = pow10floor( log(d)*min(ks)/maxgap, 3)
		globxmax = pow10ceil ( log(d)*max(ks)/mingap, 3) if mingap > 0 else xmin*100
	
	runtimes = dict()
	
	for k in ks:
		
		good_h, good_t = utilityclasses(k)[GOODK]
		great_h,great_t = utilityclasses(k)[GREATK]
		
		xmin = globxmin
		xmax = globxmax
		fasteps = np.linspace(xmin, xmax, resx)
		
		if noneps == 0:
			
			maxgap = x[1]-x[d]
			mingap = x[k]-x[good_t+1]
		
			xmin = pow10floor( log(d)*k/maxgap, 3)
			xmax = pow10ceil ( log(d)*k/mingap, 3) if mingap > 0 else xmin*100
	
			fasteps = [globxmin]+[exp(logeps) for logeps in np.linspace(log(xmin), log(xmax), resx)]+[globxmax]
		
		dictpts = dict()
		
		if noneps == 0:
			xmin = max(fasteps)
			xmax = min(fasteps)
	
		sloweps = dict()
	
		eps_to_prs[k] = dict()
		pr_to_epss[k] = dict()
	
		for key in algs:
	
			fastkey = " fast "+key
			
			dictpts[fastkey] = getdatapoints(noneps,args,x,k,fasteps,fastkey,args.f)
				
			
			eps_to_prs[k][fastkey] = [None,None,None]
			pr_to_epss[k][fastkey] = [None,None,None]
		
			listeps = [pt.eps for pt in dictpts[fastkey]]
		
			for utilityclass in cls:
		
				listprs = [pt.probs[utilityclass] for pt in dictpts[fastkey]]
			
				eps_to_prs[k][fastkey][utilityclass] = Interpolator(listeps, listprs)
				pr_to_epss[k][fastkey][utilityclass] = Interpolator(listprs, listeps)
	
		for key in algs:
		
			sloweps[key] = []
	
			for utilityclass in cls:
				for v in [0.001,0.5, 0.999]:
					sloweps[key].append(pr_to_epss[k][" fast "+key][utilityclass](v) )
				
			sloweps[key].sort()
		
		if noneps == 0:
			xmin = min(sloweps[key][0] for key in algs)
			xmax = max(sloweps[key][-1] for key in algs)
		
			if args.v == 1:
				print(xmin, xmax)
		
			xmin = pow10floor(xmin)
			xmax = pow10ceil(xmax)
		
		xmins.append(xmin)
		xmaxs.append(xmax)
		
		dictptss[k] = dictpts
		
	for k in ks:
	
		dictpts = dictptss[k]
	
		if args.s > 0:
			for key in algs:
		
				dictpts[key] = getdatapoints(noneps, args, x,k,sloweps[key],key,args.s)
				
				for pt in dictpts[key]:
					
					pt.style += ",only marks,mark size=1pt,solid,mark=square*"
		
	ind = -1
	
	legends = ["","legend to name={A}","legend to name={B}","legend to name={C}","legend to name={D}","legend to name={E}","legend to name={F}"]
		
	for k in ks:
	
		ind += 1
	
		dictpts = dictptss[k]
		
		ymax = 1.3#1.5
		ymax2 = ymax*0.5+0.5 #0.2
		
		h = 0.25#0.285
		w = 0.45
		
		
		for utilityclass in cls:
		
			if k == 1 and utilityclass != TOPK:
				continue
				
			name = getname(experiment, args,utilityclass)
		
			f = LatexFile(subdir+"{}_D{}{}K{}A{}C{}.tex".format(name,experiment, args.n, k, args.a, utilityclass) ) 
		
			utilityclassname = utilityclassnames[utilityclass]
			
			f.write(strformat(r"""
			\nextgroupplot[ylabel={$\Pr@{@}$},@,legend style={at={(0,1.1)},anchor=south west},legend cell align=left]
			\draw[color=black, thick] (axis cs:@,0.999) -- (axis cs:@,0.999);
			\fill[color=black!5, thick] (axis cs:@,1.0) rectangle (axis cs:@,@);
			""",utilityclassname, k, legends[ind], min(xmins), max(xmaxs), min(xmins), max(xmaxs), ymax) )
			
			xget = lambda pt: pt.eps
			
			if noneps == 1:
				
				xget = lambda pt: pt.zipfs
			
			yget = lambda pt: pt.probs[utilityclass]
			
			for key in algs:
				
				slowkey = key.replace(" fast ","")
				
				pts = []
				
				style = ""
				
				if " thin " not in key:
				
					if utilityclass == GOODK:
						style = r"dotted,line width=1.25pt"
					
					if utilityclass == GREATK:
						style = r"dashed,line width=1.5pt"
					
					if utilityclass == TOPK:
						style = r"solid,line width=2pt"
						
				else:
				
					style = r"solid,line width=1pt"
				
				if key in dictpts:
				
					slowpts = dictpts[key]
				
					if len(slowpts) == 0:
						continue
						
					for pt in slowpts:
					
						f.write( strformat(r"""%%%%%%%%%%%%SLOWSTART:{@}{@}
						""", pt.mode,pt.origs ))
	
						f.write( strformat(r"""\addplot[mark=*,@,@,forget plot] coordinates{
						""",style, pt.style) )
						break
					
					for pt in slowpts:
	
						f.write( strformat(r"""(@,@) % runtime @ simtime @
						""",xget(pt), yget(pt),pt.runtime, pt.simtime ) )
					
					for pt in slowpts:
					
						f.write("};\n")
						
						f.write( strformat(r"""%\addlegendentry{@}
						""", pt.legend ))
						
						f.write( strformat(r"""%%%%%%%%%%%%SLOWEND:{@}{@}
						""", pt.mode,pt.origs ))
						break
				
				pts = dictpts[" fast "+key]
				
				if len(pts) == 0:
					continue
				
				for pt in pts:
				
					f.write( strformat(r"""%%%%%%%%%%%%FASTSTART:{@}{@}
						""", pt.mode,pt.origs ))
					
					
					if " dummy " in key:
						f.write( strformat(r"""\addplot[no marks,@,@,forget plot] coordinates{
				""",style, pt.style) )
				
					else:
					
						f.write( strformat(r"""\addplot[no marks,@,@] coordinates{
				""",style, pt.style) )
					break
				
				
				for pt in pts:
	
					f.write( strformat(r"""(@,@) % runtime @ simtime @
					""",xget(pt), yget(pt),pt.runtime, pt.simtime ) )
				
				
				for pt in pts:
				
					f.write("};\n")
				
					if " dummy " in key:
				
						f.write( strformat(r"""%\addlegendentry{\scriptsize{}@}
					""", pt.legend ))
					else:
						f.write( strformat(r"""\addlegendentry{\scriptsize{}@}
					""", pt.legend ))
					
					f.write( strformat(r"""%%%%%%%%%%%%FASTEND:{@}{@}
						""", pt.mode,pt.origs ))
				
					break
			
			f.write( strformat(r"""
			\draw (axis cs:@,@) node[anchor=west] { \@{@}};
			%\draw (axis cs:@,@) node[anchor=east] {  (@)};
			""",min(xmins), ymax2, utilityclassname, k, max(xmaxs), ymax2, data) ) 
			
			epsvals = dict()
			
			for key in algs:
				epsvals[key] = pr_to_epss[k][" fast "+key][utilityclass](0.999)
				
			sortedalgs = sorted(algs, key = lambda x : epsvals[x])
			
			for key in sortedalgs:
				
				eps1 = epsvals[key]
				
				for otherkey in sortedalgs:
					
					if otherkey == key or ((" canonical " in otherkey) or (" tail " in otherkey)):
						continue
					
					eps2 = epsvals[otherkey]
					
					eps1_ = np.interp(0.01, [0.0, 1.0], [eps1, eps2])
					eps2_ = np.interp(0.99, [0.0, 1.0], [eps1, eps2])
							
					pr1 = eps_to_prs[k][" fast "+key][utilityclass](eps1_)
					pr2 = eps_to_prs[k][" fast "+otherkey][utilityclass](eps1_)
					
					
					pr2_ = max(0.000999, np.interp(0.01, [0.0, 1.0], [pr2, pr1]))
					pr1_ = min(0.999, np.interp(0.99, [0.0, 1.0], [pr2, pr1]) )
					
					if noneps == 0:	
					
						imp1 =  int(eps2_/eps1_)
							
						if imp1 >= 2:
							f.write(strformat(r"""
							\draw[<->, thick,draw=gray] (axis cs:@,@) -- node[ fill=white, midway, draw=gray, inner sep=2,outer sep=2]{\tiny $\times @$} (axis cs:@,@);
							""",eps1_,pr1_, imp1, eps2_, pr1_))
					
					
					imp2 = int(pr1_/pr2_*10)/10.0
							
					if imp2 >= 2 and imp2 < 99:
					
						if imp2 < 99:
							f.write(strformat(r"""
							\draw[<->, thick,draw=gray] (axis cs:@,@) -- node[ fill=white, midway, draw=gray, inner sep=1,outer sep=0]{\tiny $\times @$} (axis cs:@,@);
							""",eps1_,pr1_, imp2, eps1_, pr2_))
						else:
							f.write(strformat(r"""
							\draw[<->, thick,draw=gray] (axis cs:@,@) -- node[ fill=white, midway, draw=gray, inner sep=1,outer sep=0]{\tiny $\times >100$} (axis cs:@,@);
							""",eps1_,pr1_, eps1_, pr2_))
							
					
					break
				
				break
				
			f.save()
	
	if noneps == 0:
		
		for j in [0,1]:
		
			name = getname(experiment, args)
			
			clss = cls2
			
			if j == 1:
			
				if args.p < 2:
					name += "great"
				
				clss = cls
				
			
			ff = LatexFile(rootdir+"{}.tex".format(name) ) 
		
			ff.write(strformat(r"""\scriptsize
	
				\begin{tikzpicture}
				\begin{groupplot}[
				height=@\textwidth,
				width = @\textwidth,
				group style={group size=@ by @,x descriptions at=edge bottom,vertical sep=5pt, horizontal sep=1.5cm}
				,ylabel near ticks, xmode = log, xmin=@,ymin=0.0, ymax=@
				,xmax = @,xtick={0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0,1000.0,10000,100000,1000000,10000000}
				,xticklabels = {$\frac{1}{10^5}$,$\frac{1}{10^4}$,$\frac{1}{10^3}$,$\frac{1}{10^2}$,$\frac{1}{10}$,$1$,$10$,$10^2$,$10^3$,$10^4$,$10^5$,$10^6$,$10^7$}
				,ytick={0.0, 0.5, 0.999}, yticklabels={ $0\%$,$50\%$,$99.9\%$}
				,xlabel={@}, legend pos = south west
				]
				""",h,w,len(clss),len(ks), min(xmins),ymax, max(xmaxs),r"privacy loss parameter $\varepsilon$"))
			
			for k in sorted(ks, key=lambda x : -x):
	
				for utilityclass in clss:
			
					name = getname(experiment, args,utilityclass)
	
					ff.write("\n"+r"""\input{"""+subdir+(r"""{}_D{}{}K{}A{}C{}""".format(name,experiment, args.n, k, args.a,utilityclass) )+"}\n")
			
			ff.write(r"""
				\end{groupplot}
				\end{tikzpicture}
				""")
		
			if args.p > 0:
				ff.savelatex()
			
				if args.l:
					ff.pdflatex()
		
	
	for utilityclass in cls:
	
		name = getname(experiment, args,utilityclass)
	
		f_ks = LatexFile(rootdir+"{}.tex".format(name) ) 
		
		if noneps == 1:
		
			f_ks.write(strformat(r"""\scriptsize
	
			\begin{tikzpicture}
			\begin{groupplot}[
			height=@\textwidth,
			width = @\textwidth,
			group style={group size=1 by @,x descriptions at=edge bottom,vertical sep=5pt}
			,ylabel near ticks, xmin=@,ymin=0.0, ymax=@,
			,xmax = @,xtick={0.0, 0.5, 1.0, 1.5, 2.0}
			,ytick={0.0, 0.5, 0.999}, yticklabels={ $0\%$,$50\%$,$99.9\%$}
			,xlabel={@}, legend pos = south west
			]
			""",h,w,len(ks), min(xmins),ymax, max(xmaxs),r"Zipf's law parameter $s$"))
		
		else:
				
			f_ks.write(strformat(r"""\scriptsize
	
			\begin{tikzpicture}
			\begin{groupplot}[
			height=@\textwidth,
			width = @\textwidth,
			group style={group size=1 by @,x descriptions at=edge bottom,vertical sep=5pt}
			,ylabel near ticks, xmode = log, xmin=@,ymin=0.0, ymax=@
			,xmax = @,xtick={0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0,1000.0}
			,xticklabels = {$\frac{1}{10^5}$,$\frac{1}{10^4}$,$\frac{1}{10^3}$,$\frac{1}{10^2}$,$\frac{1}{10}$,$1$,$10$,$100$,$1000$}
			,ytick={0.0, 0.5, 0.999}, yticklabels={ $0\%$,$50\%$,$99.9\%$}
			,xlabel={@}, legend pos = south west
			]
			""",h,w,len(ks), min(xmins),ymax, max(xmaxs),r"privacy loss parameter $\varepsilon$"))
		
		for k in sorted(ks, key=lambda x : -x):
	
			f_ks.write("\n"+r"""\input{"""+subdir+(r"""{}_D{}{}K{}A{}C{}""".format(name,experiment, args.n, k, args.a,utilityclass) )+"}\n")
			
		f_ks.write(r"""
			\end{groupplot}
			\end{tikzpicture}
			""")
		
		if args.p < 2:
			f_ks.savelatex()
		
			if args.l == 1:
				f_ks.pdflatex()

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='')
	
	# experiment parameters
	
	parser.add_argument("-d", type=str, help="datasets (substring of dhimnpsz where zip[f]ian,[h]epth,[i]ncome,[m]edcost,[n]etflix,[p]atent,[s]earchlogs,[z]ipfian)", default="fhimnpsz")
	parser.add_argument("-k", type=int, help="k in top-k", default=100)
	parser.add_argument("-e", type=float, help="eps (privacy loss parameter for zipfian dataset)", default=1.0)
	parser.add_argument("-a", type=int, help="algs (4 compares the approaches from the paper and 5 the noise distributions)", default=4)
	
	# performance parameters
	
	parser.add_argument("-f", type=int, help="fastres (only reduce for drafts)", default=10000)
	parser.add_argument("-s", type=int, help="slowres (only reduce for drafts)", default=100)
	parser.add_argument("-x", type=int, help="xres (only reduce for drafts) ", default=100)
	
	# file system parameters
	
	parser.add_argument("-r", type=str, help="root dir", default="")
	parser.add_argument("-u", type=str, help="sub plots dir", default="")
	
	parser.add_argument("-l", type=int, help="automatically run pdflatex", default=0)
	
	parser.add_argument("-v", type=int, help="verbose", default=0)
	
	parser.add_argument("-p", type=int, help="which plots to generate (0=none, 1=all, 2=simple)", default=2)
	
	args = parser.parse_args()
	
	args.n = 0 # legacy parameter
	
	# max h+1 value thsat is still top-k
	
	assert (utilityclasses(10)[TOPK][0] == 10)
	assert (utilityclasses(100)[TOPK][0] == 100)
	assert (utilityclasses(1000)[TOPK][0] == 1000)
	
	# max h+1 value that is still great-k
	
	assert (utilityclasses(10)[GREATK][0] == 1)
	assert (utilityclasses(100)[GREATK][0] == 10)
	assert (utilityclasses(1000)[GREATK][0] == 100)
	
	# max h+1 value that is still good-k
	
	assert (utilityclasses(10)[GOODK][0] == 1)
	assert (utilityclasses(100)[GOODK][0] == 1)
	assert (utilityclasses(1000)[GOODK][0] == 10)
	
	# max t value that is still top-k
	
	assert (utilityclasses(10)[TOPK][1] == 10)
	assert (utilityclasses(100)[TOPK][1] == 100)
	assert (utilityclasses(1000)[TOPK][1] == 1000)
	
	# max t value that is still great-k
	
	assert (utilityclasses(10)[GREATK][1] == 11)
	assert (utilityclasses(100)[GREATK][1] == 110)
	assert (utilityclasses(1000)[GREATK][1] == 1100)

	# max t value that is still good-k
	
	assert (utilityclasses(10)[GOODK][1] == 15)
	assert (utilityclasses(100)[GOODK][1] == 150)
	assert (utilityclasses(1000)[GOODK][1] == 1500)
	
	for experiment in (args.d):
	
		run(experiment, args)

