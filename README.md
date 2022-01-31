# Differentially private top-k selection

Top-k selection simply means to pick the k items with the largest scores. If the scores are based on user data then it becomes tricky to preserve the privacy of the users and select something akin to the top-k at the same time. This is solved here with the differential privacy framework. The top-k selection is randomised in a way that adding/removing a user from the data changes the probability of any candidate output by a controllable parameter (typically given the greek letter epsilon). Thus, two datasets differing in a user behave very similar.y. The challenge is then to maximise utility in terms of outputting good outputs with high probability while being held back and limited by this controllable parameter epsilon (eps).

## Requirements (needed for numerical precision and convenience):

- Python3.9.9 (see [python.org](https://python.org)) older versions should work fine but have not been tested
- mpmath (see [mpmath.org](https://mpmath.org))
- numpy (see [numpy.org](https://numpy.org))

## Creating Plots

How to generate all plots in one directoy:
``
python3.9 eval.py -v 1
''
How to generate the subplots instead in a subdirectory:
``
mkdir sub
python3.9 eval.py -u sub/ -v 1
''
To obtain no console messages remove "-v 1".

## Files

- mechanisms.py (eps-DP selection and probability computations)
- lib.py (random variable arithmetics and more)
- eval.py (script to generate plots using mechanisms.py and some monte carlo sims for probability estimation)

- data/ZIPFIAN.n10000.npy (dataset f)
- data/HEPTH.n4096.npy (dataset h)
- data/INCOME.n4096.npy (dataset i)
- data/MEDCOST.npy (dataset m)
- data/NETFLIX.n17770.npy (dataset n)
- data/PATENT.n4096.npy (dataset p)
- data/SEARCHLOGS.n4096.npy (dataset s)

After running the script one should have in the same directory as the script:

- f.tex (dataset f)
- h.tex (dataset h)
- i.tex (dataset f)
- m.tex (dataset m)
- n.tex (dataset n)
- p.tex (dataset p)
- s.tex (dataset s)
- z.tex (dataset z)

Note that a full Latex distribution is needed to obtain PDFs for the latex plots.

## Misc

A list of other possible commands can be obtained with:

python3.9 eval.py -h
