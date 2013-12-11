#!/usr/bin/python2
# -*- coding: utf-8 -*-
# $File: test.py
# $Date: Wed Dec 11 13:37:47 2013 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

import pygmm


def read_data(fname):
    with open(fname) as fin:
        return map(lambda line: map(float, line.rstrip().split()), fin)

from sklearn.mixture import GMM as SKGMM


def get_gmm(where):
    if where == 'pygmm':
        return pygmm.GMM(nr_mixture = 3,
                min_covar = 1e-3,
                nr_iteration = 1,
                concurrency = 4)
    elif where == 'sklearn':
        return SKGMM(3, n_iter = 1)
    return None

def random_vector(n, dim):
    import random
    ret = []
    for j in range(n):
        ret.append([random.random() for i in range(dim)])
    return ret

X = read_data('../test.data')
X = random_vector(100, 13)
X.extend(X)
X.extend(X)
X.extend(X)
X.extend(X)
X.extend(X)
X.extend(X)
X.extend(X)
X.extend(X)
X.extend(X)
X.extend(X)
X.extend(X)
X.extend(X)
print(len(X))

gmm = get_gmm('sklearn')
def test():
    gmm.fit(X)
    gmm.score(X)

def timing(code):
    import time
    start = time.time()
    exec(code)
    print(time.time() - start)
timing("test()")

# vim: foldmethod=marker

