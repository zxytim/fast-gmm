#!/usr/bin/python2
# -*- coding: utf-8 -*-
# $File: test.py
# $Date: Wed Dec 11 00:31:53 2013 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

import pygmm

gmm = pygmm.GMM("gmm.model",
        nr_mixture = 3,
        min_covar = 1e-3,
        nr_iteration = 200,
        concurrency = 4)

def read_data(fname):
    with open(fname) as fin:
        return map(lambda line: map(float, line.rstrip().split()), fin)

X = read_data('../test.data')
gmm.fit(X)
print gmm.score(X)

# vim: foldmethod=marker

