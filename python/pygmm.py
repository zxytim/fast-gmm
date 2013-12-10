#!/usr/bin/python2
# -*- coding: utf-8 -*-
# $File: pygmm.py
# $Date: Wed Dec 11 00:44:02 2013 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

from ctypes import *
import os
from os import path

dirname = path.dirname(path.abspath(__file__))

pygmm = cdll.LoadLibrary(path.join(dirname, '../lib/pygmm.so'))
print(pygmm)
print(pygmm.__dict__)

class GMMParameter(Structure):
    _fields = [("nr_instance", c_int),
            ("nr_dim", c_int),
            ("nr_mixture", c_int),
            ("min_covar", c_double),
            ("nr_iteration", c_int),
            ("concurrency", c_int)]

class GMM(object):
    def __init__(self, model_file,
            nr_mixture = 10,
            min_covar = 1e-3,
            nr_iteration = 200,
            concurrency = 1):
        self.model_file = model_file
        self.param = GMMParameter()
        self.param.nr_mixture = c_int(nr_mixture)
        self.param.min_covar = c_double(min_covar)
        self.param.nr_iteration = c_int(nr_iteration)
        self.param.concurrency = c_int(concurrency)

    def _double_array_python_to_ctype(self, X_py):
        X_c = []
        for x in X_py:
            xs = (c_double * len(x))(*x)
            X_c.append(xs)
        X_c = (POINTER(c_double) * len(X_c))(*X_c)
        return X_c

    def _gen_param(self, X):
        self.param.nr_instance = c_int(len(X))
        self.param.nr_dim = c_int(len(X[0]))
        return self.param

    def fit(self, X):
        model_file = c_char_p(self.model_file)
        X_c = self._double_array_python_to_ctype(X)
        param = self._gen_param(X)
        pygmm.train_model(model_file, X_c, param)

    def score(self, X):
        return pygmm.score(c_char_p(self.model_file), \
                self._double_array_python_to_ctype(X), \
                self._gen_param(X))

# vim: foldmethod=marker

