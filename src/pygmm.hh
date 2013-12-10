/*
 * $File: pygmm.hh
 * $Date: Wed Dec 11 00:40:21 2013 +0800
 * $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
 */

#pragma once

extern "C" {
struct Parameter {
	int nr_instance;
	int nr_dim;

	int nr_mixture;

	double min_covar;
	int nr_iteration;
	int concurrency;
};

void train_model(const char *model_file_to_write, double **X, Parameter *param);

// only nr_instances and nr_dim are used
double score(const char *model_file_to_load, double **X, Parameter *param);

}

/**
 * vim: syntax=cpp11 foldmethod=marker
 */

