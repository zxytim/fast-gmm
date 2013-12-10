/*
 * $File: pygmm.cc
 * $Date: Wed Dec 11 00:40:11 2013 +0800
 * $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
 */

#include "pygmm.hh"
#include "gmm.hh"

#include <fstream>

using namespace std;

typedef vector<vector<real_t>> DenseDataset;

void conv_double_pp_to_vv(double **Xp, DenseDataset &X, Parameter *param) {
	X.resize(param->nr_instance);
	for (auto &x: X)
		x.resize(param->nr_dim);
	for (int i = 0; i < param->nr_instance; i ++)
		for (int j = 0; j < param->nr_dim; j ++)
			X[i][j] = Xp[i][j];
}

void train_model(const char *model_file_to_write,
		double **X_in, Parameter *param) {
	GMMTrainerBaseline trainer(param->nr_iteration, param->min_covar, param->concurrency);
	GMM gmm(param->nr_mixture, COVTYPE_DIAGONAL, &trainer);
	DenseDataset X;
	conv_double_pp_to_vv(X_in, X, param);
	gmm.fit(X);

	ofstream fout(model_file_to_write);
	gmm.dump(fout);
}

double score(const char *model_file_to_load, double **X_in, Parameter *param) {
	GMM gmm(model_file_to_load);
	DenseDataset X;
	conv_double_pp_to_vv(X_in, X, param);
	double *buffer = new double [gmm.gaussians[0]->fast_gaussian_dim];
	return gmm.log_probability_of_fast_exp(X, buffer);
}



/**
 * vim: syntax=cpp11 foldmethod=marker
 */

