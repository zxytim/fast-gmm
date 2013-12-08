/*
 * $File: gmm.cc
 * $Date: Mon Dec 09 00:44:42 2013 +0800
 * $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
 */

#include "gmm.hh"

#include <cassert>
#include <fstream>
#include <limits>

using namespace std;

Gaussian::Gaussian(int dim, int covariance_type) :
	dim(dim), covariance_type(covariance_type) {
	if (covariance_type != COVTYPE_DIAGONAL) {
		const char *msg = "only diagonal matrix supported.";
		printf("%s\n", msg);
		throw msg;
	}
	sigma.resize(dim);
	mean.resize(dim);
}

void Gaussian::sample(std::vector<real_t> &x) {
	switch (covariance_type) {
		case COVTYPE_SPHERICAL:
			throw "COVTYPE_SPHERICAL not implemented";
			break;
		case COVTYPE_DIAGONAL:
			x.resize(dim);
			for (int i = 0; i < dim; i ++)
				x[i] = random.rand_normal(mean[i], sigma[i]);
			break;
		case COVTYPE_FULL:
			throw "COVTYPE_FULL not implemented";
			break;
	}
}

vector<real_t> Gaussian::sample() {
	vector<real_t> x;
	sample(x);
	return x;
}

real_t Gaussian::log_probability_of(std::vector<real_t> &x) {
	assert((int)x.size() == dim);

	real_t sqrt_2_pi = sqrt(2 * M_PI);

	real_t prob = 0;
	switch (covariance_type) {
		case COVTYPE_SPHERICAL:
			throw "COVTYPE_SPHERICAL not implemented";
			break;
		case COVTYPE_DIAGONAL:
			for (int i = 0; i < dim; i ++) {
				real_t &s = sigma[i];
				real_t s2 = s * s;
				prob += -log(sqrt_2_pi * s) - 1.0 / (2 * s2) * (x[i] - mean[i]);
			}
			break;
		case COVTYPE_FULL:
			throw "COVTYPE_FULL not implemented";
			break;
	}
	return prob;
}

void Gaussian::dump(std::ostream &out) {
	out << dim << ' ' << covariance_type << endl;
	for (auto &m: mean) out << m << ' ';
	out << endl;

	// output sigma
	switch (covariance_type) {
		case COVTYPE_SPHERICAL:
			throw "COVTYPE_SPHERICAL not implemented";
			break;
		case COVTYPE_DIAGONAL:
			for (auto &s: sigma) out << s << ' ';
			out << endl;
			break;
		case COVTYPE_FULL:
			for (auto &row: covariance) {
				for (auto &v: row)
					out << v << ' ';
				out << endl;
			}
			break;
	}
}

void Gaussian::load(std::istream &in) {
	in >> dim >> covariance_type;
	for (auto &m: mean) in >> m;

	// input sigma
	switch (covariance_type) {
		case COVTYPE_SPHERICAL:
			throw "COVTYPE_SPHERICAL not implemented";
			break;
		case COVTYPE_DIAGONAL:
			for (auto &s: sigma) in >> s;
			break;
		case COVTYPE_FULL:
			for (auto &row: covariance) {
				for (auto &v: row)
					in >> v;
			}
			break;
	}
}

real_t Gaussian::probability_of(std::vector<real_t> &x) {
	assert((int)x.size() == dim);

	real_t sqrt_2_pi = sqrt(2 * M_PI);

	real_t prob = 1.0;
	switch (covariance_type) {
		case COVTYPE_SPHERICAL:
			throw "COVTYPE_SPHERICAL not implemented";
			break;
		case COVTYPE_DIAGONAL:
			for (int i = 0; i < dim; i ++) {
				real_t &s = sigma[i];
				real_t d = x[i] - mean[i];
				prob *= exp(- d * d / (2 * s * s)) / (sqrt_2_pi * s);
			}
			break;
		case COVTYPE_FULL:
			throw "COVTYPE_FULL not implemented";
			break;
	}
	return prob;
}


GMM::GMM(int nr_mixtures, int covariance_type,
		GMMTrainer *trainer) :
	nr_mixtures(nr_mixtures),
	covariance_type(covariance_type),
	trainer(trainer) {

	if (covariance_type != COVTYPE_DIAGONAL) {
		const char *msg = "only diagonal matrix supported.";
		printf("%s\n", msg);
		throw msg;
	}
}

real_t GMM::log_probability_of(std::vector<real_t> &x, int mixture_id) {
	assert(mixture_id >= 0 && mixture_id < (int)gaussians.size());
	return gaussians[mixture_id]->log_probability_of(x);
}

real_t GMM::log_probability_of(std::vector<real_t> &x) {
	real_t prob = 0;
	for (auto &g: gaussians)
		prob += g->log_probability_of(x);
	return prob;
}

static vector<real_t> random_vector(int dim, real_t range, Random &random) {
	vector<real_t> vec(dim);
	for (auto &v: vec) v = random.rand_real() * range;
	return vec;
}

static void add(const vector<real_t> &a, const vector<real_t> &b, vector<real_t> &c) {
	assert(a.size() == b.size() && b.size() == c.size());
	size_t n = a.size();
	for (size_t i = 0; i < n; i ++)
		c[i] = a[i] + b[i];
}

static void sub(const vector<real_t> &a, const vector<real_t> &b, vector<real_t> &c) {
	assert(a.size() == b.size() && b.size() == c.size());
	size_t n = a.size();
	for (size_t i = 0; i < n; i ++)
		c[i] = a[i] - b[i];
}

static void mult(const vector<real_t> &a, const vector<real_t> &b, vector<real_t> &c) {
	assert(a.size() == b.size() && b.size() == c.size());
	size_t n = a.size();
	for (size_t i = 0; i < n; i ++)
		c[i] = a[i] * b[i];
}

static void mult(const vector<real_t> &a, real_t f, vector<real_t> &b) {
	assert(a.size() == b.size());
	size_t n = a.size();
	for (size_t i = 0; i < n; i ++)
		b[i] = a[i] * f;
}

static void add_self(vector<real_t> &a, const vector<real_t> &b) {
	add(a, b, a);
}

static void sub_self(vector<real_t> &a, const vector<real_t> &b) {
	sub(a, b, a);
}

static void mult_self(vector<real_t> &a, const vector<real_t> &b) {
	mult(a, b, a);
}

static void mult_self(vector<real_t> &a, real_t f) {
	mult(a, f, a);
}

GMMTrainerBaseline::GMMTrainerBaseline(int nr_iter, real_t min_covar) :
	nr_iter(nr_iter), min_covar(min_covar) {
}


void GMMTrainerBaseline::init_gaussians(std::vector<std::vector<real_t>> &X) {
	assert(gmm->covariance_type == COVTYPE_DIAGONAL);

	// calculate data variance
	vector<real_t> initial_sigma(dim);
	vector<real_t> data_mean(dim);
	for (auto &x: X)
		add_self(data_mean, x);
	for (auto &v: data_mean)
		v /= X.size();
	for (auto &x: X) {
		auto v = x;
		sub_self(v, data_mean);
		for (auto &u: v)
			u = u * u;
		add_self(initial_sigma, v);
	}
	mult_self(initial_sigma, 1.0 / (X.size() - 1));
	for (auto &v: initial_sigma)
		v = sqrt(v);

	gmm->gaussians.resize(gmm->nr_mixtures);
	for (auto &g: gmm->gaussians) {
		g = new Gaussian(dim);
		g->mean = X[random.rand_int(X.size())];
		g->sigma = initial_sigma;
	}


	gmm->weights.resize(gmm->nr_mixtures);
	for (auto &w: gmm->weights)
		w = random.rand_real();
}


void GMMTrainerBaseline::clear_gaussians() {
	for (auto &g: gmm->gaussians)
		delete g;
	vector<Gaussian *>().swap(gmm->gaussians);
}

static void gassian_set_zero(Gaussian *gaussian) {
	for (auto &m: gaussian->mean)
		m = 0;
	for (auto &s: gaussian->sigma)
		s = 0;
	for (auto &row: gaussian->covariance)
		for (auto &v: row)
			v = 0;

}

void GMMTrainerBaseline::iteration(std::vector<std::vector<real_t>> &X) {
	size_t n = X.size();

	std::vector<real_t> mixture_density(gmm->nr_mixtures);
	for (int i = 0; i < gmm->nr_mixtures; i ++) {
		real_t &md = mixture_density[i] = 0;
		std::vector<real_t> mu(dim);
		auto &prob = prob_of_y_given_x[i];
		for (size_t j = 0; j < n; j ++) {
			prob[j] = gmm->gaussians[i]->probability_of(X[j]);
			md += prob[j];
		}
		gmm->weights[i] = md / n;
	}

	for (int i = 0; i < gmm->nr_mixtures; i ++) {
		auto &gaussian = gmm->gaussians[i];
		gassian_set_zero(gaussian);
		auto &prob = prob_of_y_given_x[i];
		for (size_t j = 0; j < n; j ++) {
			auto &x = X[j];
			for (int k = 0; k < dim; k ++)
				gaussian->mean[k] += x[k] * prob[j];
		}
		mult_self(gaussian->mean, 1.0 / mixture_density[i]);

		for (size_t j = 0; j < n; j ++) {
			auto &x = X[j];
			for (int k = 0; k < dim; k ++) {
				real_t d = x[k] - gaussian->mean[k];
				gaussian->sigma[k] += d * d * prob[j];
			}
		}
		mult_self(gaussian->sigma, 1.0 / mixture_density[i]);
		for (auto &s: gaussian->sigma)
			s = sqrt(s);
	}
}

void GMMTrainerBaseline::train(GMM *gmm, std::vector<std::vector<real_t>> &X) {
	if (X.size() == 0) {
		const char *msg = "X.size() == 0";
		printf("%s\n", msg);
		throw msg;
	}

	this->gmm = gmm;

	dim = X[0].size();

	prob_of_y_given_x.resize(gmm->nr_mixtures);
	for (auto &v: prob_of_y_given_x)
		v.resize(X.size());

	clear_gaussians();
	init_gaussians(X);

	for (int i = 0; i < nr_iter; i ++) {
		iteration(X);

		// monitor average log likelihood
		real_t ll = 0;
		for (auto &x: X)
			ll += gmm->log_probability_of(x);
		ll /= X.size();
		printf("iter %d: ll %lf\n", i, ll);
	}
}

void GMM::dump(ostream &out) {
	out << nr_mixtures << endl;
	for (auto &w: weights)
		out << w << ' ';
	out << endl;
	for (auto &g: gaussians)
		g->dump(out);
}

void GMM::load(istream &in) {
	in >> nr_mixtures;
	for (auto &w: weights)
		in >> w;
	for (auto &g: gaussians)
		g->load(in);
}

/**
 * vim: syntax=cpp11 foldmethod=marker
 */

