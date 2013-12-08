/*
 * $File: gmm.hh
 * $Date: Mon Dec 09 00:45:03 2013 +0800
 * $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
 */

#pragma once

#include "type.hh"
#include "random.hh"

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>


enum CovType {
	COVTYPE_SPHERICAL,
	COVTYPE_DIAGONAL,
	COVTYPE_FULL
};

class Gaussian {
	public:
		Gaussian(int dim, int covariance_type = COVTYPE_DIAGONAL);

		int dim, covariance_type;
		std::vector<real_t> mean;
		std::vector<real_t> sigma;
		std::vector<std::vector<real_t>> covariance; // not used

		real_t log_probability_of(std::vector<real_t> &x);
		real_t probability_of(std::vector<real_t> &x);

		// sample a point to @x
		void sample(std::vector<real_t> &x);
		std::vector<real_t> sample();

		void dump(std::ostream &out);
		void load(std::istream &in);

		Random random;
};

class GMM;
class GMMTrainer {
	public:
		virtual void train(GMM *gmm, std::vector<std::vector<real_t>> &X) = 0;
		virtual ~GMMTrainer() {}
};

class GMMTrainerBaseline : public GMMTrainer {
	public:
		GMMTrainerBaseline(int nr_iter = 10, real_t min_covar = 1e-3);
		virtual void train(GMM *gmm, std::vector<std::vector<real_t>> &X);
		void clear_gaussians();

		// init gaussians along with its weight
		virtual void init_gaussians(std::vector<std::vector<real_t>> &X);

		virtual void iteration(std::vector<std::vector<real_t>> &X);

		int dim;

		Random random;
		GMM *gmm;

		int nr_iter;
		real_t min_covar;

		std::vector<std::vector<real_t>> prob_of_y_given_x; // y, x
};

class GMM {
	public:
		GMM(int nr_mixtures, int covariance_type = COVTYPE_DIAGONAL,
				GMMTrainer *trainer = NULL);

		template<class Instance>
			void fit(std::vector<Instance> &X) {
				bool new_trainer = false;
				if (trainer == NULL) {
					trainer = new GMMTrainerBaseline();
					new_trainer = true;
				}
				trainer->train(this, X);
				if (new_trainer)
					delete trainer;
			}

		void dump(std::ostream &out);
		void load(std::istream &in);

		int nr_mixtures;
		int covariance_type;

		int dim;
		GMMTrainer *trainer;

		std::vector<real_t> weights;
		std::vector<Gaussian *> gaussians;

		real_t log_probability_of(std::vector<real_t> &x, int mixture_id);
		real_t log_probability_of(std::vector<real_t> &x);
};

/**
 * vim: syntax=cpp11 foldmethod=marker
 */

