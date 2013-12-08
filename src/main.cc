/*
 * $File: main.cc
 * $Date: Mon Dec 09 00:42:17 2013 +0800
 * $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
 */

#include <cstdio>
#include <fstream>

#include "gmm.hh"

#include "datamanip.hh"
#include "common.hh"

#include "tclap/CmdLine.h"

using namespace std;
using namespace TCLAP;

typedef std::vector<std::vector<real_t>> DenseDataset;

static void svm2vec(const std::vector<std::pair<int, real_t>> &input,
		std::vector<real_t> &output, int dim) {
	output = std::vector<real_t>(dim, 0);
	for (auto &item: input) {
		if (item.first > dim) // ignore out bounded values
			continue;
		output[item.first] = item.second;
	}
}

vector<real_t> string_to_double_vector(string line) {
	vector<real_t> x;
	int begin = 0, end = 0;
	int len = line.size();
	while (true) {
		while (end < len && line[end] != ' ' && line[end] != '\n')
			end ++;
		x.push_back(atof(line.substr(begin, end - begin).c_str()));
		if (end == len || line[end] == '\n')
			break;
		begin = end + 1;
		end = begin;
	}
	return x;
}

void Dataset2DenseDataset(Dataset &X0, DenseDataset &X1) {
	int n, m;
	get_data_metric(X0, n, m);

	X1.resize(X0.size());
	for (size_t i = 0; i < X0.size(); i ++)
		svm2vec(X0[i], X1[i], m);
}



struct Args {
	int concurrency;
	int K;

	string input_file;
	string output_file;
};

Args parse_args(int argc, char *argv[]) {
	Args args;
	try {
		CmdLine cmd("Gaussian Mixture Model (GMM)", ' ', "0.0.1");

		ValueArg<int> arg_concurrency("w", "concurrency", "number of workers", false, 1, "NUMBER");
		ValueArg<int> arg_K("k", "K", "number of gaussians", true, 10, "NUMBER");

		ValueArg<string> arg_input_file("i", "input", "intput file", true, "", "FILE");
		ValueArg<string> arg_output_file("o", "output", "intput file", true, "", "FILE");

		cmd.add(arg_concurrency);
		cmd.add(arg_K);
		cmd.add(arg_input_file);
		cmd.add(arg_output_file);

		cmd.parse(argc, argv);

#define GET_VALUE(name) args.name = arg_##name.getValue();
		GET_VALUE(concurrency);
		GET_VALUE(K);
		GET_VALUE(input_file);
		GET_VALUE(output_file);

	} catch (ArgException &e) {
		cerr << "error: " << e.error() << " for arg " << e.argId() << endl;
	}
	return args;
}

void read_dense_dataset(DenseDataset &X, const char *fname) {
	ifstream fin(fname);
	string line;
	while (getline(fin, line)) {
		X.push_back(string_to_double_vector(line));
	}
}

void write_dense_dataset(DenseDataset &X, const char *fname) {
	ofstream fout(fname);
	for (auto &x: X) {
		for (auto &v: x)
			fout << v << ' ';
		fout << endl;
	}
}

void fill_gaussian_2d(DenseDataset &X, Gaussian *gaussian, int nr_point) {
	for (int i = 0; i < nr_point; i ++)
		X.push_back(gaussian->sample());
}

int main(int argc, char *argv[]) {
//    srand(42); // Answer to The Ultimate Question of Life, the Universe, and Everything
//    Args args = parse_args(argc, argv);

	DenseDataset X;
//    read_dense_dataset(X, args.input_file.c_str());

	int nr_gaussian = 2;
	int nr_point_per_gaussian = 100;
	Gaussian g0(2);
	g0.mean = {0, 0};
	g0.sigma = {0.1, 0.1};

	Gaussian g1(2);
	g1.mean = {1, 0};
	g1.sigma = {0.1, 0.1};

	fill_gaussian_2d(X, &g0, nr_point_per_gaussian);
	fill_gaussian_2d(X, &g1, nr_point_per_gaussian);

	write_dense_dataset(X, "test.data");

	int nr_mixture = 2;
	GMM gmm(nr_mixture);
	gmm.fit(X);

	ofstream fout("gmm-test.model");
	gmm.dump(fout);

	return 0;
}

/**
 * vim: syntax=cpp11 foldmethod=marker
 */

