#pragma once
#include <cstdint>
#include <random>
#include <vector>
#include "phylo/types.hpp"

namespace phylo {

class Rng {
public:
  explicit Rng(uint64_t seed) : gen_(seed) {}

  double runif(double a = 0, double b = 1);
  double rexp(double rate);
  double rnorm(double m, double s);
  double rgamma(double shape, double rate);
  double rbeta(double a, double b);
  int    rpois(double lambda);
  int    rbinom(int n, double p);
  uint64_t raw() { return gen_(); }

  std::vector<int> sample(int n, int k, bool replace,
                          const std::vector<double>* w = nullptr);

  static double dgamma(double x, double sh, double rt);
  static double dbeta (double x, double a,  double b);
  static double dpois (int k, double lambda);
  static double dbinom(int k, int n, double p);
  static double dnorm (double x, double m, double s);
  static double pgamma(double x, double sh, double rt);
  static double qgamma(double p, double sh, double rt);
  static double pnorm (double x, double m, double s);

private:
  std::mt19937_64 gen_;
};

Vec rdirichlet(Rng& rng, const Vec& alpha);
std::pair<Tree, double> rcoal(Rng& rng, int n, const std::vector<std::string>& labels);

} // namespace phylo
