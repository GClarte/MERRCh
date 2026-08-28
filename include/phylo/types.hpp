#pragma once
#include <Eigen/Dense>
#include <array>
#include <string>
#include <vector>

namespace phylo {

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

struct Tree {
  std::vector<std::array<int, 2>> edge;   // edge[b] = {parent, child}, node ids 1-based
  std::vector<double> edge_length;
  std::vector<std::string> tip_label;
  int Nnode = 0;
  std::vector<int> root_children;         // R: tr$root[[1]]
  std::vector<int> root_nodes;            // R: tr$root[[2]]

  int n_tips()   const { return static_cast<int>(tip_label.size()); }
  int n_edges()  const { return static_cast<int>(edge.size()); }
  int max_node() const;
  int branch_to(int node) const;                 // branch whose child==node, else -1
  std::array<int, 2> children_of(int node) const;// {-1,-1} if leaf
  int parent_of(int node) const;                 // parent node, else -1
};

using TransSet = std::vector<Mat>;                // one nph x nph matrix per transform

struct State {
  Tree tr;
  std::vector<std::vector<std::vector<int>>>    X;   // [ch][branch] -> transforms (0-based)
  std::vector<std::vector<std::vector<double>>> Tps; // [ch][branch] -> times in (0,1)
  Eigen::MatrixXd L;   // ncogn x nedges
  Eigen::MatrixXi NL;  // ncogn x nedges
  std::vector<double> la;
  std::vector<double> taux;
  double rho = 0.0;
  double bruit = 0.0;
  std::vector<Vec> P;                 // [ch] transform probabilities
  double weight = 0.0;
  std::vector<std::vector<Mat>> Lin;  // [ch][node] pruning matrices (nph x ncogn)
};

enum class TreePrior { Unif, Yule, Coal };

struct Prior {
  Eigen::Vector2d Prirho{1.0, 500.0};    // gamma(shape,rate)
  std::vector<Vec> hyperpbini;           // dirichlet hyperparams per channel
  Eigen::MatrixXd pribeta;               // 2 x nch (beta hyperparams)
  std::vector<double> bruittemp;         // annealing noise schedule
  std::vector<std::vector<int>> tipprior;// monophyly constraints (1-based tips)

  // noise RW proposal + support (pribruit)
  double bruit_prop_sd = 0.001;
  double bruit_lo = 1e-5;
  double bruit_hi = 1e-1;
};

struct Param {
  std::vector<std::vector<std::array<int, 2>>> passages; // [ch] list of (from0,to0)
  int nch = 0;
  std::vector<int> nph;
  int npart = 0, npas = 0, npasfin = 0, Nmin = 0;
  std::vector<Vec> loiini;
  std::vector<double> Prob, Probfin;     // length 7
  Eigen::Vector2d agemax{100.0, 0.4};    // gamma(shape,rate) when variable
  bool   agemax_fixed = false;
  double agemax_value = 0.0;
  std::vector<std::string> tiplabel;
  std::vector<std::vector<int>> ReconstructClade;
  std::vector<std::pair<std::vector<int>, std::array<double, 2>>> Cladeage;
  TreePrior priortree = TreePrior::Unif;
  std::vector<std::array<double, 4>> Prila; // (shape,rate,lo,hi) per channel
  double height_lo = 150.0, height_hi = 1000.0;
};

using Data = std::vector<Eigen::MatrixXi>; // one ncogn x nlangues matrix per channel, -1 == NA

} // namespace phylo
