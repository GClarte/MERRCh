// include/phylo/transition.hpp
#pragma once
#include "phylo/types.hpp"
namespace phylo {
TransSet build_trposs(const std::vector<std::array<int,2>>& passages_ch,
                      double taux, int nph);
Mat transition_matrix(const std::vector<int>& m, const TransSet& Trposs,
                      int k, double bruit, double length);
Mat loiapp(const std::vector<int>& x, const std::vector<double>& l,
           const Vec& loiini, const TransSet& Trposs, double bruit,
           double length, const std::vector<double>& Tps);
}
