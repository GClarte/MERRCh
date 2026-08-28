// include/phylo/pruning.hpp
#pragma once
#include "phylo/types.hpp"
namespace phylo {
std::vector<Mat> pruning_full(const Tree&,const std::vector<std::vector<int>>& X,
   const std::vector<Mat>& M,const Eigen::MatrixXd& L,const Eigen::MatrixXi& Dat,
   int nph,const std::vector<int>& root,const Vec& loiini,
   const std::vector<Mat>& loiappliste);

std::vector<Mat> pruning_eco(const Tree&,const std::vector<std::vector<int>>& X,
   const std::vector<Mat>& M,const Eigen::MatrixXi& Dat,int nph,
   const std::vector<int>& nodes,const std::vector<Mat>& lkldint,
   const Eigen::MatrixXd& L,const Vec& loiini,const std::vector<Mat>& loiappliste);

// sum over channels & root nodes of sum_{c<ncogn} log(loiini . lin[ch][node].col(c))
double lkldcogn(const std::vector<std::vector<Mat>>& lin,int nch,
   const std::vector<int>& root_children,const std::vector<int>& root_nodes,
   int quelcogn,const std::vector<Vec>& loiini);

double lkldcogn_channel(const std::vector<Mat>& lin_ch,
   const std::vector<int>& root_children,const std::vector<int>& root_nodes,
   int quelcogn,const Vec& loiini);
}
