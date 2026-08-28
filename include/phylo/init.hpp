// include/phylo/init.hpp
#pragma once
#include "phylo/types.hpp"
#include "phylo/rng.hpp"
namespace phylo {
std::vector<std::vector<int>> transf(const Tree&,const TransSet&,double la,const Vec& pr,Rng&);
std::pair<Tree,double> initree(int n,const std::vector<std::string>& labels,
                               const std::vector<std::vector<int>>& tipprior,Rng&);
std::pair<Tree,double> iniaveccontraintes(const Param&,const Prior&,Rng&);
State initialisation1partbruit(const Data&,const Param&,const Prior&,Rng&);
}
