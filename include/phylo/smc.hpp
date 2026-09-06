// include/phylo/smc.hpp
#pragma once
#include <string>
#include "phylo/types.hpp"
#include "phylo/rng.hpp"
namespace phylo {
struct SMCResult {
  std::vector<State> particles;
  std::vector<std::vector<double>> pdshist;
  std::vector<std::vector<int>> histgeneal;   // <-- add this line
};
SMCResult SMCbruit(const Data&,const Param&,const Prior&,Rng&,int ncores);
void save_result(const SMCResult&, const Param& P, const std::string& path, bool additional_results=false);

}

