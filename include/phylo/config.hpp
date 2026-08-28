// include/phylo/config.hpp
#pragma once
#include <array>
#include <string>
#include "phylo/types.hpp"
namespace phylo {

struct CliOptions {
  std::string data_csv = "datasets/dataset.csv";
  std::string config   = "config/europe_sl.cfg";
  std::string out      = "EuropeBNZ.txt";
  uint64_t    seed     = 12345;
  int         ncores   = 1;
};
CliOptions parse_cli(int argc, char** argv);

struct RawConfig {
  std::vector<int> char_cols, meanings;
  std::vector<std::string> langues;
  int npart=1500, npas=4000, npasfin=40000, Nmin=750;
  std::vector<double> prob, probfin, bruittemp;
  std::array<double,2> agemax{100.0,0.4}; bool agemax_fixed=false;
  std::array<double,2> prirho{1.0,500.0};
  std::array<double,4> prila{1.0,500.0,0.0,1.0};
  std::array<double,2> pribeta{10.0,1.0};
  std::string treeprior="unif";
  double height_lo=150.0, height_hi=1000.0;
  std::vector<std::vector<int>> tipprior;
  std::vector<std::pair<std::vector<int>,std::array<double,2>>> cladeage;
  int lang_col=4, meaning_col=3;
};

RawConfig load_config(const std::string& path);
// fill Prior/Param once nph is known (from data)
void finalize(const RawConfig&, const std::vector<int>& nph, Param&, Prior&);
}
