#pragma once
#include "phylo/types.hpp"
#include "phylo/rng.hpp"
namespace phylo {

// Working state shared between gibbs moves and topology moves.
struct Work {
  Tree tr;
  std::vector<std::vector<std::vector<int>>>    X;    // [ch][branch]
  std::vector<std::vector<std::vector<double>>> Tps;  // [ch][branch]
  std::vector<std::vector<Mat>> M;                    // [ch][branch]
  std::vector<std::vector<Mat>> loiapp;               // [ch][branch]
  std::vector<std::vector<Mat>> lin;                  // [ch][node]
  Eigen::MatrixXd L; Eigen::MatrixXi NL;
  std::vector<double> la, taux;
  double rho=0, bruit=0;
  std::vector<Vec> P;
  std::vector<TransSet> Trposs;                       // [ch]
};

int gibbstopopart2(Work& W, const Data& Dat, const Param& P, const Prior& Pr,
                   int ncogn, bool fin, Rng& rng);


double lkldtopotout(const Tree&,const std::vector<std::vector<std::vector<int>>>& X,
   const std::vector<double>& lambda,const Eigen::MatrixXi& NL,double rho,
   const Eigen::MatrixXd& L,const std::vector<Vec>& P);

double log_prior_tree(const Tree&,const Param&,const Prior&);

State gibbsbr2(const Data& Dat,int ncogn,int m,int mm,
               const Prior&,const Param&,State statini,bool fin,Rng&);
}
