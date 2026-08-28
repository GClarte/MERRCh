// src/transition.cpp
#include "phylo/transition.hpp"
#include <cmath>
namespace phylo {

TransSet build_trposs(const std::vector<std::array<int,2>>& passages_ch,double taux,int nph){
  TransSet T; T.reserve(passages_ch.size());
  for(auto& pr:passages_ch){
    Mat A=Mat::Identity(nph,nph);
    A(pr[0],pr[1])=taux; A(pr[0],pr[0])=1.0-taux;
    T.push_back(std::move(A));
  }
  return T;
}

Mat transition_matrix(const std::vector<int>& m,const TransSet& T,int k,double bruit,double length){
  Mat M=Mat::Identity(k,k);
  for(int t:m) M=M*T[t];
  double taux=std::exp(-bruit*length);
  return M*taux + Mat::Constant(k,k,(1.0-taux)/k);
}

Mat loiapp(const std::vector<int>& x,const std::vector<double>& l,const Vec& loiini,
           const TransSet& T,double bruit,double Length,const std::vector<double>& Tps){
  int k=(int)loiini.size();
  if(l.empty()) return Mat(k,0);
  int Q=(int)x.size();
  Mat out(k,(int)l.size());
  if(Q>0){
    std::vector<int> r(l.size()); int rmax=0;
    for(size_t j=0;j<l.size();++j){ int c=0; for(double t:Tps) if(t>l[j]) ++c; r[j]=c; rmax=std::max(rmax,c); }
    Mat M(k,Q+1); M.col(0)=loiini;
    for(int i=1;i<=std::max(rmax,1);++i) M.col(i)=T[x[i-1]].transpose()*M.col(i-1); // rowvec %*% Trposs
    for(size_t j=0;j<l.size();++j){
      double a=std::exp(-bruit*(1.0-l[j])*Length);
      out.col(j)=M.col(r[j])*a + Vec::Constant(k,(1.0-a)/k);
    }
  } else {
    for(size_t j=0;j<l.size();++j){
      double a=std::exp(-bruit*(1.0-l[j])*Length);
      out.col(j)=loiini*a + Vec::Constant(k,(1.0-a)/k);
    }
  }
  return out;
}
} // namespace phylo
