// src/pruning.cpp
#include "phylo/pruning.hpp"
#include "phylo/tree.hpp"
#include <cmath>
namespace phylo {

static Mat nvelleproba(const Eigen::VectorXd& Lbr,const std::vector<int>&,
                       const Vec&,const Mat& Matprob,const Mat& Mtb,const Mat& loiappmat){
  int nph=Matprob.rows(), ncol=Matprob.cols();
  Mat out=Matprob; int app=0;
  for(int c=0;c<ncol;++c){
    if(Lbr(c)==0.0) out.col(c)=Mtb*Matprob.col(c);
    else { double s=loiappmat.col(app).dot(Matprob.col(c)); out.col(c)=Vec::Constant(nph,s); ++app; }
  }
  return out;
}

static void prune_rec(const Tree& tr,const std::vector<std::vector<int>>& X,
   const std::vector<Mat>& M,const Eigen::MatrixXd& L,const Eigen::MatrixXi& Dat,
   int i,int nph,std::vector<Mat>& lin,const Vec& loiini,const std::vector<Mat>& loiapp){
  auto c=tr.children_of(i);
  if(c[0]<0){
    int ncogn=Dat.rows();
    Mat m=Mat::Zero(nph,ncogn);
    for(int jj=0;jj<ncogn;++jj){ int d=Dat(jj,i-1); if(d<0) m.col(jj).setOnes(); else m(d-1,jj)=1.0; }
    lin[i]=m; return;
  }
  int w0=c[0], w1=c[1], u0=tr.edge[w0][1], u1=tr.edge[w1][1];
  prune_rec(tr,X,M,L,Dat,u0,nph,lin,loiini,loiapp);
  prune_rec(tr,X,M,L,Dat,u1,nph,lin,loiini,loiapp);
  Mat M1=nvelleproba(L.col(w0),X[w0],loiini,lin[u0],M[w0],loiapp[w0]);
  Mat M2=nvelleproba(L.col(w1),X[w1],loiini,lin[u1],M[w1],loiapp[w1]);
  lin[i]=M1.cwiseProduct(M2);
}

std::vector<Mat> pruning_full(const Tree& tr,const std::vector<std::vector<int>>& X,
   const std::vector<Mat>& M,const Eigen::MatrixXd& L,const Eigen::MatrixXi& Dat,
   int nph,const std::vector<int>& root,const Vec& loiini,const std::vector<Mat>& loiapp){
  std::vector<Mat> lin(tr.max_node()+1);
  for(int j:root) prune_rec(tr,X,M,L,Dat,j,nph,lin,loiini,loiapp);
  return lin;
}

std::vector<Mat> pruning_eco(const Tree& tr,const std::vector<std::vector<int>>& X,
   const std::vector<Mat>& M,const Eigen::MatrixXi&,int,
   const std::vector<int>& nodes,const std::vector<Mat>& lkldint,
   const Eigen::MatrixXd& L,const Vec& loiini,const std::vector<Mat>& loiapp){
  std::vector<Mat> lin=lkldint;
  for(int j:nodes){
    auto c=tr.children_of(j); if(c[0]<0) continue;
    int w0=c[0],w1=c[1],u0=tr.edge[w0][1],u1=tr.edge[w1][1];
    Mat M1=nvelleproba(L.col(w0),X[w0],loiini,lin[u0],M[w0],loiapp[w0]);
    Mat M2=nvelleproba(L.col(w1),X[w1],loiini,lin[u1],M[w1],loiapp[w1]);
    lin[j]=M1.cwiseProduct(M2);
  }
  return lin;
}

static double sum_log_dot(const Vec& loiini,const Mat& m,int lo,int hi){
  double s=0; for(int c=lo;c<hi;++c) s+=std::log(loiini.dot(m.col(c))); return s;
}

double lkldcogn_channel(const std::vector<Mat>& lin,const std::vector<int>& r1,
   const std::vector<int>& r2,int quelcogn,const Vec& loiini){
  if(lin.empty()) return 0.0;
  int ncol=0; for(auto& m:lin) if(m.cols()>0){ ncol=(int)m.cols(); break; }
  double s=0;
  if(quelcogn==0){ /* R quirk: uses columns 1:0 -> single col; treated as empty here */ }
  else if(quelcogn==ncol){ for(int x:r2) s+=sum_log_dot(loiini,lin[x],0,quelcogn); }
  else { for(int x:r1) s+=sum_log_dot(loiini,lin[x],quelcogn,ncol);
         for(int x:r2) s+=sum_log_dot(loiini,lin[x],0,quelcogn); }
  return s;
}

double lkldcogn(const std::vector<std::vector<Mat>>& lin,int nch,
   const std::vector<int>& r1,const std::vector<int>& r2,int quelcogn,const std::vector<Vec>& loiini){
  double s=0; for(int z=0;z<nch;++z) s+=lkldcogn_channel(lin[z],r1,r2,quelcogn,loiini[z]);
  return s;
}
} // namespace phylo
