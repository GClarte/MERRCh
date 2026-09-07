#include "phylo/gibbs.hpp"
#include "phylo/tree.hpp"
#include "phylo/clades.hpp"
#include "phylo/transition.hpp"
#include "phylo/pruning.hpp"
#include "phylo/topology.hpp"
#include <cmath>
#include <numeric>
#include <memory>

namespace phylo {

double lkldtopotout(const Tree& tr,const std::vector<std::vector<std::vector<int>>>& X,
   const std::vector<double>& lambda,const Eigen::MatrixXi& NL,double rho,
   const Eigen::MatrixXd& L,const std::vector<Vec>& P){
  double r=0;
  for(size_t i=0;i<X.size();++i){
    for(size_t b=0;b<X[i].size();++b){
      int S=(int)X[i][b].size();
      r+=Rng::dpois(S,lambda[i]*tr.edge_length[b]);
      for(int x:X[i][b]) r+=std::log(P[i][x]);
    }
  }
  for(int b=0;b<tr.n_edges();++b)
    for(int c=0;c<NL.rows();++c) r+=Rng::dpois(NL(c,b),rho*tr.edge_length[b]);
  for(int b=0;b<L.cols();++b)
    for(int c=0;c<L.rows();++c) if(L(c,b)>0) r+=Rng::dbeta(L(c,b),NL(c,b),1.0);
  return r;
}

double log_prior_tree(const Tree& tr,const Param& P,const Prior& Pr){
  for(auto& tip:Pr.tipprior) if(!is_monophyletic(tr,tip)) return -INFINITY;
  double h=hauteur(tr,tr.root_nodes[0]);
  if(P.priortree==TreePrior::Unif){
    for(auto& cl:P.Cladeage) if(!age_constraint(tr,cl.first,cl.second[0],cl.second[1])) return -INFINITY;
    if(!(h>P.height_lo && h<P.height_hi)) return -INFINITY;
    return -(tr.n_tips()-1)*std::log(h);
  } else if(P.priortree==TreePrior::Yule){
    return -(double)tr.n_tips()*std::log(std::accumulate(tr.edge_length.begin(),tr.edge_length.end(),0.0));
  }
  double Ltot=std::accumulate(tr.edge_length.begin(),tr.edge_length.end(),0.0);
  double partopo=.001;
  return (tr.n_tips()-1)*std::log1p(-std::exp(-partopo*Ltot/2));
}

// ---------- parameter moves ----------
static double gibbslambda(const std::vector<std::vector<int>>& X,const Tree& tr,
                          const std::array<double,4>& pri,Rng& rng){
  int n=0; for(auto& b:X) n+=(int)b.size();
  double l=std::accumulate(tr.edge_length.begin(),tr.edge_length.end(),0.0);
  double sh=pri[0]+n, rt=pri[1]+l, lo=pri[2], hi=pri[3];
  double U=rng.runif();
  double p=Rng::pgamma(lo,sh,rt)+(Rng::pgamma(hi,sh,rt)-Rng::pgamma(lo,sh,rt))*U;
  return Rng::qgamma(p,sh,rt);
}
static Vec gibbsprobas(const std::vector<std::vector<int>>& X,const Vec& hyper,int nbtransf,Rng& rng){
  Vec a=hyper;
  for(auto& b:X) for(int t:b) a[t]+=1.0;
  (void)nbtransf; return rdirichlet(rng,a);
}
static double gibbsrho(double rho,const Eigen::MatrixXi& NL,const Tree& tr,
                       const Eigen::Vector2d& pri,Rng& rng){
  double rhot=rng.rnorm(rho,.00005); if(rhot<0) return rho;
  double l1=Rng::dgamma(rhot,pri[0],pri[1])-Rng::dgamma(rho,pri[0],pri[1]);
  double l2=0;
  for(int b=0;b<NL.cols();++b) for(int c=0;c<NL.rows();++c)
    l2+=Rng::dpois(NL(c,b),tr.edge_length[b]*rhot)-Rng::dpois(NL(c,b),tr.edge_length[b]*rho);
  return (std::log(rng.runif())<l1+l2)?rhot:rho;
}

// ---------- age moves (helpers for gibbstemps) ----------
static int random_internal(const Tree& tr,Rng& rng){
  int nf=tr.n_tips(), root=tr.root_nodes[0];
  std::vector<int> Q; for(int q=nf+1;q<=tr.max_node();++q) if(q!=root||tr.root_nodes.size()>1) Q.push_back(q);
  return Q[rng.sample((int)Q.size(),1,false)[0]];
}
static Tree modif(const Tree& tra,double,Rng& rng){
  Tree tr=tra; int j=random_internal(tr,rng);
  auto A=tr.children_of(j); int B=tr.branch_to(j);
  double u=std::min(tr.edge_length[A[0]],tr.edge_length[A[1]]);
  double v = (B<0)? 0.0 : tr.edge_length[B];
  double w=rng.runif(0,u+v);
  tr.edge_length[A[0]]+=-u+w; tr.edge_length[A[1]]+=-u+w;
  if(B>=0) tr.edge_length[B]+= -v+u+v-w;
  return tr;
}
static std::pair<Tree,double> modifst(const Tree& tra,Rng& rng){
  Tree tr=tra; int j=random_internal(tr,rng);
  auto ec=tr.children_of(j); int ep=tr.branch_to(j);
  double hmin=std::min(hauteur(tr,tr.edge[ec[0]][1]),hauteur(tr,tr.edge[ec[1]][1]));
  double hmax=hauteur(tr,tr.edge[ep][0]);
  double new_h=rng.runif(hmin,hmax);
  auto st=sous_arbre(tr,j); double hj=hauteur(tr,j);
  for(int b:st) tr.edge_length[b]*=new_h/hj;
  tr.edge_length[ep]=hmax-new_h;
  double corr=(st.size()-1)*new_h/hauteur(tr,j); // == st.size()-1 after rescale
  return {tr,corr};
}
static std::pair<Tree,double> modifrootage(const Tree& tra,const Eigen::Vector2d& ag,Rng& rng){
  Tree tr=tra; int root=tr.root_nodes[0]; auto m=tr.children_of(root);
  double rootage=hauteur(tr,root);
  double a0=hauteur(tr,tr.edge[m[0]][1]), a1=hauteur(tr,tr.edge[m[1]][1]);
  double nra=rng.rgamma(ag[0],ag[1]); while(nra<a0||nra<a1) nra=rng.rgamma(ag[0],ag[1]);
  double corr=Rng::dgamma(rootage,ag[0],ag[1])-Rng::dgamma(nra,ag[0],ag[1]);
  tr.edge_length[m[0]]=nra-a0; tr.edge_length[m[1]]=nra-a1;
  for(double l:tr.edge_length) if(l<=0) return {tra,0.0};
  return {tr,corr};
}
static std::pair<Tree,double> modifresc(const Tree& tra,const Eigen::Vector2d& ag,Rng& rng){
  Tree tr=tra; double rootage=hauteur(tr,tr.root_nodes[0]);
  double nra=rng.rgamma(ag[0],ag[1]);
  for(auto& l:tr.edge_length) l*=nra/rootage;
  double corr=(tr.n_edges()-1)*(std::log(nra)-std::log(rootage))
            + Rng::dgamma(rootage,ag[0],ag[1])-Rng::dgamma(nra,ag[0],ag[1]);
  for(double l:tr.edge_length) if(l<=0) return {tra,0.0};
  return {tr,corr};
}

static void gibbstemps(Work& W,const Data& Dat,const Param& P,const Prior& Pr,int ncogn,Rng& rng){
  int a=W.tr.n_edges();
  int qq = P.agemax_fixed ? (2 + rng.sample(2*a-2,1,false)[0]) : rng.sample(2*a,1,false)[0];
  Tree trc; double corr=0;
  if(qq==1){ auto r=modifrootage(W.tr,P.agemax,rng); trc=r.first; corr=r.second; }
  else if(qq==0){ auto r=modifresc(W.tr,P.agemax,rng); trc=r.first; corr=r.second; }
  else if(qq<a+1){ trc=modif(W.tr,P.agemax_value,rng); }
  else { auto r=modifst(W.tr,rng); trc=r.first; corr=r.second; }

  int nch=P.nch;
  std::vector<std::vector<Mat>> loiappc(nch), Mc(nch);
  for(int x=0;x<nch;++x){ loiappc[x].resize(trc.n_edges()); Mc[x].resize(trc.n_edges());
    for(int y=0;y<trc.n_edges();++y){
      std::vector<double> lv; for(int c=0;c<W.L.rows();++c) if(W.L(c,y)>0) lv.push_back(W.L(c,y));
      Mc[x][y]=transition_matrix(W.X[x][y],W.Trposs[x],P.nph[x],W.bruit,trc.edge_length[y]);
      loiappc[x][y]=loiapp(W.X[x][y],lv,P.loiini[x],W.Trposs[x],W.bruit,trc.edge_length[y],W.Tps[x][y]);
    }
  }
  std::vector<std::vector<Mat>> lint(nch);
  for(int x=0;x<nch;++x)
    lint[x]=pruning_full(trc,W.X[x],Mc[x],W.L,Dat[x],P.nph[x],trc.root_nodes,P.loiini[x],loiappc[x]);
  double bet=lkldcogn(lint,nch,trc.root_children,trc.root_nodes,ncogn,P.loiini)
           - lkldcogn(W.lin,nch,W.tr.root_children,W.tr.root_nodes,ncogn,P.loiini);
  double acc=lkldtopotout(trc,W.X,W.la,W.NL,W.rho,W.L,W.P)
           - lkldtopotout(W.tr,W.X,W.la,W.NL,W.rho,W.L,W.P)
           + log_prior_tree(trc,P,Pr)-log_prior_tree(W.tr,P,Pr)+bet+corr;
  if(std::log(rng.runif())<acc){ W.tr=trc; W.lin=lint; W.loiapp=loiappc; W.M=Mc; }
}

// ---------- transformations move (only v=3 / v=4 as in R) ----------
static void gibbstransf(Work& W,int j,const Data& Dat,const Param& P,int ncogn,Rng& rng){
  const Tree& tr=W.tr; int nedges=tr.n_edges();
  std::vector<double> prob(nedges,1.0); if(nedges>=2){ prob[nedges-1]=10; prob[nedges-2]=10; }
  int v=(rng.runif()<5.0/6.0)?3:4;

  auto resample_branch=[&](int w,std::vector<int>& Xc,std::vector<double>& Tc){
    int q=rng.rpois(tr.edge_length[w]*W.la[j]);
    Xc=rng.sample((int)W.Trposs[j].size(),q,true,&*std::make_shared<std::vector<double>>(
        std::vector<double>(W.P[j].data(),W.P[j].data()+W.P[j].size())));
    if(q>0){ std::vector<double> y(q+1); double s=0; for(int t=0;t<=q;++t){ s+=rng.rexp(1); y[t]=s; }
             Tc.resize(q); for(int t=0;t<q;++t) Tc[t]=y[t]/y[q]; }
    else Tc.clear();
  };

  auto lv_for=[&](int w){ std::vector<double> lv; for(int c=0;c<W.L.rows();++c) if(W.L(c,w)>0) lv.push_back(W.L(c,w)); return lv; };

  if(v==3){
    int w=rng.sample(nedges,1,false,&prob)[0];
    int I=tr.edge[w][0];
    auto Xc=W.X[j]; auto Tc=W.Tps[j]; auto Mc=W.M[j]; auto lac=W.loiapp[j];
    resample_branch(w,Xc[w],Tc[w]);
    Mc[w]=transition_matrix(Xc[w],W.Trposs[j],P.nph[j],W.bruit,tr.edge_length[w]);
    lac[w]=loiapp(Xc[w],lv_for(w),P.loiini[j],W.Trposs[j],W.bruit,tr.edge_length[w],Tc[w]);
    auto path=chemin_rac(tr,I,tr.root_nodes);
    auto listc=pruning_eco(tr,Xc,Mc,Dat[j],P.nph[j],path,W.lin[j],W.L,P.loiini[j],lac);
    double l1=lkldcogn_channel(listc,tr.root_children,tr.root_nodes,ncogn,P.loiini[j])
             -lkldcogn_channel(W.lin[j],tr.root_children,tr.root_nodes,ncogn,P.loiini[j]);
    if(std::log(rng.runif())<l1){ W.X[j]=Xc; W.Tps[j]=Tc; W.M[j]=Mc; W.loiapp[j]=lac; W.lin[j]=listc; }
  } else {
    auto Xc=W.X[j]; auto Tc=W.Tps[j]; auto Mc=W.M[j]; auto lac=W.loiapp[j];
    for(int w=0;w<nedges;++w){
      resample_branch(w,Xc[w],Tc[w]);
      Mc[w]=transition_matrix(Xc[w],W.Trposs[j],P.nph[j],W.bruit,tr.edge_length[w]);
      lac[w]=loiapp(Xc[w],lv_for(w),P.loiini[j],W.Trposs[j],W.bruit,tr.edge_length[w],Tc[w]);
    }
    auto listc=pruning_full(tr,Xc,Mc,W.L,Dat[j],P.nph[j],tr.root_nodes,P.loiini[j],lac);
    double l1=lkldcogn_channel(listc,tr.root_children,tr.root_nodes,ncogn,P.loiini[j])
             -lkldcogn_channel(W.lin[j],tr.root_children,tr.root_nodes,ncogn,P.loiini[j]);
    if(std::log(rng.runif())<l1){ W.X[j]=Xc; W.Tps[j]=Tc; W.M[j]=Mc; W.loiapp[j]=lac; W.lin[j]=listc; }
  }
}

// ---------- borrowing move ----------
static void gibbsL(Work& W,const Data& Dat,const Param& P,int ncogn,Rng& rng){
  const Tree& tr=W.tr; int nch=P.nch, ncogntot=(int)Dat[0].rows();
  int i=rng.sample(tr.n_edges(),1,false)[0];
  double tel=tr.edge_length[i];
  Eigen::VectorXi Lte(ncogntot); Eigen::VectorXd Lp(ncogntot);
  for(int c=0;c<ncogntot;++c){ Lte[c]=rng.rpois(W.rho*tel); Lp[c]=(Lte[c]==0)?0.0:rng.rbeta(Lte[c],1.0); }
  auto path=chemin_rac(tr,tr.edge[i][0],tr.root_nodes);
  Eigen::MatrixXd Lt=W.L; for(int c=0;c<ncogntot;++c) Lt(c,i)=Lp[c];
  std::vector<double> lkld(ncogntot,0.0);
  std::vector<std::vector<Mat>> listc(nch);
  std::vector<std::vector<Mat>> loiappc=W.loiapp;
  for(int x=0;x<nch;++x){
    std::vector<double> lv; for(int c=0;c<ncogntot;++c) if(Lt(c,i)>0) lv.push_back(Lt(c,i));
    loiappc[x][i]=loiapp(W.X[x][i],lv,P.loiini[x],W.Trposs[x],W.bruit,tel,W.Tps[x][i]);
    listc[x]=pruning_eco(tr,W.X[x],W.M[x],Dat[x],P.nph[x],path,W.lin[x],Lt,P.loiini[x],loiappc[x]);
    if(ncogn>0) for(int rn:tr.root_nodes) for(int c=0;c<ncogn;++c)
      lkld[c]+=std::log(P.loiini[x].dot(listc[x][rn].col(c)))
              -std::log(P.loiini[x].dot(W.lin[x][rn].col(c)));
  }
  Eigen::MatrixXd Lres=W.L; Eigen::MatrixXi NLt=W.NL;
  for(int c=0;c<ncogntot;++c){
    double thr=(c<ncogn)?lkld[c]:0.0;
    if(std::log(rng.runif())<thr){ Lres(c,i)=Lt(c,i); NLt(c,i)=Lte[c]; }
  }
  auto loiappfin=W.loiapp; std::vector<std::vector<Mat>> listres(nch);
  for(int x=0;x<nch;++x){
    std::vector<double> lv; for(int c=0;c<ncogntot;++c) if(Lres(c,i)>0) lv.push_back(Lres(c,i));
    loiappfin[x][i]=loiapp(W.X[x][i],lv,P.loiini[x],W.Trposs[x],W.bruit,tel,W.Tps[x][i]);
    listres[x]=pruning_eco(tr,W.X[x],W.M[x],Dat[x],P.nph[x],path,W.lin[x],Lres,P.loiini[x],loiappfin[x]);
  }
  W.L=Lres; W.NL=NLt; W.loiapp=loiappfin; W.lin=listres;
}

// ---------- transition-prob (beta) move ----------
static void gibbstaux(Work& W,const Data& Dat,const Param& P,const Prior& Pr,int ncogn,Rng& rng){
  int nch=P.nch;
  std::vector<double> tauxt(nch);
  for(int i=0;i<nch;++i) tauxt[i]=rng.rbeta(Pr.pribeta(0,i),Pr.pribeta(1,i));
  for(int i=0;i<nch;++i){
    auto Tp=build_trposs(P.passages[i],tauxt[i],P.nph[i]);
    std::vector<Mat> lac(W.tr.n_edges()), Mt(W.tr.n_edges());
    for(int y=0;y<W.tr.n_edges();++y){
      std::vector<double> lv; for(int c=0;c<W.L.rows();++c) if(W.L(c,y)>0) lv.push_back(W.L(c,y));
      lac[y]=loiapp(W.X[i][y],lv,P.loiini[i],Tp,W.bruit,W.tr.edge_length[y],W.Tps[i][y]);
      Mt[y]=transition_matrix(W.X[i][y],Tp,P.nph[i],W.bruit,W.tr.edge_length[y]);
    }
    auto lint=pruning_full(W.tr,W.X[i],Mt,W.L,Dat[i],P.nph[i],W.tr.root_nodes,P.loiini[i],lac);
    double bet=lkldcogn_channel(lint,W.tr.root_children,W.tr.root_nodes,ncogn,P.loiini[i])
             -lkldcogn_channel(W.lin[i],W.tr.root_children,W.tr.root_nodes,ncogn,P.loiini[i]);
    if(std::log(rng.runif())<bet){ W.taux[i]=tauxt[i]; W.Trposs[i]=Tp; W.lin[i]=lint; W.loiapp[i]=lac; W.M[i]=Mt; }
  }
}

// ---------- noise move ----------
static void gibbsbruit(Work& W,const Data& Dat,const Param& P,const Prior& Pr,int ncogn,Rng& rng){
  double bruitt=rng.rnorm(W.bruit,Pr.bruit_prop_sd);
  double gam;
  if(bruitt>Pr.bruit_lo && bruitt<Pr.bruit_hi) gam=W.bruit-bruitt; else return;
  int nch=P.nch;
  std::vector<std::vector<Mat>> lac(nch),Mt(nch),lint(nch);
  for(int x=0;x<nch;++x){ lac[x].resize(W.tr.n_edges()); Mt[x].resize(W.tr.n_edges());
    for(int y=0;y<W.tr.n_edges();++y){
      std::vector<double> lv; for(int c=0;c<W.L.rows();++c) if(W.L(c,y)>0) lv.push_back(W.L(c,y));
      lac[x][y]=loiapp(W.X[x][y],lv,P.loiini[x],W.Trposs[x],bruitt,W.tr.edge_length[y],W.Tps[x][y]);
      Mt[x][y]=transition_matrix(W.X[x][y],W.Trposs[x],P.nph[x],bruitt,W.tr.edge_length[y]);
    }
    lint[x]=pruning_full(W.tr,W.X[x],Mt[x],W.L,Dat[x],P.nph[x],W.tr.root_nodes,P.loiini[x],lac[x]);
  }
  double bet=lkldcogn(lint,nch,W.tr.root_children,W.tr.root_nodes,ncogn,P.loiini)
           - lkldcogn(W.lin,nch,W.tr.root_children,W.tr.root_nodes,ncogn,P.loiini);
  if(std::log(rng.runif())<bet+gam){ W.bruit=bruitt; W.lin=lint; W.loiapp=lac; W.M=Mt; }
}

// ---------- driver ----------
State gibbsbr2(const Data& Dat,int ncogn,int m,int mm,const Prior& Pr,const Param& P,
               State st,bool fin,Rng& rng){
  int nch=P.nch;
  Work W; W.tr=st.tr; W.X=st.X; W.Tps=st.Tps; W.L=st.L; W.NL=st.NL;
  W.la=st.la; W.taux=st.taux; W.rho=st.rho; W.bruit=st.bruit; W.P=st.P;
  W.Trposs.resize(nch);
  for(int i=0;i<nch;++i) W.Trposs[i]=build_trposs(P.passages[i],W.taux[i],P.nph[i]);
  W.M.resize(nch); W.loiapp.resize(nch); W.lin.resize(nch);
  for(int i=0;i<nch;++i){ W.M[i].resize(W.tr.n_edges()); W.loiapp[i].resize(W.tr.n_edges());
    for(int y=0;y<W.tr.n_edges();++y){
      std::vector<double> lv; for(int c=0;c<W.L.rows();++c) if(W.L(c,y)>0) lv.push_back(W.L(c,y));
      W.M[i][y]=transition_matrix(W.X[i][y],W.Trposs[i],P.nph[i],W.bruit,W.tr.edge_length[y]);
      W.loiapp[i][y]=loiapp(W.X[i][y],lv,P.loiini[i],W.Trposs[i],W.bruit,W.tr.edge_length[y],W.Tps[i][y]);
    }
    W.lin[i]=pruning_full(W.tr,W.X[i],W.M[i],W.L,Dat[i],P.nph[i],W.tr.root_nodes,P.loiini[i],W.loiapp[i]);
  }
  const std::vector<double>& Prob = fin?P.Probfin:P.Prob;

  for(int it=0;it<m;++it){
    for(int step=0;step<mm;++step){
      int quel=rng.sample(7,1,false,&*std::make_shared<std::vector<double>>(Prob))[0]+1;
      if(quel==1){ for(int j=0;j<nch;++j) gibbstransf(W,j,Dat,P,ncogn,rng); }
      else if(quel==2){
        for(int j=0;j<nch;++j){ W.la[j]=gibbslambda(W.X[j],W.tr,P.Prila[j],rng);
          W.P[j]=gibbsprobas(W.X[j],Pr.hyperpbini[j],(int)W.Trposs[j].size(),rng); }
        W.rho=gibbsrho(W.rho,W.NL,W.tr,Pr.Prirho,rng);
      }
      else if(quel==3) gibbstemps(W,Dat,P,Pr,ncogn,rng);
      else if(quel==4) gibbsL(W,Dat,P,ncogn,rng);
      else if(quel==5) gibbstopopart2(W,Dat,P,Pr,ncogn,fin,rng);
      else if(quel==6) gibbsbruit(W,Dat,P,Pr,ncogn,rng);
      else if(quel==7) gibbstaux(W,Dat,P,Pr,ncogn,rng);
    }
    for(int i=0;i<nch;++i)
      W.lin[i]=pruning_full(W.tr,W.X[i],W.M[i],W.L,Dat[i],P.nph[i],W.tr.root_nodes,P.loiini[i],W.loiapp[i]);
  }

  State out; out.tr=W.tr; out.X=W.X; out.Tps=W.Tps; out.L=W.L; out.NL=W.NL;
  out.la=W.la; out.taux=W.taux; out.rho=W.rho; out.bruit=W.bruit; out.P=W.P;
  out.weight=st.weight; out.Lin=W.lin;
  return out;
}
} // namespace phylo

