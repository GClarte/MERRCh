// src/init.cpp
#include "phylo/init.hpp"
#include "phylo/tree.hpp"
#include "phylo/clades.hpp"
#include "phylo/transition.hpp"
#include "phylo/pruning.hpp"
#include "phylo/gibbs.hpp"
#include <algorithm>
#include <cmath>
namespace phylo {

std::vector<std::vector<int>> transf(const Tree& tr,const TransSet& T,double la,const Vec& pr,Rng& rng){
  int n=(int)T.size(); std::vector<std::vector<int>> LL(tr.n_edges());
  std::vector<double> w(pr.data(),pr.data()+pr.size());
  for(int i=0;i<tr.n_edges();++i){ int p=rng.rpois(tr.edge_length[i]*la);
    LL[i]=rng.sample(n,p,true,&w); }
  return LL;
}

std::pair<Tree,double> initree(int n,const std::vector<std::string>& labels,
                               const std::vector<std::vector<int>>& tipprior,Rng& rng){
  double corr=0; int cpt=n+2;
  std::vector<int> act; for(int i=1;i<=n;++i) act.push_back(i);
  std::vector<std::array<int,2>> edge; std::vector<double> el;
  std::vector<double> haut(2*n,0.0);
  std::vector<int> ancclades;

  auto erase_all=[&](std::vector<int>& v,const std::vector<int>& rm){
    v.erase(std::remove_if(v.begin(),v.end(),[&](int x){return std::find(rm.begin(),rm.end(),x)!=rm.end();}),v.end());
  };
  for(auto& tp:tipprior) erase_all(act,tp);

  for(size_t i=0;i<tipprior.size();++i){
    std::vector<int> actt=tipprior[i];
    for(size_t k=0;k<i;++k){
      bool all=std::all_of(tipprior[k].begin(),tipprior[k].end(),
        [&](int x){return std::find(tipprior[i].begin(),tipprior[i].end(),x)!=tipprior[i].end();});
      if(all){ erase_all(actt,tipprior[k]); actt.push_back(ancclades[k]);
               act.erase(std::remove(act.begin(),act.end(),ancclades[k]),act.end()); }
    }
    for(int jj=0;jj<(int)actt.size()-1;++jj){
      auto pick=rng.sample((int)actt.size(),2,false); int q0=actt[pick[0]],q1=actt[pick[1]];
      corr-=std::log((double)actt.size());
      erase_all(actt,{q0,q1});
      double x=rng.rexp(1);
      double mx=std::max(haut[q0],haut[q1]);
      edge.push_back({cpt,q0}); el.push_back(x-haut[q0]+mx);
      edge.push_back({cpt,q1}); el.push_back(x-haut[q1]+mx);
      haut[cpt]=mx+x; actt.push_back(cpt); ++cpt; corr+=Rng::dgamma(x,1.0,1.0);
    }
    for(int a:actt) act.push_back(a);
    ancclades.push_back(actt.back());
  }
  while(act.size()>2){
    auto pick=rng.sample((int)act.size(),2,false); int q0=act[pick[0]],q1=act[pick[1]];
    corr-=std::log((double)act.size()); erase_all(act,{q0,q1});
    double x=rng.rexp(1); double mx=std::max(haut[q0],haut[q1]);
    edge.push_back({cpt,q0}); el.push_back(x-haut[q0]+mx);
    edge.push_back({cpt,q1}); el.push_back(x-haut[q1]+mx);
    haut[cpt]=mx+x; act.push_back(cpt); ++cpt; corr+=Rng::dgamma(x,1.0,1.0);
  }
  double x=rng.rexp(1); double mx=std::max(haut[act[0]],haut[act[1]]);
  edge.push_back({n+1,act[0]}); el.push_back(x-haut[act[0]]+mx);
  edge.push_back({n+1,act[1]}); el.push_back(x-haut[act[1]]+mx);

  Tree tr; tr.edge=edge; tr.edge_length=el; tr.tip_label=labels; tr.Nnode=n-1;
  tr.root_nodes={n+1};
  for(auto& e:tr.edge) if(e[0]==n+1) tr.root_children.push_back(e[1]);
  return {tr,corr};
}

// autreini: rescale + sample node heights respecting clade-age constraints
static std::pair<Tree,double> autreini(Tree tr,const Param& P,double corr,Rng& rng){
  const auto& cladage=P.Cladeage;
  if(cladage.empty()){
    double agemax = P.agemax_fixed ? P.agemax_value : rng.rgamma(P.agemax(0),P.agemax(1));
    double h=hauteur(tr,tr.root_nodes[0]); for(auto& l:tr.edge_length) l*=agemax/h;
    return {tr,corr};
  }
  double corrb=corr; int root=tr.root_nodes[0];
  double agemax;
  if(!P.agemax_fixed){
    std::vector<int> nca; for(auto& cl:cladage) nca.push_back(nearest_common_ancestor(tr,cl.first));
    double rootmin=0,rootmax=INFINITY; bool anyroot=false;
    for(size_t z=0;z<nca.size();++z) if(nca[z]==root){ anyroot=true;
      rootmin=std::max(rootmin,cladage[z].second[0]); rootmax=std::min(rootmax,cladage[z].second[1]); }
    if(anyroot){
      double Fmin=Rng::pgamma(rootmin,P.agemax(0),P.agemax(1));
      double Fmax=Rng::pgamma(std::isfinite(rootmax)?rootmax:1e9,P.agemax(0),P.agemax(1));
      agemax=Rng::qgamma(rng.runif(Fmin,Fmax),P.agemax(0),P.agemax(1));
      corrb+=Rng::dgamma(agemax,P.agemax(0),P.agemax(1));
    } else {
      double lomin=INFINITY; for(auto& cl:cladage) lomin=std::min(lomin,cl.second[0]);
      agemax=rng.rgamma(P.agemax(0),P.agemax(1)); while(agemax<lomin) agemax=rng.rgamma(P.agemax(0),P.agemax(1));
      corrb+=Rng::dgamma(agemax,P.agemax(0),P.agemax(1));
    }
  } else agemax=P.agemax_value;

  int N=tr.max_node();
  std::vector<double> h(N+1,std::nan("")); for(int i=1;i<=tr.n_tips();++i) h[i]=0;
  h[root]=agemax;
  auto parcours=parcours_depth(tr,root);

  // clade constraints
  std::vector<int> qn; std::vector<double> qlo,qhi;
  for(auto& cl:cladage){ int a=nearest_common_ancestor(tr,cl.first);
    auto it=std::find(qn.begin(),qn.end(),a);
    double lo=cl.second[0], hi=std::min(cl.second[1],agemax);
    if(it==qn.end()){ qn.push_back(a); qlo.push_back(lo); qhi.push_back(hi); }
    else { size_t k=it-qn.begin(); qlo[k]=std::max(qlo[k],lo); qhi[k]=std::min(qhi[k],hi); } }

  for(int nd:parcours){
    auto it=std::find(qn.begin(),qn.end(),nd); if(it==qn.end()) continue;
    size_t k=it-qn.begin(); if(nd==root) continue;
    double hmin=qlo[k];
    for(int b:sous_arbre(tr,nd)){ int child=tr.edge[b][1];
      if(std::find(qn.begin(),qn.end(),child)!=qn.end() && !std::isnan(h[child])) hmin=std::max(hmin,h[child]); }
    double hmax=qhi[k]; int part=nd;
    while((part=tr.parent_of(part))>=0){ auto jt=std::find(qn.begin(),qn.end(),part);
      if(jt!=qn.end()) hmax=std::min(hmax,qhi[jt-qn.begin()]); }
    if(hmin>hmax) return {tr,std::nan("")};
    h[nd]=rng.runif(hmin,hmax); corrb+=std::log(1.0/(hmax-hmin));
  }

  bool ok=true;
  for(int u:parcours){
    if(!std::isnan(h[u])) continue;
    int part=tr.parent_of(u); int cpt=1;
    while(std::isnan(h[part])){ part=tr.parent_of(part); ++cpt; }
    double hhh=rng.rbeta(1,cpt);
    auto c=tr.children_of(u); double mch=std::max(h[tr.edge[c[0]][1]],h[tr.edge[c[1]][1]]);
    h[u]=mch+(h[part]-mch)*hhh; ok=ok && h[u]>0; corrb+=Rng::dbeta(hhh,1,cpt);
  }
  if(ok) for(int b=0;b<tr.n_edges();++b) tr.edge_length[b]=h[tr.edge[b][0]]-h[tr.edge[b][1]];
  return {tr,corrb};
}

std::pair<Tree,double> iniaveccontraintes(const Param& P,const Prior& Pr,Rng& rng){
  for(;;){
    auto v=initree((int)P.tiplabel.size(),P.tiplabel,Pr.tipprior,rng);
    auto vv=autreini(v.first,P,v.second,rng);
    if(std::isnan(vv.second)) continue;
    bool ok=std::all_of(vv.first.edge_length.begin(),vv.first.edge_length.end(),[](double l){return l>0;});
    for(auto& tp:Pr.tipprior) ok=ok && is_monophyletic(vv.first,tp);
    for(auto& cl:P.Cladeage) ok=ok && age_constraint(vv.first,cl.first,cl.second[0],cl.second[1]);
    if(ok) return vv;
  }
}

State initialisation1partbruit(const Data& Dat,const Param& P,const Prior& Pr,Rng& rng){
  int nfeuilles=(int)Dat[0].cols(), k=(int)Dat[0].rows(), nch=P.nch;
  double bruit=Pr.bruittemp[0];
  auto ZZ=iniaveccontraintes(P,Pr,rng); Tree tr=ZZ.first;

  std::vector<double> taux(nch);
  for(int i=0;i<nch;++i) taux[i]=rng.rbeta(Pr.pribeta(0,i),Pr.pribeta(1,i));
  std::vector<TransSet> Trposs(nch);
  for(int i=0;i<nch;++i) Trposs[i]=build_trposs(P.passages[i],taux[i],P.nph[i]);

  std::vector<Vec> Pb(nch); for(int i=0;i<nch;++i) Pb[i]=rdirichlet(rng,Pr.hyperpbini[i]);
  std::vector<double> la(nch);
  for(int i=0;i<nch;++i){ auto pr=P.Prila[i]; double r=rng.rgamma(pr[0],pr[1]);
    while(r>pr[3]||r<pr[2]) r=rng.rgamma(pr[0],pr[1]);
    la[i]=r; }

  std::vector<std::vector<std::vector<int>>> X(nch);
  for(int i=0;i<nch;++i) X[i]=transf(tr,Trposs[i],la[i],Pb[i],rng);

  double rho=rng.rgamma(Pr.Prirho(0),Pr.Prirho(1)); double corrrho=1.0;

  Eigen::MatrixXi NL(k,tr.n_edges()); Eigen::MatrixXd L(k,tr.n_edges());
  for(int b=0;b<tr.n_edges();++b) for(int c=0;c<k;++c){
    NL(c,b)=rng.rpois(rho*tr.edge_length[b]);
    L(c,b)=(NL(c,b)!=0)?rng.rbeta(NL(c,b),1.0):0.0;
  }
  std::vector<std::vector<std::vector<double>>> Tps(nch);
  for(int i=0;i<nch;++i){ Tps[i].resize(tr.n_edges());
    for(int b=0;b<tr.n_edges();++b){ int Q=(int)X[i][b].size();
      if(Q>0){ std::vector<double> y(Q+1); double s=0; for(int t=0;t<=Q;++t){s+=rng.rexp(1);y[t]=s;}
        Tps[i][b].resize(Q); for(int t=0;t<Q;++t) Tps[i][b][t]=y[t]/y[Q]; } } }

  std::vector<std::vector<Mat>> loiapp_(nch),M(nch),lin(nch);
  for(int i=0;i<nch;++i){ loiapp_[i].resize(tr.n_edges()); M[i].resize(tr.n_edges());
    for(int b=0;b<tr.n_edges();++b){
      std::vector<double> lv; for(int c=0;c<k;++c) if(L(c,b)>0) lv.push_back(L(c,b));
      loiapp_[i][b]=loiapp(X[i][b],lv,P.loiini[i],Trposs[i],bruit,tr.edge_length[b],Tps[i][b]);
      M[i][b]=transition_matrix(X[i][b],Trposs[i],P.nph[i],bruit,tr.edge_length[b]);
    }
    lin[i]=pruning_full(tr,X[i],M[i],L,Dat[i],P.nph[i],tr.root_nodes,P.loiini[i],loiapp_[i]);
  }
  int ncogntot=k;
  double weight=lkldcogn(lin,nch,tr.root_children,tr.root_nodes,ncogntot,P.loiini)
              + std::log((double)nfeuilles*(nfeuilles-1))
              + log_prior_tree(tr,P,Pr) + corrrho + ZZ.second;

  State s; s.tr=tr; s.X=X; s.Tps=Tps; s.L=L; s.NL=NL; s.la=la; s.taux=taux;
  s.rho=rho; s.bruit=bruit; s.P=Pb; s.weight=weight; s.Lin=lin;
  return s;
}
} // namespace phylo
