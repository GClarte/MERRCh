// src/smc.cpp
#include "phylo/smc.hpp"
#include "phylo/init.hpp"
#include "phylo/gibbs.hpp"
#include "phylo/tree.hpp"
#include "phylo/transition.hpp"
#include "phylo/pruning.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <chrono>
#include <cstdio>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace phylo {

static void normalize_log(std::vector<double>& p){
  double mx=*std::max_element(p.begin(),p.end());
  for(auto& v:p) v-=mx;
  double s=0; for(double v:p) s+=std::exp(v);
  double lg=std::log(s); for(auto& v:p) v-=lg;
}
static double ess(const std::vector<double>& logw){
  double s2=0; for(double v:logw) s2+=std::exp(2*v);
  return 1.0/s2;
}

static void forward_bruit(State& st,const Data& Dat,const Param& P,const Prior& Pr,int j){
  int nch=P.nch; double bruit=Pr.bruittemp[j];
  std::vector<TransSet> Trposs(nch);
  for(int i=0;i<nch;++i) Trposs[i]=build_trposs(P.passages[i],st.taux[i],P.nph[i]);
  std::vector<std::vector<Mat>> loiapp_(nch),M(nch);
  for(int x=0;x<nch;++x){ loiapp_[x].resize(st.tr.n_edges()); M[x].resize(st.tr.n_edges());
    for(int y=0;y<st.tr.n_edges();++y){
      std::vector<double> lv; for(int c=0;c<st.L.rows();++c) if(st.L(c,y)>0) lv.push_back(st.L(c,y));
      loiapp_[x][y]=loiapp(st.X[x][y],lv,P.loiini[x],Trposs[x],bruit,st.tr.edge_length[y],st.Tps[x][y]);
      M[x][y]=transition_matrix(st.X[x][y],Trposs[x],P.nph[x],bruit,st.tr.edge_length[y]);
    }
  }
  std::vector<std::vector<Mat>> lin(nch);
  for(int x=0;x<nch;++x)
    lin[x]=pruning_full(st.tr,st.X[x],M[x],st.L,Dat[x],P.nph[x],st.tr.root_nodes,P.loiini[x],loiapp_[x]);
  int ncogntot=(int)Dat[0].rows();
  double lnew=lkldcogn(lin,nch,st.tr.root_children,st.tr.root_nodes,ncogntot,P.loiini);
  double lold=lkldcogn(st.Lin,nch,st.tr.root_children,st.tr.root_nodes,ncogntot,P.loiini);
  st.bruit=bruit; st.Lin=lin; st.weight=st.weight+(lnew-lold);
}

static void pasparalbruit(std::vector<State>& state,int pas,const Data& Dat,
                          const Prior& Pr,const Param& P,int ncores,int ncogn,bool fin,
                          Rng& master){
  int npart=(int)state.size();
  std::vector<uint64_t> seeds(npart); for(auto& s:seeds) s=master.raw();
  #ifdef _OPENMP
  #pragma omp parallel for num_threads(ncores) schedule(dynamic)
  #endif
  for(int i=0;i<npart;++i){ Rng rng(seeds[i]);
    state[i]=gibbsbr2(Dat,ncogn,1,pas,Pr,P,state[i],fin,rng); }
  (void)ncores;
}

SMCResult SMCbruit(const Data& Dat,const Param& P,const Prior& Pr,Rng& master,int ncores){
  int npart=P.npart, ncogntot=(int)Dat[0].rows();
  auto t0=std::chrono::steady_clock::now();
  size_t nsteps=Pr.bruittemp.size();

  std::vector<State> state(npart);
  { std::vector<uint64_t> seeds(npart); for(auto& s:seeds) s=master.raw();
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(ncores) schedule(dynamic)
    #endif
    for(int i=0;i<npart;++i){ Rng rng(seeds[i]); state[i]=initialisation1partbruit(Dat,P,Pr,rng); }
  }

  std::vector<std::vector<double>> pdshist;
  std::vector<std::vector<int>> histgeneal;
  std::vector<double> pds(npart); for(int i=0;i<npart;++i) pds[i]=state[i].weight;
  normalize_log(pds); pdshist.push_back(pds);

  auto maybe_resample=[&](std::vector<double>& w){
    if(ess(w)<P.Nmin){
      std::vector<double> pr(npart); for(int i=0;i<npart;++i) pr[i]=std::exp(w[i]);
      auto idx=master.sample(npart,npart,true,&pr);
      histgeneal.push_back(std::vector<int>(idx.begin(),idx.end()));
      std::vector<State> ns(npart);
      for(int i=0;i<npart;++i){ ns[i]=state[idx[i]]; ns[i].weight=-std::log((double)npart); }
      state.swap(ns); std::fill(w.begin(),w.end(),-std::log((double)npart));
    } else {
      std::vector<int> id(npart);
      for(int i=0;i<npart;++i) id[i]=i;
      histgeneal.push_back(std::move(id));
    }
  };

  std::fprintf(stderr,"[SMC] start: %d particles, %zu temperature steps, %d cores\n",
               npart, nsteps, ncores);

  maybe_resample(pds);
  pasparalbruit(state,P.npas,Dat,Pr,P,ncores,ncogntot,false,master);

  for(size_t jstep=1;jstep<nsteps;++jstep){
    for(auto& s:state) forward_bruit(s,Dat,P,Pr,(int)jstep);
    for(int i=0;i<npart;++i) pds[i]=state[i].weight;
    normalize_log(pds); pdshist.push_back(pds);

    double curess=ess(pds);
    bool willresample=(curess<P.Nmin);

    maybe_resample(pds);
    pasparalbruit(state,P.npas,Dat,Pr,P,ncores,ncogntot,false,master);

    double elapsed=std::chrono::duration<double>(
                     std::chrono::steady_clock::now()-t0).count();
    double frac=(double)(jstep+1)/(double)nsteps;
    double eta=(frac>0)? elapsed*(1.0/frac-1.0) : 0.0;
    std::fprintf(stderr,
      "[SMC] step %zu/%zu (%.1f%%)  ESS=%.1f  %s  elapsed=%.1fs  ETA=%.1fs\n",
      jstep+1, nsteps, 100.0*frac, curess,
      willresample?"resampled":"kept    ", elapsed, eta);
    std::fflush(stderr);
  }

  std::fprintf(stderr,"[SMC] final MCMC sweep (npasfin=%d)...\n", P.npasfin);
  pasparalbruit(state,P.npasfin,Dat,Pr,P,ncores,ncogntot,true,master);

  for(auto& s:state) s.tr.root_nodes={s.tr.root_nodes[0]};

  double total=std::chrono::duration<double>(
                 std::chrono::steady_clock::now()-t0).count();
  std::fprintf(stderr,"[SMC] done in %.1fs\n", total);

  return {std::move(state),std::move(pdshist),std::move(histgeneal)};
}




void save_result(const SMCResult& r,const std::string& path){
  std::ofstream f(path);
  f<<"# particle\tweight\trho\tbruit\tla...\ttaux...\tnewick\n";
  for(size_t i=0;i<r.particles.size();++i){ const State& s=r.particles[i];
    f<<i<<"\t"<<s.weight<<"\t"<<s.rho<<"\t"<<s.bruit<<"\t";
    for(double l:s.la) f<<l<<",";
    f<<"\t";
    for(double t:s.taux) f<<t<<",";
    f<<"\t";
    f<<to_newick(s.tr)<<"\n";
  }

  // genealogy: row = resample step, column = particle, value = chosen parent index
  std::ofstream g(path+".geneal");
  g<<"# step\tparents(comma-separated, one entry per particle)\n";
  for(size_t step=0;step<r.histgeneal.size();++step){
    g<<step<<"\t";
    const auto& row=r.histgeneal[step];
    for(size_t i=0;i<row.size();++i){ g<<row[i]; if(i+1<row.size()) g<<","; }
    g<<"\n";
  }

  // weight history: row = SMC step, column = particle, value = normalized log weight
  std::ofstream h(path+".pdshist");
  h<<"# step\tlogweights(comma-separated, one entry per particle)\n";
  for(size_t step=0;step<r.pdshist.size();++step){
    h<<step<<"\t";
    const auto& row=r.pdshist[step];
    for(size_t i=0;i<row.size();++i){ h<<row[i]; if(i+1<row.size()) h<<","; }
    h<<"\n";
  }

  // trees only, as a NEXUS TREES block (multiPhylo)
  std::ofstream n(path+".nex");
  n<<"#NEXUS\n";
  n<<"BEGIN TREES;\n";
  for(size_t i=0;i<r.particles.size();++i){
    std::string nwk=to_newick(r.particles[i].tr);
    if(!nwk.empty() && nwk.back()!=';') nwk.push_back(';');   // ensure trailing ';'
    n<<"    TREE tree_"<<i<<" = [&U] "<<nwk<<"\n";
  }
  n<<"END;\n";
}


} // namespace phylo
