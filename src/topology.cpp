// src/topology.cpp
#include "phylo/gibbs.hpp"
#include "phylo/topology.hpp"
#include "phylo/tree.hpp"
#include "phylo/transition.hpp"
#include "phylo/pruning.hpp"
#include <algorithm>
#include <cmath>
#include <set> 

namespace phylo {
namespace {

using VVI = std::vector<std::vector<std::vector<int>>>;
using VVD = std::vector<std::vector<std::vector<double>>>;
using VVM = std::vector<std::vector<Mat>>;

// Result of a single SPR/outgroup proposal, mirroring the R list returned by
// modificationtopo*.  d==-1 => rejected proposal; d==0 => outgroup move
// (full recompute); d>0 => SPR (eco pruning along path to root).
struct MoveResult {
  Tree trt;
  VVI  Xt;
  VVD  Tpst;
  Eigen::MatrixXi NLt;
  Eigen::MatrixXd Lt;
  VVM  Mt, loiappt;
  double corrb = 0.0;
  int    d = -1, f = -1;
  std::vector<double> quoisur;
  std::vector<double> la;
  double rho = 0.0, bruit = 0.0;
};

// -------- small helpers --------
inline bool contains(const std::vector<int>& v, int x){
  return std::find(v.begin(), v.end(), x) != v.end();
}
inline void erase_val(std::vector<int>& v, int x){
  v.erase(std::remove(v.begin(), v.end(), x), v.end());
}
inline int count_intersect(const std::vector<int>& a, const std::vector<int>& b){
  int c = 0; for(int x : a) if(contains(b, x)) ++c; return c;
}
inline std::vector<double> sorted_unif(Rng& rng, int n){
  std::vector<double> v(n); for(auto& x : v) x = rng.runif();
  std::sort(v.begin(), v.end()); return v;
}
inline std::vector<int> sampleX(Rng& rng, const Vec& P, int q){
  std::vector<double> w(P.data(), P.data() + P.size());
  return rng.sample(static_cast<int>(P.size()), q, true, &w);
}
inline double sumlogP(const Vec& P, const std::vector<int>& x){
  double s = 0; for(int t : x) s += std::log(P[t]); return s;
}
inline std::vector<double> Lcol_pos(const Eigen::MatrixXd& L, int col){
  std::vector<double> v;
  for(int c = 0; c < L.rows(); ++c) if(L(c, col) > 0) v.push_back(L(c, col));
  return v;
}

MoveResult reject_move(const Work& W){
  MoveResult r;
  r.trt = W.tr; r.Xt = W.X; r.Tpst = W.Tps; r.NLt = W.NL; r.Lt = W.L;
  r.Mt = W.M; r.loiappt = W.loiapp;
  r.corrb = 0.0; r.d = -1; r.f = -1;
  r.quoisur.assign(W.X.size(), -1.0);
  r.la = W.la; r.rho = W.rho; r.bruit = W.bruit;
  return r;
}

// =====================================================================
//  Shared core for modificationtopo2 / modificationtopo3
//    resample_i        : topo2 resamples latent vars on branch i (above e)
//    reattach_root_a   : topo2 reattaches (d,a) when a is a root node
//  (both differences are irrelevant for a single-root tree, where the
//   "a is root" branch is never taken, but are kept for fidelity.)
// =====================================================================
MoveResult modificationtopo_core(const Work& W, const Data& Dat, const Param& P,
                                 const Prior& Pr, Rng& rng,
                                 bool resample_i, bool reattach_root_a){
  (void)Dat;
  const Tree& tr = W.tr;
  const int nch    = static_cast<int>(W.X.size());
  const int nnodes = tr.max_node();                 // R: length(lkldinter[[1]])
  const int ntip   = tr.n_tips();
  const double agemax = P.agemax_fixed ? P.agemax_value
                                       : hauteur(tr, tr.root_nodes[0]);

  const std::vector<int> root1 = tr.root_children;  // R: root[[1]]
  const std::vector<int> root2 = tr.root_nodes;     // R: root[[2]]

  std::vector<int> feuilles; feuilles.reserve(ntip);
  for(int t = 1; t <= ntip; ++t) feuilles.push_back(t);
  auto ht = hauteur_tous(tr, feuilles, agemax);      // {own height, parent height}

  // ----- pick moving node e and regraft target a -----
  std::vector<int> pool;
  for(int nd = 1; nd <= nnodes; ++nd) if(!contains(root2, nd)) pool.push_back(nd);
  auto perm = rng.sample(static_cast<int>(pool.size()),
                         static_cast<int>(pool.size()), false);
  std::vector<int> etot; etot.reserve(pool.size());
  for(int idx : perm) etot.push_back(pool[idx]);

  int e = -1, i = -1, d = -1, f = -1, ifr = -1, a = -1;
  double haut = 0.0;
  bool ouiounon = true;
  for(size_t it = 0; it < etot.size() && ouiounon; ++it){
    e   = etot[it];
    i   = tr.branch_to(e);
    d   = tr.edge[i][0];
    haut = hauteur(tr, d);
    auto ch = tr.children_of(d);
    ifr = (tr.edge[ch[0]][1] == e) ? ch[1] : ch[0];
    f   = tr.edge[ifr][1];

    std::vector<int> quiposs;
    for(int z = 1; z <= nnodes; ++z)
      if(ht.first[z] < haut && ht.second[z] > haut && z != d) quiposs.push_back(z);

    if(!quiposs.empty()){
      auto pr = rng.sample(static_cast<int>(quiposs.size()),
                           static_cast<int>(quiposs.size()), false);
      std::vector<int> We = feuilles_sous_arbre(tr, e);
      int chosen = -1;
      for(int idx : pr){
        int z = quiposs[idx];
        std::vector<int> WW = feuilles_sous_arbre(tr, z);
        WW.insert(WW.end(), We.begin(), We.end());
        bool ok = (z != f);
        if(ok) for(const auto& x : Pr.tipprior){
          int inter = count_intersect(WW, x);
          if(!(inter == 0 || inter == (int)WW.size() || inter == (int)x.size())){ ok = false; break; }
        }
        if(ok){ chosen = z; break; }
      }
      if(chosen >= 0){ a = chosen; ouiounon = false; }
    }
  }
  if(ouiounon) return reject_move(W);

  // ----- working copies -----
  Tree trt = tr;
  VVI  Xt = W.X;   VVD  Tpst = W.Tps;
  VVM  Mt = W.M;   VVM  loiappt = W.loiapp;
  Eigen::MatrixXd Lt = W.L;   Eigen::MatrixXi NLt = W.NL;
  double corrb = 0.0;
  const int ncogntot = static_cast<int>(W.L.rows());

  auto updateML = [&](int ii, int b){
    Mt[ii][b] = transition_matrix(Xt[ii][b], W.Trposs[ii], P.nph[ii],
                                  W.bruit, trt.edge_length[b]);
    loiappt[ii][b] = loiapp(Xt[ii][b], Lcol_pos(Lt, b), P.loiini[ii],
                            W.Trposs[ii], W.bruit, trt.edge_length[b], Tpst[ii][b]);
  };

  // ================= DETACH e (remove d, splice sibling f) =================
  if(!contains(root2, d)){
    int k  = tr.branch_to(d);
    int dd = tr.edge[k][0];
    const double lk = tr.edge_length[k], lifr = tr.edge_length[ifr], den = lk + lifr;
    trt.edge[k] = {dd, f};
    trt.edge_length[k] = lk + lifr;
    for(int ii = 0; ii < nch; ++ii){
      std::vector<int> merged = W.X[ii][k];
      merged.insert(merged.end(), W.X[ii][ifr].begin(), W.X[ii][ifr].end());
      Xt[ii][k] = merged;
      std::vector<double> t;
      for(double tv : W.Tps[ii][k])   t.push_back(tv * lk / den);
      for(double tv : W.Tps[ii][ifr]) t.push_back((lk + tv * lifr) / den);
      Tpst[ii][k] = t;
    }
    for(int c = 0; c < ncogntot; ++c){
      double val = ((W.L(c, ifr) * lifr + lk) * (W.NL(c, ifr) > 0 ? 1.0 : 0.0)
                 + ((W.NL(c, ifr) == 0 && W.NL(c, k) > 0) ? 1.0 : 0.0) * W.L(c, k) * lk) / den;
      Lt(c, k)  = val;
      NLt(c, k) = W.NL(c, k) + W.NL(c, ifr);
    }
    for(int ii = 0; ii < nch; ++ii) updateML(ii, k);
    // corrb : reverse-move laplacian on the merged branch (dbinom NOT logged, per R)
    for(int c = 0; c < ncogntot; ++c)
      if(NLt(c, k) > 1 && Lt(c, k) > lk / trt.edge_length[k]){
        double prob = lk / (lk + W.L(c, ifr) * lifr);
        corrb += std::exp(Rng::dbinom(W.NL(c, k), NLt(c, k) - 1, prob));
      }
    for(int c = 0; c < ncogntot; ++c)
      if(W.L(c, k) > 0 && W.L(c, ifr) > 0)
        corrb += Rng::dbeta(W.L(c, k), W.NL(c, k), 1.0);
  } else {
    // d is a root node -> f becomes a new root; reverse move must redraw priors
    erase_val(trt.root_nodes, d); trt.root_nodes.push_back(f);
    for(int ii = 0; ii < nch; ++ii)
      corrb += sumlogP(W.P[ii], W.X[ii][ifr])
             + Rng::dpois((int)W.X[ii][ifr].size(), W.la[ii] * tr.edge_length[ifr]);
    for(int c = 0; c < ncogntot; ++c){
      if(W.L(c, ifr) != 0) corrb += Rng::dbeta(W.L(c, ifr), W.NL(c, ifr), 1.0);
      corrb += Rng::dpois(W.NL(c, ifr), tr.edge_length[ifr] * W.rho);
    }
  }
  if(contains(root1, d)){ erase_val(trt.root_children, d); trt.root_children.push_back(f); }

  // ================= ATTACH e next to a =================
  const double u = ht.second[a] - haut;
  int j = -1, y = -1;
  if(!contains(root2, a)){
    j  = tr.branch_to(a);
    const double ww = tr.edge_length[j];
    y  = tr.edge[j][0];
    trt.edge[j]   = {y, d};
    trt.edge[ifr] = {d, a};
    trt.edge_length[ifr] = ww - u;
    trt.edge_length[j]   = u;

    // split transforms of original branch j between j (above split) and ifr (below)
    for(int ii = 0; ii < nch; ++ii){
      std::vector<int> Xj, Xifr; std::vector<double> Tj, Tifr;
      for(size_t q = 0; q < W.Tps[ii][j].size(); ++q){
        if(W.Tps[ii][j][q] < u / ww){ Xj.push_back(W.X[ii][j][q]);  Tj.push_back(ww * W.Tps[ii][j][q] / u); }
        else                        { Xifr.push_back(W.X[ii][j][q]);Tifr.push_back((W.Tps[ii][j][q] * ww - u) / (ww - u)); }
      }
      Xt[ii][j] = Xj; Xt[ii][ifr] = Xifr; Tpst[ii][j] = Tj; Tpst[ii][ifr] = Tifr;
    }
    for(int c = 0; c < ncogntot; ++c){ NLt(c,j)=0; Lt(c,j)=0; NLt(c,ifr)=0; Lt(c,ifr)=0; }

    // single borrowing: deterministic placement
    for(int c = 0; c < ncogntot; ++c){
      if(W.NL(c,j) == 1 && W.L(c,j) < u/ww){ NLt(c,j)=1;   Lt(c,j)  = W.L(c,j)*ww/u; }
      if(W.NL(c,j) == 1 && W.L(c,j) > u/ww){ NLt(c,ifr)=1; Lt(c,ifr)= (W.L(c,j)*ww - u)/(ww-u); }
    }
    // multiple borrowings: split count binomially
    std::vector<int> sss2;
    for(int c = 0; c < ncogntot; ++c){
      if(W.NL(c,j) > 1 && W.L(c,j) < u/ww){
        NLt(c,j) = W.NL(c,j); Lt(c,j) = W.L(c,j)*ww/u;
      } else if(W.NL(c,j) > 1 && W.L(c,j) > u/ww){
        double prob = u / (W.L(c,j)*ww);
        int drawn = rng.rbinom(W.NL(c,j) - 1, prob);
        NLt(c,j)   = drawn;
        NLt(c,ifr) = W.NL(c,j) - drawn;
        Lt(c,ifr)  = (W.L(c,j)*ww - u)/(ww - u);
        corrb -= Rng::dbinom(drawn, W.NL(c,j) - 1, prob);
        sss2.push_back(c);
      }
    }
    for(int c : sss2)
      if(NLt(c,j) > 0){ Lt(c,j) = rng.rbeta(NLt(c,j), 1.0); corrb -= Rng::dbeta(Lt(c,j), NLt(c,j), 1.0); }

    for(int ii = 0; ii < nch; ++ii){ updateML(ii,j); updateML(ii,ifr); }
  } else {
    // a is a root node (never reached for a single-root tree)
    const double ww = agemax - hauteur(tr, a);
    erase_val(trt.root_nodes, a); trt.root_nodes.push_back(d);
    if(reattach_root_a){                       // present in topo2, omitted in topo3
      trt.edge[ifr] = {d, a};
      trt.edge_length[ifr] = ww - u;
    }
    for(int ii = 0; ii < nch; ++ii){
      int q = rng.rpois(W.la[ii] * u);
      Xt[ii][ifr]   = sampleX(rng, W.P[ii], q);
      Tpst[ii][ifr] = sorted_unif(rng, (int)Xt[ii][ifr].size());
      corrb -= sumlogP(W.P[ii], Xt[ii][ifr])
             + Rng::dpois((int)Xt[ii][ifr].size(), W.la[ii] * u);
    }
    for(int c = 0; c < ncogntot; ++c){
      NLt(c,ifr) = rng.rpois(u * W.rho);
      Lt(c,ifr)  = (NLt(c,ifr) != 0) ? rng.rbeta(NLt(c,ifr), 1.0) : 0.0;
    }
    for(int ii = 0; ii < nch; ++ii) updateML(ii, ifr);
    for(int c = 0; c < ncogntot; ++c){
      corrb -= Rng::dpois(NLt(c,ifr), u * W.rho);
      if(Lt(c,ifr) != 0) corrb -= Rng::dbeta(Lt(c,ifr), NLt(c,ifr), 1.0);
    }
  }
  if(contains(root1, a)){ erase_val(trt.root_children, a); trt.root_children.push_back(d); }

  // ================= (topo2 only) resample latent vars on branch i (above e) =========
  if(resample_i){
    for(int ii = 0; ii < nch; ++ii){
      int q = rng.rpois(W.la[ii] * trt.edge_length[i]);
      Xt[ii][i]   = sampleX(rng, W.P[ii], q);
      Tpst[ii][i] = sorted_unif(rng, (int)Xt[ii][i].size());
      corrb -= sumlogP(W.P[ii], Xt[ii][i])
             + Rng::dpois((int)Xt[ii][i].size(), W.la[ii] * trt.edge_length[i]);
      corrb += sumlogP(W.P[ii], W.X[ii][i])
             + Rng::dpois((int)W.X[ii][i].size(), W.la[ii] * tr.edge_length[i]);
    }
    for(int c = 0; c < ncogntot; ++c){
      NLt(c,i) = rng.rpois(trt.edge_length[i] * W.rho);
      Lt(c,i)  = (NLt(c,i) != 0) ? rng.rbeta(NLt(c,i), 1.0) : 0.0;
    }
    for(int ii = 0; ii < nch; ++ii) updateML(ii, i);
    for(int c = 0; c < ncogntot; ++c){
      corrb -= Rng::dpois(NLt(c,i), trt.edge_length[i] * W.rho);
      if(Lt(c,i) != 0) corrb -= Rng::dbeta(Lt(c,i), NLt(c,i), 1.0);
    }
    for(int c = 0; c < ncogntot; ++c){
      corrb += Rng::dpois(W.NL(c,i), tr.edge_length[i] * W.rho);
      if(W.L(c,i) != 0) corrb += Rng::dbeta(W.L(c,i), W.NL(c,i), 1.0);
    }
  }

  // ----- quoisur (VV[[11]]) -----
  std::vector<double> quoisur(nch, 0.0);
  if(!contains(root2, a)){
    for(int x = 0; x < nch; ++x)
      quoisur[x] = (double)W.X[x][j].size()
                 + 0.5 * ((a <= ntip && contains(root2, y)) ? 1.0 : 0.0);
  }

  if(testarbre3(trt, trt.root_nodes, nnodes, feuilles)){
    MoveResult r;
    r.trt = std::move(trt); r.Xt = std::move(Xt); r.Tpst = std::move(Tpst);
    r.NLt = std::move(NLt); r.Lt = std::move(Lt);
    r.Mt = std::move(Mt);  r.loiappt = std::move(loiappt);
    r.corrb = corrb; r.d = d; r.f = f; r.quoisur = std::move(quoisur);
    r.la = W.la; r.rho = W.rho; r.bruit = W.bruit;
    return r;
  }
  return reject_move(W);
}

MoveResult modificationtopo2(const Work& W, const Data& Dat, const Param& P,
                             const Prior& Pr, Rng& rng){
  return modificationtopo_core(W, Dat, P, Pr, rng, /*resample_i=*/true,
                               /*reattach_root_a=*/true);
}

MoveResult modificationtopo3(const Work& W, const Data& Dat, const Param& P,
                             const Prior& Pr, Rng& rng){
  return modificationtopo_core(W, Dat, P, Pr, rng, /*resample_i=*/false,
                               /*reattach_root_a=*/false);
}




// =====================================================================
//  Shared core for modificationtopofrere / modificationtopofrere2
//    resample_i : frere resamples latent vars on branch i (above e) and
//                 applies the a-in-root[[1]] fixup
//    add_resc   : frere returns corrb+resc; frere2 returns corrb only
//                 (frere2 computes resc but discards it -- faithful to R)
//  Node e is chosen among nodes with a grandparent; a = e's uncle.
// =====================================================================
MoveResult modificationtopofrere_core(const Work& W, const Data& Dat, const Param& P,
                                      const Prior& Pr, Rng& rng,
                                      bool resample_i, bool add_resc){
  (void)Pr;
  (void)Dat;
  const Tree& tr   = W.tr;
  const int nch    = static_cast<int>(W.X.size());
  const int nnodes = tr.max_node();
  const int ntip   = tr.n_tips();
  const std::vector<int> root1 = tr.root_children;   // R: root[[1]]
  const std::vector<int> root2 = tr.root_nodes;      // R: root[[2]]
  const int ncogntot = static_cast<int>(W.L.rows());
  
  std::vector<int> feuilles; feuilles.reserve(ntip);
  for(int t = 1; t <= ntip; ++t) feuilles.push_back(t);
  
  // ----- candidate moving nodes: those possessing a grandparent -----
  std::vector<int> etot;
  for(int nd = 1; nd <= nnodes; ++nd) if(!contains(root2, nd)) etot.push_back(nd);
  std::vector<int> parent(etot.size()), gp(etot.size());
  std::vector<double> quiposs(etot.size(), 0.0);
  for(size_t z = 0; z < etot.size(); ++z){
    int p = tr.parent_of(etot[z]);
    parent[z] = p;
    int g = (p > 0) ? tr.parent_of(p) : -1;
    gp[z] = g;
    quiposs[z] = (g > 0) ? 1.0 : 0.0;
  }
  double tot = 0; for(double q : quiposs) tot += q;
  if(tot <= 0) return reject_move(W);
  
  int ij = rng.sample(static_cast<int>(etot.size()), 1, false, &quiposs)[0];
  int e = etot[ij];
  int d = parent[ij];
  int g = gp[ij];
  int a;                              // uncle of e = other child of g
  { auto ch = tr.children_of(g); a = (tr.edge[ch[0]][1] == d) ? tr.edge[ch[1]][1]
  : tr.edge[ch[0]][1]; }
  int i = tr.branch_to(e);
  auto st = sous_arbre(tr, e);        // branch indices of e's subtree
  int ifr;                            // sibling branch of e
  { auto ch = tr.children_of(d); ifr = (tr.edge[ch[0]][1] == e) ? ch[1] : ch[0]; }
  int f = tr.edge[ifr][1];
  
  // ----- working copies -----
  Tree trt = tr;
  VVI  Xt = W.X;   VVD  Tpst = W.Tps;
  VVM  Mt = W.M;   VVM  loiappt = W.loiapp;
  Eigen::MatrixXd Lt = W.L;   Eigen::MatrixXi NLt = W.NL;
  double corrb = 0.0, resc = 0.0;
  
  auto updateML = [&](int ii, int b){
    Mt[ii][b] = transition_matrix(Xt[ii][b], W.Trposs[ii], P.nph[ii],
                                  W.bruit, trt.edge_length[b]);
    loiappt[ii][b] = loiapp(Xt[ii][b], Lcol_pos(Lt, b), P.loiini[ii],
                            W.Trposs[ii], W.bruit, trt.edge_length[b], Tpst[ii][b]);
  };
  
  // ================= DETACH e (remove d, splice sibling f into branch k) ====
  int k  = tr.branch_to(d);
  int dd = tr.edge[k][0];                    // == g
  const double lk = tr.edge_length[k], lifr = tr.edge_length[ifr], den = lk + lifr;
  trt.edge[k] = {dd, f};
  trt.edge_length[k] = lk + lifr;
  
  // ================= graft position on branch j (above a) ==================
  int j  = tr.branch_to(a);
  const double ww = tr.edge_length[j];
  const double u  = rng.runif(0, ww);
  int y  = tr.edge[j][0];                     // == g
  trt.edge[j]   = {y, d};
  trt.edge[ifr] = {d, a};
  trt.edge_length[j]   = u;
  trt.edge_length[ifr] = ww - u;
  
  // rescale e's subtree (st) and branch i to fit the new slot
  const double ha = hauteur(tr, a), he = hauteur(tr, e);
  const double ratio = (ha + ww - u) / (he + tr.edge_length[i]);
  for(int b : st) trt.edge_length[b] = tr.edge_length[b] * ratio;
  trt.edge_length[i] = tr.edge_length[i] * ratio;
  { std::set<int> up; for(int b : st) up.insert(tr.edge[b][0]);
  resc = -1.0 * static_cast<double>(up.size()) * std::log(ratio); } // jacrescale=-1
  
  // merge latent vars onto branch k (identical to topo2 detach case)
  for(int ii = 0; ii < nch; ++ii){
    std::vector<int> merged = W.X[ii][k];
    merged.insert(merged.end(), W.X[ii][ifr].begin(), W.X[ii][ifr].end());
    Xt[ii][k] = merged;
    std::vector<double> t;
    for(double tv : W.Tps[ii][k])   t.push_back(tv * lk / den);
    for(double tv : W.Tps[ii][ifr]) t.push_back((lk + tv * lifr) / den);
    Tpst[ii][k] = t;
  }
  for(int c = 0; c < ncogntot; ++c){
    double val = ((W.L(c, ifr) * lifr + lk) * (W.NL(c, ifr) > 0 ? 1.0 : 0.0)
                    + ((W.NL(c, ifr) == 0 && W.NL(c, k) > 0) ? 1.0 : 0.0) * W.L(c, k) * lk) / den;
    Lt(c, k)  = val;
    NLt(c, k) = W.NL(c, k) + W.NL(c, ifr);
  }
  for(int ii = 0; ii < nch; ++ii) updateML(ii, k);
  
  // reverse-move density on the merged branch (+ the frere-specific graft prior ratio)
  corrb += std::log(1.0 / den) - std::log(1.0 / ww);
  for(int c = 0; c < ncogntot; ++c){
    if(NLt(c, k) > 1 && Lt(c, k) > lk / trt.edge_length[k]){
      double prob = lk / (lk + W.L(c, ifr) * lifr);
      corrb += std::exp(Rng::dbinom(W.NL(c, k), NLt(c, k) - 1, prob)); // raw (R quirk)
    } 
  }
  for(int c = 0; c < ncogntot; ++c)
      if(W.L(c, k) > 0 && W.L(c, ifr) > 0)
        corrb += Rng::dbeta(W.L(c, k), W.NL(c, k), 1.0);
      
      if(contains(root1, d)){ erase_val(trt.root_children, d); trt.root_children.push_back(f); }
      
      // ================= split original branch j between j (above) and ifr (below) =====
      for(int ii = 0; ii < nch; ++ii){
        std::vector<int> Xj, Xifr; std::vector<double> Tj, Tifr;
        for(size_t q = 0; q < W.Tps[ii][j].size(); ++q){
          if(W.Tps[ii][j][q] < u / ww){ Xj.push_back(W.X[ii][j][q]);   Tj.push_back(ww * W.Tps[ii][j][q] / u); }
          else                        { Xifr.push_back(W.X[ii][j][q]); Tifr.push_back((W.Tps[ii][j][q] * ww - u) / (ww - u)); }
        }
        Xt[ii][j] = Xj; Xt[ii][ifr] = Xifr; Tpst[ii][j] = Tj; Tpst[ii][ifr] = Tifr;
      }
      for(int c = 0; c < ncogntot; ++c){ NLt(c,j)=0; Lt(c,j)=0; NLt(c,ifr)=0; Lt(c,ifr)=0; }
      
      for(int c = 0; c < ncogntot; ++c){
        if(W.NL(c,j) == 1 && W.L(c,j) < u/ww){ NLt(c,j)=1;   Lt(c,j)  = W.L(c,j)*ww/u; }
        if(W.NL(c,j) == 1 && W.L(c,j) > u/ww){ NLt(c,ifr)=1; Lt(c,ifr)= (W.L(c,j)*ww - u)/(ww-u); }
      }
      std::vector<int> sss2;
      for(int c = 0; c < ncogntot; ++c){
        if(W.NL(c,j) > 1 && W.L(c,j) < u/ww){
          NLt(c,j) = W.NL(c,j); Lt(c,j) = W.L(c,j)*ww/u;
        } else if(W.NL(c,j) > 1 && W.L(c,j) > u/ww){
          double prob = u / (W.L(c,j)*ww);
          int drawn = rng.rbinom(W.NL(c,j) - 1, prob);
          NLt(c,j)   = drawn;
          NLt(c,ifr) = W.NL(c,j) - drawn;
          Lt(c,ifr)  = (W.L(c,j)*ww - u)/(ww - u);
          corrb -= Rng::dbinom(drawn, W.NL(c,j) - 1, prob);
          sss2.push_back(c);
        }
      }
      for(int c : sss2)
        if(NLt(c,j) > 0){ Lt(c,j) = rng.rbeta(NLt(c,j), 1.0); corrb -= Rng::dbeta(Lt(c,j), NLt(c,j), 1.0); }
        for(int ii = 0; ii < nch; ++ii){ updateML(ii,j); updateML(ii,ifr); }
        
        // ================= (frere only) a-in-root[[1]] fixup + resample branch i ====
        if(resample_i){
          if(contains(root1, a)){ erase_val(trt.root_children, a); trt.root_children.push_back(d); }
          
          for(int ii = 0; ii < nch; ++ii){
            int q = rng.rpois(W.la[ii] * tr.edge_length[i]);            // count uses OLD length
            Xt[ii][i]   = sampleX(rng, W.P[ii], q);
            Tpst[ii][i] = sorted_unif(rng, (int)Xt[ii][i].size());
            corrb -= sumlogP(W.P[ii], Xt[ii][i])
              + Rng::dpois((int)Xt[ii][i].size(), W.la[ii] * trt.edge_length[i]); // NEW length
            corrb += sumlogP(W.P[ii], W.X[ii][i])
              + Rng::dpois((int)W.X[ii][i].size(), W.la[ii] * tr.edge_length[i]); // OLD length
          }
          for(int c = 0; c < ncogntot; ++c){
            NLt(c,i) = rng.rpois(trt.edge_length[i] * W.rho);
            Lt(c,i)  = (NLt(c,i) != 0) ? rng.rbeta(NLt(c,i), 1.0) : 0.0;
          }
          for(int ii = 0; ii < nch; ++ii) updateML(ii, i);
          for(int c = 0; c < ncogntot; ++c){
            corrb -= Rng::dpois(NLt(c,i), trt.edge_length[i] * W.rho);
            if(Lt(c,i) != 0) corrb -= Rng::dbeta(Lt(c,i), NLt(c,i), 1.0);
          }
          for(int c = 0; c < ncogntot; ++c){
            corrb += Rng::dpois(W.NL(c,i), tr.edge_length[i] * W.rho);
            if(W.L(c,i) != 0) corrb += Rng::dbeta(W.L(c,i), W.NL(c,i), 1.0);
          }
        }
        
        // ----- quoisur (VV[[11]]) -----
        std::vector<double> quoisur(nch, 0.0);
        if(!contains(root2, a)){                    // always true here (a has a parent)
          for(int x = 0; x < nch; ++x)
            quoisur[x] = (double)W.X[x][j].size()
            + 0.5 * ((a <= ntip && contains(root2, y)) ? 1.0 : 0.0);
        }
        
        if(testarbre3(trt, trt.root_nodes, nnodes, feuilles)){
          MoveResult r;
          r.trt = std::move(trt); r.Xt = std::move(Xt); r.Tpst = std::move(Tpst);
          r.NLt = std::move(NLt); r.Lt = std::move(Lt);
          r.Mt = std::move(Mt);  r.loiappt = std::move(loiappt);
          r.corrb = corrb + (add_resc ? resc : 0.0);
          r.d = d; r.f = f; r.quoisur = std::move(quoisur);
          r.la = W.la; r.rho = W.rho; r.bruit = W.bruit;
          return r;
        }
        return reject_move(W);
}

MoveResult modificationtopofrere(const Work& W, const Data& Dat, const Param& P,
                                 const Prior& Pr, Rng& rng){
  return modificationtopofrere_core(W, Dat, P, Pr, rng, /*resample_i=*/true,
                                    /*add_resc=*/true);
}

MoveResult modificationtopofrere2(const Work& W, const Data& Dat, const Param& P,
                                  const Prior& Pr, Rng& rng){
  return modificationtopofrere_core(W, Dat, P, Pr, rng, /*resample_i=*/false,
                                    /*add_resc=*/false);
}


// ---- masked column sums (match R's dpois/dbeta over columns) ----
static double sum_dpois_col(const Eigen::MatrixXi& NL,int col,double rate,int n){
  double s=0; for(int c=0;c<n;++c) s+=Rng::dpois(NL(c,col),rate); return s; }
static double sum_dbeta_col(const Eigen::MatrixXd& L,const Eigen::MatrixXi& NL,int col,int n){
  double s=0; for(int c=0;c<n;++c) if(L(c,col)>0) s+=Rng::dbeta(L(c,col),NL(c,col),1.0); return s; }

inline std::vector<int> branches_with_parent(const Tree& T,int nd){
  std::vector<int> r; for(int b=0;b<T.n_edges();++b) if(T.edge[b][0]==nd) r.push_back(b); return r; }

// outgroup borrowing split: "above" gets the pre-graft part, "below" keeps the rest
// (uses the +1 binomial convention specific to the outgroup moves). Updates corrb.
static void outgroup_split(int nch,int n,int above,int below,double haut,double lenhk,
                           const VVI& X,const VVD& Tps,const Eigen::MatrixXi& NL,const Eigen::MatrixXd& L,
                           VVI& Xt,VVD& Tpst,Eigen::MatrixXi& NLt,Eigen::MatrixXd& Lt,double& corrb,Rng& rng){
  const double thr=haut/lenhk;
  for(int ii=0;ii<nch;++ii){
    std::vector<int> Xa,Xb; std::vector<double> Ta,Tb;
    for(size_t q=0;q<Tps[ii][below].size();++q){
      if(Tps[ii][below][q]<thr){ Xa.push_back(X[ii][below][q]); Ta.push_back(lenhk*Tps[ii][below][q]/haut); }
      else { Xb.push_back(X[ii][below][q]); Tb.push_back((Tps[ii][below][q]*lenhk-haut)/(lenhk-haut)); }
    }
    Xt[ii][above]=Xa; Xt[ii][below]=Xb; Tpst[ii][above]=Ta; Tpst[ii][below]=Tb;
  }
  for(int c=0;c<n;++c){ NLt(c,below)=0; Lt(c,below)=0; NLt(c,above)=0; Lt(c,above)=0; }
  for(int c=0;c<n;++c){
    if(NL(c,below)==1 && L(c,below)<thr){ NLt(c,above)=1; Lt(c,above)=L(c,below)*lenhk/haut; }
    if(NL(c,below)==1 && L(c,below)>thr){ NLt(c,below)=1; Lt(c,below)=(L(c,below)*lenhk-haut)/(lenhk-haut); }
  }
  std::vector<int> sss2;
  for(int c=0;c<n;++c){
    if(NL(c,below)>1 && L(c,below)<thr){ NLt(c,above)=NL(c,below); Lt(c,above)=L(c,below)*lenhk/haut; }
    else if(NL(c,below)>1 && L(c,below)>thr){
      double prob=(L(c,below)*lenhk-haut)/(L(c,below)*lenhk);
      int drawn=rng.rbinom(NL(c,below)-1,prob)+1;       // R: rbinom(...)+1
      NLt(c,below)=drawn; NLt(c,above)=NL(c,below)-drawn;
      Lt(c,below)=(L(c,below)*lenhk-haut)/(lenhk-haut);
      corrb-=Rng::dbinom(drawn-1,NL(c,below)-1,prob);   // R uses NLt[,hk]-1 == drawn-1
      sss2.push_back(c);
    }
  }
  for(int c:sss2) if(NLt(c,above)>0){ Lt(c,above)=rng.rbeta(NLt(c,above),1.0); corrb-=Rng::dbeta(Lt(c,above),NLt(c,above),1.0); }
}

// extended validity used by the outgroup moves
static bool tree_valid_ext(const Tree& T,int nnodes,int ntip,const std::vector<int>& feuilles){
  if(!testarbre3(T,T.root_nodes,nnodes,feuilles)) return false;
  std::vector<int> a; for(auto&e:T.edge) a.push_back(e[1]); for(int r:T.root_nodes) a.push_back(r);
  std::sort(a.begin(),a.end()); if((int)a.size()!=nnodes) return false;
  for(int q=0;q<nnodes;++q) if(a[q]!=q+1) return false;
  std::set<int> par; for(auto&e:T.edge) par.insert(e[0]);
  std::vector<int> pv(par.begin(),par.end());
  if((int)pv.size()!=nnodes-ntip) return false;
  for(int q=0;q<(int)pv.size();++q) if(pv[q]!=ntip+1+q) return false;
  return true;
}

// =====================================================================
//  modificationtopooutgroupsansresc  (agemax fixed)
// =====================================================================
MoveResult modificationtopooutgroupsansresc(const Work& W, const Data& Dat, const Param& P,
                                            const Prior& Pr, Rng& rng){
  (void)Pr;
  (void)Dat;
  const Tree& tr = W.tr;
  const int nch = (int)W.X.size();
  const int nnodes = tr.max_node();
  const int ntip = tr.n_tips();
  const int n = (int)W.L.rows();
  const std::vector<int> root2 = tr.root_nodes;
  const int rootnode = root2[0];
  const double agemax = P.agemax_fixed ? P.agemax_value : hauteur(tr, rootnode);
  std::vector<int> feuilles; for(int t=1;t<=ntip;++t) feuilles.push_back(t);
  auto in=[&](const std::vector<int>& v,int x){ return std::find(v.begin(),v.end(),x)!=v.end(); };
  
  Tree trt=tr; VVI Xt=W.X; VVD Tpst=W.Tps; VVM Mt=W.M, loiappt=W.loiapp;
  Eigen::MatrixXd Lt=W.L; Eigen::MatrixXi NLt=W.NL; double corrb=0.0;
  auto updateAll=[&](){ for(int ii=0;ii<nch;++ii) for(int b=0;b<trt.n_edges();++b){
    Mt[ii][b]=transition_matrix(Xt[ii][b],W.Trposs[ii],P.nph[ii],W.bruit,trt.edge_length[b]);
    loiappt[ii][b]=loiapp(Xt[ii][b],Lcol_pos(Lt,b),P.loiini[ii],W.Trposs[ii],W.bruit,trt.edge_length[b],Tpst[ii][b]); } };
  
  if(rng.runif() < 0.5){
    // ---------- turn a node into the outgroup ----------
    std::vector<int> excl=root2;
    { auto rc=tr.children_of(rootnode); if(rc[0]>=0){ excl.push_back(tr.edge[rc[0]][1]); excl.push_back(tr.edge[rc[1]][1]); } }
    std::vector<int> cand; for(int z=1;z<=nnodes;++z) if(!in(excl,z)) cand.push_back(z);
    if(cand.empty()) return reject_move(W);
    int e=cand[rng.sample((int)cand.size(),1,false)[0]];
    int i=tr.branch_to(e), d=tr.edge[i][0];
    auto st=sous_arbre(tr,e);
    int ifr; { auto ch=tr.children_of(d); ifr=(tr.edge[ch[0]][1]==e)?ch[1]:ch[0]; }
    int f=tr.edge[ifr][1], g=tr.parent_of(d), j=tr.branch_to(d);
    
    trt.edge[j]={d,rootnode}; trt.edge[ifr]={g,f};
    trt.edge_length[ifr]=tr.edge_length[j]+tr.edge_length[ifr];
    auto bbb=branches_with_parent(trt,rootnode);
    double hautmax=agemax-hauteur(tr,e); for(int b:bbb) hautmax=std::min(hautmax,trt.edge_length[b]);
    if(hautmax<0) return reject_move(W);
    trt.edge_length[j]=rng.runif(0,hautmax);
    corrb-=std::log(1.0/hautmax);
    for(int b:bbb) trt.edge_length[b]-=trt.edge_length[j];
    trt.edge_length[i]=agemax-hauteur(tr,e);
    trt.root_nodes={d};
    
    double haut=hauteur(trt,e);
    auto htt=hauteur_tous(trt,feuilles,agemax);
    std::vector<int> forb={e,f}; for(int r:root2) forb.push_back(r); for(int b:st) forb.push_back(tr.edge[b][1]);
    int nq=0; for(int z=1;z<=nnodes;++z) if(htt.second[z]>haut && !in(forb,z)) ++nq;
    corrb+=std::log((double)(tr.n_edges()-2))-std::log((double)nq);
    
    // merge branch j into ifr (original lengths)
    const double ljo=tr.edge_length[j], lifro=tr.edge_length[ifr], deno=ljo+lifro;
    for(int ii=0;ii<nch;++ii){
      std::vector<int> m=W.X[ii][j]; m.insert(m.end(),W.X[ii][ifr].begin(),W.X[ii][ifr].end()); Xt[ii][ifr]=m;
      std::vector<double> t; for(double tv:W.Tps[ii][j]) t.push_back(tv*ljo);
      for(double tv:W.Tps[ii][ifr]) t.push_back(ljo+tv*lifro);
      for(double& tv:t) tv/=deno;
      Tpst[ii][ifr]=t;
    }
    for(int c=0;c<n;++c){
      Lt(c,ifr)=((ljo+W.L(c,ifr)*lifro)*(W.NL(c,ifr)>0?1.0:0.0)
                   +((W.NL(c,ifr)==0&&W.NL(c,j)>0)?1.0:0.0)*W.L(c,j)*ljo)/deno;
      NLt(c,ifr)=W.NL(c,j)+W.NL(c,ifr);
    }
    for(int c=0;c<n;++c) if(NLt(c,ifr)>1 && Lt(c,ifr)>ljo/deno){
      double prob=ljo/(ljo+W.L(c,ifr)*lifro);
      corrb+=std::exp(Rng::dbinom(W.NL(c,j),NLt(c,ifr)-1,prob));           // raw (R log=FALSE)
    }
    for(int c=0;c<n;++c) if(W.L(c,j)>0 && W.L(c,ifr)>0) corrb+=Rng::dbeta(W.L(c,j),W.NL(c,j),1.0);
    
    // resample branch i (with reverse term) and branch j (forward only)
    for(int ii=0;ii<nch;++ii){
      int qi=rng.rpois(W.la[ii]*trt.edge_length[i]);
      Xt[ii][i]=sampleX(rng,W.P[ii],qi); Tpst[ii][i]=sorted_unif(rng,(int)Xt[ii][i].size());
      corrb-=sumlogP(W.P[ii],Xt[ii][i])+Rng::dpois((int)Xt[ii][i].size(),W.la[ii]*trt.edge_length[i]);
      corrb+=sumlogP(W.P[ii],W.X[ii][i])+Rng::dpois((int)W.X[ii][i].size(),W.la[ii]*tr.edge_length[i]);
      int qj=rng.rpois(W.la[ii]*trt.edge_length[j]);
      Xt[ii][j]=sampleX(rng,W.P[ii],qj); Tpst[ii][j]=sorted_unif(rng,(int)Xt[ii][j].size());
      corrb-=sumlogP(W.P[ii],Xt[ii][j])+Rng::dpois((int)Xt[ii][j].size(),W.la[ii]*trt.edge_length[j]);
    }
    for(int c=0;c<n;++c){ NLt(c,i)=rng.rpois(trt.edge_length[i]*W.rho); Lt(c,i)=NLt(c,i)?rng.rbeta(NLt(c,i),1.0):0.0; }
    corrb-=sum_dpois_col(NLt,i,trt.edge_length[i]*W.rho,n); corrb+=sum_dpois_col(W.NL,i,tr.edge_length[i]*W.rho,n);
    corrb-=sum_dbeta_col(Lt,NLt,i,n);                       corrb+=sum_dbeta_col(W.L,W.NL,i,n);
    for(int c=0;c<n;++c){ NLt(c,j)=rng.rpois(trt.edge_length[j]*W.rho); Lt(c,j)=NLt(c,j)?rng.rbeta(NLt(c,j),1.0):0.0; }
    corrb-=sum_dpois_col(NLt,j,trt.edge_length[j]*W.rho,n);
    corrb-=sum_dbeta_col(Lt,NLt,j,n);
    
    updateAll();
    
    // reverse-direction (trc) term: only the draw + corrb line matter
    { double range=hauteur(trt,g)-std::max(hauteur(trt,e),hauteur(trt,f));
      if(range>0){ rng.runif(0,range); corrb+=std::log(1.0/range); } else corrb=std::nan(""); }
  } else {
    // ---------- move the existing outgroup ----------
    auto aaa=branches_with_parent(tr,rootnode);
    int emax=aaa[0], emin=aaa[0];
    for(int b:aaa){ if(tr.edge_length[b]>tr.edge_length[emax]) emax=b; if(tr.edge_length[b]<tr.edge_length[emin]) emin=b; }
    int e=tr.edge[emax][1], f=tr.edge[emin][1];
    auto st=sous_arbre(tr,e);
    double haut_e=hauteur(tr,e);
    auto htt=hauteur_tous(tr,feuilles,agemax);
    std::vector<int> forb={e,f}; for(int r:root2) forb.push_back(r); for(int b:st) forb.push_back(tr.edge[b][1]);
    std::vector<int> quiposs; for(int z=1;z<=nnodes;++z) if(htt.second[z]>haut_e && !in(forb,z)) quiposs.push_back(z);
    if(quiposs.empty()) return reject_move(W);
    int a=quiposs[rng.sample((int)quiposs.size(),1,false)[0]];
    int z=tr.parent_of(a), hk=tr.branch_to(a);
    corrb+= -std::log((double)(tr.n_edges()-2))+std::log((double)quiposs.size());
    int i=tr.branch_to(e), d=tr.edge[i][0];
    int ifr; { auto ch=tr.children_of(d); ifr=(tr.edge[ch[0]][1]==e)?ch[1]:ch[0]; }
    f=tr.edge[ifr][1];
    
    trt.edge[ifr]={z,rootnode}; trt.edge[hk]={rootnode,a};
    double range=hauteur(tr,z)-std::max(hauteur(tr,e),hauteur(tr,a));
    if(range<=0) return reject_move(W);
    double haut=rng.runif(0,range);
    if(haut<0) return reject_move(W);
    trt.edge_length[ifr]=haut;
    trt.edge_length[hk]=tr.edge_length[hk]-haut;
    trt.edge_length[i]=hauteur(tr,z)-haut-hauteur(tr,e);
    auto ccc=branches_with_parent(trt,f);
    for(int b:ccc) trt.edge_length[b]+=agemax-hauteur(trt,f);
    corrb-=std::log(1.0/range);
    trt.root_nodes={f};
    
    outgroup_split(nch,n,/*above=*/ifr,/*below=*/hk,haut,tr.edge_length[hk],
                   W.X,W.Tps,W.NL,W.L,Xt,Tpst,NLt,Lt,corrb,rng);
    for(int ii=0;ii<nch;++ii)
      corrb+=sumlogP(W.P[ii],W.X[ii][ifr])+Rng::dpois((int)W.X[ii][ifr].size(),W.la[ii]*tr.edge_length[ifr]);
    corrb+=sum_dpois_col(W.NL,ifr,tr.edge_length[ifr]*W.rho,n)+sum_dbeta_col(W.L,W.NL,ifr,n);
    
    for(int ii=0;ii<nch;++ii){
      int qi=rng.rpois(W.la[ii]*trt.edge_length[i]);
      Xt[ii][i]=sampleX(rng,W.P[ii],qi); Tpst[ii][i]=sorted_unif(rng,(int)Xt[ii][i].size());
      corrb-=sumlogP(W.P[ii],Xt[ii][i])+Rng::dpois((int)Xt[ii][i].size(),W.la[ii]*trt.edge_length[i]);
      corrb+=sumlogP(W.P[ii],W.X[ii][i])+Rng::dpois((int)W.X[ii][i].size(),W.la[ii]*tr.edge_length[i]);
    }
    for(int c=0;c<n;++c){ NLt(c,i)=rng.rpois(trt.edge_length[i]*W.rho); Lt(c,i)=NLt(c,i)?rng.rbeta(NLt(c,i),1.0):0.0; }
    corrb-=sum_dpois_col(NLt,i,trt.edge_length[i]*W.rho,n); corrb+=sum_dpois_col(W.NL,i,tr.edge_length[i]*W.rho,n);
    corrb-=sum_dbeta_col(Lt,NLt,i,n);                       corrb+=sum_dbeta_col(W.L,W.NL,i,n);
    
    updateAll();
    
    // reverse-direction (trc) term
    { auto bbb=branches_with_parent(trt,rootnode);
      double hmax=agemax-hauteur(trt,e); for(int b:bbb) hmax=std::min(hmax,trt.edge_length[b]);
      if(hmax>0){ rng.runif(0,hmax); corrb-=std::log(1.0/hmax); } else corrb=std::nan(""); }
  }
  
  if(!std::isnan(corrb) && tree_valid_ext(trt,nnodes,ntip,feuilles)){
    MoveResult r; r.trt=std::move(trt); r.Xt=std::move(Xt); r.Tpst=std::move(Tpst);
    r.NLt=std::move(NLt); r.Lt=std::move(Lt); r.Mt=std::move(Mt); r.loiappt=std::move(loiappt);
    r.corrb=corrb; r.d=0; r.f=0; r.quoisur.assign(nch,0.2); r.la=W.la; r.rho=W.rho; r.bruit=W.bruit;
    return r;
  }
  return reject_move(W);
}

// =====================================================================
//  modificationtopooutgroupsansresc_agevar  (agemax variable)
//  NOTE: branch-1 draws the root split from rgamma(shape=0,...) in the
//  original, which is degenerate (edge length -> 0) and thus always
//  rejects. Ported faithfully; effectively a no-op transform branch.
// =====================================================================
MoveResult modificationtopooutgroupsansresc_agevar(const Work& W, const Data& Dat, const Param& P,
                                                   const Prior& Pr, Rng& rng){
  (void)Pr;
  (void)Dat;
  const Tree& tr = W.tr;
  const int nch=(int)W.X.size(), nnodes=tr.max_node(), ntip=tr.n_tips(), n=(int)W.L.rows();
  const std::vector<int> root2=tr.root_nodes; const int rootnode=root2[0];
  const double agemax = P.agemax_fixed ? P.agemax_value : hauteur(tr,rootnode);
  std::vector<int> feuilles; for(int t=1;t<=ntip;++t) feuilles.push_back(t);
  auto in=[&](const std::vector<int>& v,int x){ return std::find(v.begin(),v.end(),x)!=v.end(); };
  
  Tree trt=tr; VVI Xt=W.X; VVD Tpst=W.Tps; VVM Mt=W.M, loiappt=W.loiapp;
  Eigen::MatrixXd Lt=W.L; Eigen::MatrixXi NLt=W.NL; double corrb=0.0;
  auto updateAll=[&](){ for(int ii=0;ii<nch;++ii) for(int b=0;b<trt.n_edges();++b){
    Mt[ii][b]=transition_matrix(Xt[ii][b],W.Trposs[ii],P.nph[ii],W.bruit,trt.edge_length[b]);
    loiappt[ii][b]=loiapp(Xt[ii][b],Lcol_pos(Lt,b),P.loiini[ii],W.Trposs[ii],W.bruit,trt.edge_length[b],Tpst[ii][b]); } };
  
  if(rng.runif() < 0.5){
    std::vector<int> excl=root2;
    { auto rc=tr.children_of(rootnode); if(rc[0]>=0){ excl.push_back(tr.edge[rc[0]][1]); excl.push_back(tr.edge[rc[1]][1]); } }
    std::vector<int> cand; for(int z=1;z<=nnodes;++z) if(!in(excl,z)) cand.push_back(z);
    if(cand.empty()) return reject_move(W);
    int e=cand[rng.sample((int)cand.size(),1,false)[0]];
    int i=tr.branch_to(e), d=tr.edge[i][0];
    auto st=sous_arbre(tr,e);
    int ifr; { auto ch=tr.children_of(d); ifr=(tr.edge[ch[0]][1]==e)?ch[1]:ch[0]; }
    int f=tr.edge[ifr][1], g=tr.parent_of(d), j=tr.branch_to(d);
    
    trt.edge[j]={d,rootnode}; trt.edge[ifr]={g,f};
    trt.edge_length[ifr]=tr.edge_length[j]+tr.edge_length[ifr];
    corrb+=std::log(1.0/(hauteur(tr,g)-std::max(hauteur(tr,e),hauteur(tr,f))));
    double roothgt=hauteur(tr,rootnode), rate=10.0/roothgt;
    double hautplus=0.0;                        // rgamma(shape=0,...) == 0  (degenerate)
    trt.edge_length[j]=hautplus;
    corrb-=Rng::dgamma(hautplus,0.0,rate);      // -> +Inf; move will reject (len j == 0)
    trt.edge_length[i]=hautplus+hauteur(tr,rootnode)-hauteur(tr,e);
    trt.root_nodes={d};
    
    double haut=hauteur(trt,e);
    auto htt=hauteur_tous(trt,feuilles,agemax);
    std::vector<int> forb={e,f}; for(int r:root2) forb.push_back(r); for(int b:st) forb.push_back(tr.edge[b][1]);
    int nq=0; for(int z=1;z<=nnodes;++z) if(htt.second[z]>haut && !in(forb,z)) ++nq;
    corrb+=std::log((double)(tr.n_edges()-2))-std::log((double)nq);
    
    const double ljo=tr.edge_length[j], lifro=tr.edge_length[ifr], deno=ljo+lifro;
    for(int ii=0;ii<nch;++ii){
      std::vector<int> m=W.X[ii][j]; m.insert(m.end(),W.X[ii][ifr].begin(),W.X[ii][ifr].end()); Xt[ii][ifr]=m;
      std::vector<double> t; for(double tv:W.Tps[ii][j]) t.push_back(tv*ljo);
      for(double tv:W.Tps[ii][ifr]) t.push_back(ljo+tv*lifro);
      for(double& tv:t) tv/=deno;
      Tpst[ii][ifr]=t;
    }
    for(int c=0;c<n;++c){
      Lt(c,ifr)=((ljo+W.L(c,ifr)*lifro)*(W.NL(c,ifr)>0?1.0:0.0)
                   +((W.NL(c,ifr)==0&&W.NL(c,j)>0)?1.0:0.0)*W.L(c,j)*ljo)/deno;
      NLt(c,ifr)=W.NL(c,j)+W.NL(c,ifr);
    }
    for(int c=0;c<n;++c) if(NLt(c,ifr)>1 && Lt(c,ifr)>ljo/deno){
      double prob=ljo/(ljo+W.L(c,ifr)*lifro); corrb+=std::exp(Rng::dbinom(W.NL(c,j),NLt(c,ifr)-1,prob)); }
    for(int c=0;c<n;++c) if(W.L(c,j)>0 && W.L(c,ifr)>0) corrb+=Rng::dbeta(W.L(c,j),W.NL(c,j),1.0);
    
    for(int ii=0;ii<nch;++ii){
      int qi=rng.rpois(W.la[ii]*trt.edge_length[i]);
      Xt[ii][i]=sampleX(rng,W.P[ii],qi); Tpst[ii][i]=sorted_unif(rng,(int)Xt[ii][i].size());
      corrb-=sumlogP(W.P[ii],Xt[ii][i])+Rng::dpois((int)Xt[ii][i].size(),W.la[ii]*trt.edge_length[i]);
      corrb+=sumlogP(W.P[ii],W.X[ii][i])+Rng::dpois((int)W.X[ii][i].size(),W.la[ii]*tr.edge_length[i]);
      int qj=rng.rpois(W.la[ii]*trt.edge_length[j]);
      Xt[ii][j]=sampleX(rng,W.P[ii],qj); Tpst[ii][j]=sorted_unif(rng,(int)Xt[ii][j].size());
      corrb-=sumlogP(W.P[ii],Xt[ii][j])+Rng::dpois((int)Xt[ii][j].size(),W.la[ii]*trt.edge_length[j]);
    }
    for(int c=0;c<n;++c){ NLt(c,i)=rng.rpois(trt.edge_length[i]*W.rho); Lt(c,i)=NLt(c,i)?rng.rbeta(NLt(c,i),1.0):0.0; }
    corrb-=sum_dpois_col(NLt,i,trt.edge_length[i]*W.rho,n); corrb+=sum_dpois_col(W.NL,i,tr.edge_length[i]*W.rho,n);
    corrb-=sum_dbeta_col(Lt,NLt,i,n);                       corrb+=sum_dbeta_col(W.L,W.NL,i,n);
    for(int c=0;c<n;++c){ NLt(c,j)=rng.rpois(trt.edge_length[j]*W.rho); Lt(c,j)=NLt(c,j)?rng.rbeta(NLt(c,j),1.0):0.0; }
    corrb-=sum_dpois_col(NLt,j,trt.edge_length[j]*W.rho,n);
    corrb-=sum_dbeta_col(Lt,NLt,j,n);
    updateAll();
    // (no trc block in agevar branch 1)
  } else {
    auto aaa=branches_with_parent(tr,rootnode);
    int emax=aaa[0], emin=aaa[0];
    for(int b:aaa){ if(tr.edge_length[b]>tr.edge_length[emax]) emax=b; if(tr.edge_length[b]<tr.edge_length[emin]) emin=b; }
    int e=tr.edge[emax][1], f=tr.edge[emin][1];
    auto st=sous_arbre(tr,e);
    double haut_e=hauteur(tr,e);
    auto htt=hauteur_tous(tr,feuilles,agemax);
    corrb+=Rng::dgamma(tr.edge_length[emin],1.0,10.0/hauteur(tr,f));   // min(root branches)
    std::vector<int> forb={e,f}; for(int r:root2) forb.push_back(r); for(int b:st) forb.push_back(tr.edge[b][1]);
    std::vector<int> quiposs; for(int z=1;z<=nnodes;++z) if(htt.second[z]>haut_e && !in(forb,z)) quiposs.push_back(z);
    if(quiposs.empty()) return reject_move(W);
    int a=quiposs[rng.sample((int)quiposs.size(),1,false)[0]];
    int z=tr.parent_of(a), hk=tr.branch_to(a);
    corrb+= -std::log((double)(tr.n_edges()-2))+std::log((double)quiposs.size());
    int i=tr.branch_to(e), d=tr.edge[i][0];
    int ifr; { auto ch=tr.children_of(d); ifr=(tr.edge[ch[0]][1]==e)?ch[1]:ch[0]; }
    f=tr.edge[ifr][1];
    trt.edge[ifr]={z,rootnode}; trt.edge[hk]={rootnode,a};
    double range=hauteur(tr,z)-std::max(hauteur(tr,e),hauteur(tr,a));
    if(range<=0) return reject_move(W);
    double haut=rng.runif(0,range);
    if(haut<0) return reject_move(W);
    trt.edge_length[ifr]=haut; trt.edge_length[hk]=tr.edge_length[hk]-haut;
    trt.edge_length[i]=hauteur(tr,z)-haut-hauteur(tr,e);
    corrb-=std::log(1.0/range);                 // (agevar: no ccc extension)
    trt.root_nodes={f};
    
    outgroup_split(nch,n,ifr,hk,haut,tr.edge_length[hk],W.X,W.Tps,W.NL,W.L,Xt,Tpst,NLt,Lt,corrb,rng);
    for(int ii=0;ii<nch;++ii)
      corrb+=sumlogP(W.P[ii],W.X[ii][ifr])+Rng::dpois((int)W.X[ii][ifr].size(),W.la[ii]*tr.edge_length[ifr]);
    corrb+=sum_dpois_col(W.NL,ifr,tr.edge_length[ifr]*W.rho,n)+sum_dbeta_col(W.L,W.NL,ifr,n);
    
    for(int ii=0;ii<nch;++ii){
      int qi=rng.rpois(W.la[ii]*trt.edge_length[i]);
      Xt[ii][i]=sampleX(rng,W.P[ii],qi); Tpst[ii][i]=sorted_unif(rng,(int)Xt[ii][i].size());
      corrb-=sumlogP(W.P[ii],Xt[ii][i])+Rng::dpois((int)Xt[ii][i].size(),W.la[ii]*trt.edge_length[i]);
      corrb+=sumlogP(W.P[ii],W.X[ii][i])+Rng::dpois((int)W.X[ii][i].size(),W.la[ii]*tr.edge_length[i]);
    }
    for(int c=0;c<n;++c){ NLt(c,i)=rng.rpois(trt.edge_length[i]*W.rho); Lt(c,i)=NLt(c,i)?rng.rbeta(NLt(c,i),1.0):0.0; }
    corrb-=sum_dpois_col(NLt,i,trt.edge_length[i]*W.rho,n); corrb+=sum_dpois_col(W.NL,i,tr.edge_length[i]*W.rho,n);
    corrb-=sum_dbeta_col(Lt,NLt,i,n);                       corrb+=sum_dbeta_col(W.L,W.NL,i,n);
    updateAll();
    // (no trc block in agevar branch 2)
  }
  
  if(!std::isnan(corrb) && tree_valid_ext(trt,nnodes,ntip,feuilles)){
    MoveResult r; r.trt=std::move(trt); r.Xt=std::move(Xt); r.Tpst=std::move(Tpst);
    r.NLt=std::move(NLt); r.Lt=std::move(Lt); r.Mt=std::move(Mt); r.loiappt=std::move(loiappt);
    r.corrb=corrb; r.d=0; r.f=0; r.quoisur.assign(nch,0.2); r.la=W.la; r.rho=W.rho; r.bruit=W.bruit;
    return r;
  }
  return reject_move(W);
}

} // namespace  (end anonymous)

// =====================================================================
//  Public dispatcher (declared in topology.hpp) — replaces the stub.
// =====================================================================
int gibbstopopart2(Work& W, const Data& Dat, const Param& P, const Prior& Pr,
                   int ncogn, bool fin, Rng& rng){
  (void)fin;
  const int nch = P.nch;
  const bool multiroot = W.tr.root_nodes.size() > 1;
  int qqqq = rng.sample(3,1,false)[0] + 1;
  
  MoveResult VV; int commentbis = 0;
  if(qqqq==1){
    if(rng.runif()>0.5 || multiroot){ VV=modificationtopo2(W,Dat,P,Pr,rng);       commentbis=1; }
    else                            { VV=modificationtopofrere(W,Dat,P,Pr,rng);   commentbis=2; }
  } else if(qqqq==2){
    if(rng.runif()>0.5 || multiroot){ VV=modificationtopo3(W,Dat,P,Pr,rng);       commentbis=3; }
    else                            { VV=modificationtopofrere2(W,Dat,P,Pr,rng);  commentbis=4; }
  } else {
    if(P.agemax_fixed){ VV=modificationtopooutgroupsansresc(W,Dat,P,Pr,rng);        commentbis=6; }
    else              { VV=modificationtopooutgroupsansresc_agevar(W,Dat,P,Pr,rng); commentbis=6; }
  }
  
  const double gam = VV.corrb;
  const double alphabis = log_prior_tree(VV.trt,P,Pr) - log_prior_tree(W.tr,P,Pr);
  if(alphabis==-INFINITY || VV.d==-1)                     // rejected proposal
    return (gam!=0.0) ? -6 : 0;
  
  std::vector<std::vector<Mat>> lint(nch);
  if(VV.d==0){                                            // outgroup: full recompute
    for(int x=0;x<nch;++x)
      lint[x]=pruning_full(VV.trt,VV.Xt[x],VV.Mt[x],VV.Lt,Dat[x],P.nph[x],
                           VV.trt.root_nodes,P.loiini[x],VV.loiappt[x]);
  } else {                                                // SPR: eco recompute along path
    auto path1=chemin_rac(VV.trt,VV.d,VV.trt.root_nodes);
    for(int x=0;x<nch;++x)
      lint[x]=pruning_eco(VV.trt,VV.Xt[x],VV.Mt[x],Dat[x],P.nph[x],path1,
                          W.lin[x],VV.Lt,P.loiini[x],VV.loiappt[x]);
  }
  
  const double alph = lkldtopotout(VV.trt,VV.Xt,W.la,VV.NLt,W.rho,VV.Lt,W.P)
    - lkldtopotout(W.tr, W.X, W.la,W.NL, W.rho,W.L, W.P) + alphabis;
  const double bet  = lkldcogn(lint, nch, VV.trt.root_children, VV.trt.root_nodes, ncogn, P.loiini)
    - lkldcogn(W.lin,nch, W.tr.root_children,  W.tr.root_nodes,  ncogn, P.loiini);
  
  double alphaaccept = std::min(0.0, alph + bet + gam);
  if(std::isnan(alphaaccept)) alphaaccept = -INFINITY;
  
  if(std::log(rng.runif()) < alphaaccept){               // accept -> commit into W
    W.tr=VV.trt; W.X=VV.Xt; W.NL=VV.NLt; W.L=VV.Lt; W.Tps=VV.Tpst;
    W.M=VV.Mt; W.loiapp=VV.loiappt; W.lin=lint;
    return commentbis;
  }
  return -commentbis;
}

} // namespace phylo

