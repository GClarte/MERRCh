// src/tree.cpp
#include "phylo/tree.hpp"
#include <algorithm>
#include <sstream>
namespace phylo {

int Tree::max_node() const { int m=0; for(auto&e:edge) m=std::max({m,e[0],e[1]}); return m; }
int Tree::branch_to(int node) const {
  for(int b=0;b<n_edges();++b) if(edge[b][1]==node) return b;
  return -1;
}
std::array<int,2> Tree::children_of(int node) const {
  std::array<int,2> r{-1,-1}; int k=0;
  for(int b=0;b<n_edges();++b) if(edge[b][0]==node && k<2) r[k++]=b;
  return r;
}
int Tree::parent_of(int node) const { int b=branch_to(node); return b<0?-1:edge[b][0]; }

std::vector<int> ancetres(const Tree& tr,int x){
  std::vector<int> u{x}; int i=x; const auto& rr=tr.root_nodes;
  while(std::find(rr.begin(),rr.end(),i)==rr.end()){
    int b=tr.branch_to(i); i=tr.edge[b][0]; u.insert(u.begin(),i);
  }
  return u;
}
std::vector<int> desc(const Tree& tr,int i){
  auto c=tr.children_of(i);
  if(c[0]<0) return {i};
  auto l=desc(tr,tr.edge[c[0]][1]); auto r=desc(tr,tr.edge[c[1]][1]);
  l.insert(l.end(),r.begin(),r.end()); return l;
}
std::vector<int> feuilles_sous_arbre(const Tree& tr,int i){ return desc(tr,i); }
std::vector<int> sous_arbre(const Tree& tr,int i){
  auto c=tr.children_of(i);
  if(c[0]<0) return {};
  std::vector<int> out{c[0],c[1]};
  auto a=sous_arbre(tr,tr.edge[c[0]][1]); auto b=sous_arbre(tr,tr.edge[c[1]][1]);
  out.insert(out.end(),a.begin(),a.end()); out.insert(out.end(),b.begin(),b.end());
  return out;
}
double hauteur(const Tree& tr,int i){
  auto c=tr.children_of(i);
  if(c[0]<0) return 0.0;
  return tr.edge_length[c[0]]+hauteur(tr,tr.edge[c[0]][1]);
}
std::vector<int> parcours_depth(const Tree& tr,int j){
  auto c=tr.children_of(j);
  if(c[0]<0) return {j};
  auto a=parcours_depth(tr,tr.edge[c[0]][1]);
  auto b=parcours_depth(tr,tr.edge[c[1]][1]);
  a.insert(a.end(),b.begin(),b.end()); a.push_back(j); return a;
}
std::pair<std::vector<double>,std::vector<double>>
hauteur_tous(const Tree& tr,const std::vector<int>& tips,double agemax){
  int N=tr.max_node();
  std::vector<double> h(N+1,0.0), h2(N+1,0.0); int act=-1;
  for(int t:tips){ act=t; int b;
    while((b=tr.branch_to(act))>=0){ int up=tr.edge[b][0]; h[up]=tr.edge_length[b]+h[act]; h2[act]=h[up]; act=up; } }
  if(act>=0) h2[act]=agemax;
  return {h,h2};
}
int nearest_common_ancestor(const Tree& tr,const std::vector<int>& tip){
  if(tip.size()>1){
    std::vector<std::vector<int>> k; for(int t:tip) k.push_back(ancetres(tr,t));
    size_t mn=k[0].size(); for(auto&v:k) mn=std::min(mn,v.size());
    size_t i=0; while(i<mn){ bool same=true; for(auto&v:k) if(v[i]!=k[0][i]){same=false;break;}
      if(!same) break; ++i; }
    return k[0][i-1];
  }
  return tr.edge[tr.branch_to(tip[0])][0];
}
std::vector<int> chemin_rac(const Tree& tr,int i,const std::vector<int>& root){
  std::vector<int> out;
  while(std::find(root.begin(),root.end(),i)==root.end()){ out.push_back(i); i=tr.parent_of(i); }
  out.push_back(i); return out;
}
bool testarbre3(const Tree& tr,const std::vector<int>& root,int nnodes,const std::vector<int>&){
  std::vector<int> qui=root, actif=root;
  for(int i=0;i<nnodes/2;++i){
    std::vector<int> nxt;
    for(int x:actif){ auto c=tr.children_of(x);
      if(c[0]>=0){ nxt.push_back(tr.edge[c[0]][1]); nxt.push_back(tr.edge[c[1]][1]); } }
    actif.swap(nxt); qui.insert(qui.end(),actif.begin(),actif.end());
  }
  if((int)qui.size()!=nnodes) return false;
  std::sort(qui.begin(),qui.end());
  for(int i=0;i<nnodes;++i) if(qui[i]!=i+1) return false;
  for(double l:tr.edge_length) if(l<=0) return false;
  return true;
}
static void newick_rec(const Tree& tr,int node,std::ostringstream& os){
  auto c=tr.children_of(node);
  if(c[0]<0){ os<<tr.tip_label[node-1]; return; }
  os<<"("; newick_rec(tr,tr.edge[c[0]][1],os); os<<":"<<tr.edge_length[c[0]]<<",";
  newick_rec(tr,tr.edge[c[1]][1],os); os<<":"<<tr.edge_length[c[1]]<<")";
}
std::string to_newick(const Tree& tr){
  std::ostringstream os; newick_rec(tr,tr.root_nodes[0],os); os<<";"; return os.str();
}
} // namespace phylo
