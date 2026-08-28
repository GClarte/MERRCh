// src/clades.cpp
#include "phylo/clades.hpp"
#include "phylo/tree.hpp"
#include <algorithm>
namespace phylo {
bool is_monophyletic(const Tree& tr,const std::vector<int>& tip){
  std::vector<std::vector<int>> k; for(int t:tip) k.push_back(ancetres(tr,t));
  std::vector<int> roots; for(auto&v:k) roots.push_back(v[0]);
  std::sort(roots.begin(),roots.end()); roots.erase(std::unique(roots.begin(),roots.end()),roots.end());
  std::vector<int> d;
  if(roots.size()==1){
    size_t mn=k[0].size(); for(auto&v:k) mn=std::min(mn,v.size());
    size_t i=0; while(i<mn){ bool s=true; for(auto&v:k) if(v[i]!=k[0][i]){s=false;break;} if(!s) break; ++i; }
    d=desc(tr,k[0][i-1]);
  } else for(int r:roots){ auto dd=desc(tr,r); d.insert(d.end(),dd.begin(),dd.end()); }
  return d.size()==tip.size();
}
bool age_constraint(const Tree& tr,const std::vector<int>& clade,double lo,double hi){
  int aa=nearest_common_ancestor(tr,clade);
  double l=0; int temp=aa; auto c=tr.children_of(temp);
  while(c[0]>=0){ l+=tr.edge_length[c[0]]; temp=tr.edge[c[0]][1]; c=tr.children_of(temp); }
  return l>lo && l<hi;
}
} // namespace phylo
