// include/phylo/tree.hpp
#pragma once
#include <string>
#include "phylo/types.hpp"
namespace phylo {
std::vector<int> ancetres(const Tree&, int tip);
std::vector<int> desc(const Tree&, int node);
std::vector<int> feuilles_sous_arbre(const Tree&, int node);
std::vector<int> sous_arbre(const Tree&, int node);
double hauteur(const Tree&, int node);
std::vector<int> parcours_depth(const Tree&, int node);
std::pair<std::vector<double>,std::vector<double>>
    hauteur_tous(const Tree&, const std::vector<int>& tips, double agemax);
int nearest_common_ancestor(const Tree&, const std::vector<int>& tips);
std::vector<int> chemin_rac(const Tree&, int node, const std::vector<int>& root);
bool testarbre3(const Tree&, const std::vector<int>& root, int nnodes,
                const std::vector<int>& tips);
std::string to_newick(const Tree&);
}
