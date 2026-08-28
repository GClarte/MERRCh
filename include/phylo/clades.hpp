// include/phylo/clades.hpp
#pragma once
#include "phylo/types.hpp"
namespace phylo {
bool is_monophyletic(const Tree&, const std::vector<int>& tips);
bool age_constraint(const Tree&, const std::vector<int>& clade_tips, double lo, double hi);
}
