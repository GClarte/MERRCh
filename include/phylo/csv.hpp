// include/phylo/csv.hpp
#pragma once
#include <string>
#include <vector>
#include "phylo/types.hpp"
namespace phylo {
struct DataBundle { Data dat; std::vector<int> nph; };

// Long format: col1 = language, col2 = meaning, cols 3.. = characters.
// char_cols is OPTIONAL: if empty, all columns after the first two are used;
// otherwise it selects/reorders 1-based character columns (must be >= 3).
DataBundle load_data(const std::string& csv_path,
                     const std::vector<std::string>& langues,
                     const std::vector<int>& char_cols = {});
}

