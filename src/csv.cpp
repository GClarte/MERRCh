// src/csv.cpp
#include "phylo/csv.hpp"
#include <algorithm>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
namespace phylo {

static std::vector<std::string> split_csv(const std::string& line){
  std::vector<std::string> out; std::string cur; bool q=false;
  for(size_t i=0;i<line.size();++i){ char ch=line[i];
    if(q){ if(ch=='"'){ if(i+1<line.size()&&line[i+1]=='"'){cur+='"';++i;} else q=false; } else cur+=ch; }
    else { if(ch=='"') q=true; else if(ch==','){ out.push_back(cur); cur.clear(); } else cur+=ch; }
  }
  out.push_back(cur); return out;
}
static std::string trim(const std::string& s){
  size_t a=s.find_first_not_of(" \t\r\n"); if(a==std::string::npos) return "";
  size_t b=s.find_last_not_of(" \t\r\n"); return s.substr(a,b-a+1);
}

DataBundle load_data(const std::string& path,
                     const std::vector<std::string>& langues,
                     const std::vector<int>& char_cols){   // empty => auto
  std::ifstream f(path,std::ios::binary);
  if(!f) throw std::runtime_error("cannot open "+path);
  const int nlang=(int)langues.size();
  std::map<std::string,int> lang_idx;
  for(int i=0;i<nlang;++i) lang_idx[langues[i]]=i;

  std::vector<std::vector<std::string>> rows;
  int ncol=-1; std::string line; bool first=true;
  while(std::getline(f,line)){
    if(first){ if(line.size()>=3 && (unsigned char)line[0]==0xEF) line=line.substr(3);
               first=false; continue; }                 // skip header + BOM
    if(trim(line).empty()) continue;
    auto cells=split_csv(line);
    if(ncol==-1) ncol=(int)cells.size();
    rows.push_back(std::move(cells));
  }
  if(rows.empty()) throw std::runtime_error("no data rows in "+path);
  if(ncol<3) throw std::runtime_error("need >=3 columns: language,meaning,character...");

  // ---- resolve which columns are characters (1-based) ----
  std::vector<int> cols;
  if(char_cols.empty()){
    for(int c=3;c<=ncol;++c) cols.push_back(c);          // default: all after first two
  } else {
    for(int c:char_cols){
      if(c<1||c>ncol) throw std::runtime_error("char_cols index "+std::to_string(c)
                          +" out of range (file has "+std::to_string(ncol)+" columns)");
      if(c==1||c==2) throw std::runtime_error("char_cols cannot include column 1 (language)"
                          " or 2 (meaning)");
      cols.push_back(c);
    }
  }
  const int nch=(int)cols.size();
  if(nch==0) throw std::runtime_error("no character columns selected");

  // ---- distinct meanings -> row index (string-keyed, sorted ascending) ----  [CHANGED]
  std::set<std::string> mset;
  for(auto& r:rows)
    if(r.size()>=2){ std::string m=trim(r[1]); if(!m.empty()) mset.insert(m); }
  std::vector<std::string> meanings(mset.begin(),mset.end());
  const int ncogn=(int)meanings.size();
  if(ncogn==0) throw std::runtime_error("no meanings found in column 2 of "+path);
  std::map<std::string,int> mean_idx;
  for(int i=0;i<ncogn;++i) mean_idx[meanings[i]]=i;

  // factor levels per selected character column (byte-wise; UTF-8 safe)
  std::vector<std::map<std::string,int>> levels(nch);
  std::vector<int> nph(nch,0);
  for(int c=0;c<nch;++c){
    int col=cols[c]-1;                                   // 1-based -> 0-based
    std::set<std::string> uniq;
    for(auto& r:rows) if(col<(int)r.size()){ std::string v=trim(r[col]); if(!v.empty()) uniq.insert(v); }
    int lv=1; for(auto& v:uniq) levels[c][v]=lv++;
    nph[c]=(int)uniq.size();
  }

  Data dat(nch);
  for(int c=0;c<nch;++c) dat[c]=Eigen::MatrixXi::Constant(ncogn,nlang,-1);

  for(auto& r:rows){
    if((int)r.size()<ncol) continue;
    auto li=lang_idx.find(trim(r[0]));  if(li==lang_idx.end()) continue;
    std::string mv=trim(r[1]);          if(mv.empty()) continue;   // [CHANGED]
    auto mi=mean_idx.find(mv);          if(mi==mean_idx.end()) continue;
    for(int c=0;c<nch;++c){
      int col=cols[c]-1;
      std::string v=trim(r[col]); if(v.empty()) continue;
      auto it=levels[c].find(v); if(it!=levels[c].end()) dat[c](mi->second,li->second)=it->second;
    }
  }
  return { std::move(dat),std::move(nph) };
}


} // namespace phylo

