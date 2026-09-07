// src/config.cpp
#include "phylo/config.hpp"
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
namespace phylo {

static std::string trim(const std::string& s){
  size_t a=s.find_first_not_of(" \t\r\n"); if(a==std::string::npos) return "";
  size_t b=s.find_last_not_of(" \t\r\n"); return s.substr(a,b-a+1);
}
static std::vector<std::string> split(const std::string& s,char d){
  std::vector<std::string> o; std::stringstream ss(s); std::string t;
  while(std::getline(ss,t,d)) o.push_back(trim(t));
  return o;
}
static std::vector<int> parse_ints(const std::string& s){
  std::vector<int> o;
  for(auto& tok:split(s,',')){
    if(tok.empty()) continue;
    auto colon=tok.find(':'), dash=tok.find('-');
    if(colon!=std::string::npos){ int a=std::stoi(tok.substr(0,colon)),b=std::stoi(tok.substr(colon+1));
      for(int i=a;i<=b;++i) o.push_back(i); }
    else if(dash!=std::string::npos){ int a=std::stoi(tok.substr(0,dash)),b=std::stoi(tok.substr(dash+1));
      for(int i=a;i<=b;++i) o.push_back(i); }
    else o.push_back(std::stoi(tok));
  }
  return o;
}
static std::vector<double> parse_dbls(const std::string& s){
  std::vector<double> o; for(auto& t:split(s,',')) if(!t.empty()) o.push_back(std::stod(t)); return o;
}

CliOptions parse_cli(int argc,char** argv){
  CliOptions o;
  for(int i=1;i<argc;++i){ std::string a=argv[i];
    auto next=[&](){ return std::string(argv[++i]); };
if     (a=="--data")               o.data_csv=next();
else if(a=="--config")             o.config=next();
else if(a=="--name")               o.name=next();
else if(a=="--ncores")             o.ncores=std::stoi(next());
else if(a=="--seed"){              o.seed=std::stoull(next()); o.seed_set=true; }
else if(a=="--additional_results") o.additional_results=true;


  }
  return o;
}

RawConfig load_config(const std::string& path){
  std::ifstream f(path); if(!f) throw std::runtime_error("cannot open config "+path);
  RawConfig c; std::string raw,line;
  while(std::getline(f,line)){
    while(!line.empty() && line.back()=='\\'){ line.pop_back(); std::string nx; std::getline(f,nx); line+=nx; }
    std::string s=trim(line);
    if(s.empty()||s[0]=='#') continue;
    auto eq=s.find('='); if(eq==std::string::npos) continue;
    std::string k=trim(s.substr(0,eq)), v=trim(s.substr(eq+1));
    auto hash=v.find('#'); if(hash!=std::string::npos) v=trim(v.substr(0,hash));
    if(k=="char_cols") c.char_cols=parse_ints(v);
    else if(k=="meanings") c.meanings=parse_ints(v);
    else if(k=="langues") c.langues=split(v,',');
    else if(k=="npart") c.npart=std::stoi(v);
    else if(k=="npas") c.npas=std::stoi(v);
    else if(k=="npasfin") c.npasfin=std::stoi(v);
    else if(k=="Nmin") c.Nmin=std::stoi(v);
    else if(k=="prob") c.prob=parse_dbls(v);
    else if(k=="probfin") c.probfin=parse_dbls(v);
    else if(k=="bruittemp") c.bruittemp=parse_dbls(v);
    else if(k=="agemax"){ auto d=parse_dbls(v); if(d.size()==1){c.agemax_fixed=true;c.agemax={d[0],0};}
                          else c.agemax={d[0],d[1]}; }
    else if(k=="prirho"){ auto d=parse_dbls(v); c.prirho={d[0],d[1]}; }
    else if(k=="prila"){ auto d=parse_dbls(v); c.prila={d[0],d[1],d[2],d[3]}; }
    else if(k=="pribeta"){ auto d=parse_dbls(v); c.pribeta={d[0],d[1]}; }
    else if(k=="treeprior") c.treeprior=v;
    else if(k=="height_lo") c.height_lo=std::stod(v);
    else if(k=="height_hi") c.height_hi=std::stod(v);
    else if(k=="language_col") c.lang_col=std::stoi(v);
    else if(k=="meaning_col") c.meaning_col=std::stoi(v);
    else if(k=="tipprior"){ for(auto& g:split(v,';')) c.tipprior.push_back(parse_ints(g)); }
    else if(k=="cladeage"){
      for(auto& g:split(v,';')){ auto lr=split(g,':'); if(lr.size()!=2) continue;
        auto tips=parse_ints(lr[0]); auto ages=parse_dbls(lr[1]);
        c.cladeage.push_back({tips,{ages[0],ages.size()>1?ages[1]:INFINITY}}); }
    }
  }
  return c;
}

static std::vector<std::array<int,2>> toutestransf(int n){
  std::vector<std::array<int,2>> a;
  for(int i=0;i<n;++i) for(int j=0;j<n;++j) if(i!=j) a.push_back({i,j});
  return a;
}

void finalize(const RawConfig& c,const std::vector<int>& nph,Param& P,Prior& Pr){
  int nch=(int)nph.size();
  P.nch=nch; P.nph=nph;
  P.npart=c.npart; P.npas=c.npas; P.npasfin=c.npasfin; P.Nmin=c.Nmin;
  P.Prob=c.prob; P.Probfin=c.probfin;
  P.agemax_fixed=c.agemax_fixed;
  P.agemax=Eigen::Vector2d(c.agemax[0],c.agemax[1]);
  P.agemax_value = c.agemax_fixed ? c.agemax[0] : 0.0;
  P.tiplabel=c.langues;
  P.Cladeage=c.cladeage;
  P.priortree = c.treeprior=="yule"?TreePrior::Yule:(c.treeprior=="coal"?TreePrior::Coal:TreePrior::Unif);
  P.height_lo=c.height_lo; P.height_hi=c.height_hi;
  P.passages.resize(nch); for(int i=0;i<nch;++i) P.passages[i]=toutestransf(nph[i]);
  P.loiini.resize(nch); for(int i=0;i<nch;++i) P.loiini[i]=Vec::Constant(nph[i],1.0/nph[i]);
  P.Prila.assign(nch,c.prila);

  Pr.Prirho=Eigen::Vector2d(c.prirho[0],c.prirho[1]);
  Pr.hyperpbini.resize(nch);
  for(int i=0;i<nch;++i) Pr.hyperpbini[i]=Vec::Ones(P.passages[i].size());
  Pr.pribeta=Eigen::MatrixXd(2,nch);
  for(int i=0;i<nch;++i){ Pr.pribeta(0,i)=c.pribeta[0]; Pr.pribeta(1,i)=c.pribeta[1]; }
  Pr.bruittemp=c.bruittemp;
  Pr.tipprior=c.tipprior;
}
} // namespace phylo
