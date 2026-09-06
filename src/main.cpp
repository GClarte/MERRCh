#include "phylo/config.hpp"
#include "phylo/csv.hpp"
#include "phylo/smc.hpp"
#include <iostream>
#include <ctime>
#include <fstream>
#include <random>
#include <stdexcept>
using namespace phylo;

static std::string dated_base(const std::string& name){
  std::time_t t=std::time(nullptr);
  std::tm tm=*std::localtime(&t);
  char buf[32];
  std::strftime(buf,sizeof(buf),"%Y-%m-%d_%H-%M-%S",&tm);
  return name+"_"+buf;
}

static uint64_t make_seed(CliOptions& cli){
  if(cli.seed_set) return cli.seed;
  std::random_device rd;
  uint64_t s=(static_cast<uint64_t>(rd())<<32)|rd();
  cli.seed=s;
  return s;
}

static void copy_config(const std::string& src,const std::string& dst,
                        uint64_t seed,bool seed_was_random){
  std::ifstream in(src,std::ios::binary);
  if(!in) throw std::runtime_error("cannot read config: "+src);
  std::ofstream out(dst,std::ios::binary);
  if(!out) throw std::runtime_error("cannot write config copy: "+dst);
  out<<in.rdbuf();
  out<<"\n# ---- added by run ----\n";
  out<<"seed = "<<seed;
  if(seed_was_random) out<<"   # auto-generated (not user-supplied)";
  out<<"\n";
}

int main(int argc,char** argv){
  try{
    CliOptions cli=parse_cli(argc,argv);
    RawConfig raw=load_config(cli.config);
    DataBundle db=load_data(cli.data_csv,raw.langues,raw.char_cols);
    Param param; Prior prior;
    finalize(raw,db.nph,param,prior);

    uint64_t seed=make_seed(cli);
    std::fprintf(stderr,"[seed] %llu%s\n",(unsigned long long)seed,
                 cli.seed_set?" (user-supplied)":" (random -- pass --seed to reproduce)");

    Rng rng(seed);
    SMCResult res=SMCbruit(db.dat,param,prior,rng,cli.ncores);

    std::string base=dated_base(cli.name);
    save_result(res,param,base,cli.additional_results);
    copy_config(cli.config,base+"_cfg.txt",seed,!cli.seed_set);

    std::cout<<"done: "<<res.particles.size()<<" particles -> "<<base<<"*\n";
    if(cli.additional_results)
      std::cout<<"additional results written (.changes .transfprob .annot.nex)\n";
  }catch(const std::exception& e){
    std::cerr<<"error: "<<e.what()<<"\n"; return 1;
  }
  return 0;
}

