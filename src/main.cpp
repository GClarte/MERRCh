#include "phylo/config.hpp"
#include "phylo/csv.hpp"
#include "phylo/smc.hpp"
#include <iostream>
using namespace phylo;

int main(int argc,char** argv){
  try{
    CliOptions cli=parse_cli(argc,argv);
    RawConfig raw=load_config(cli.config);
    DataBundle db=load_data(cli.data_csv,raw.langues,raw.char_cols);
    Param param; Prior prior;
    finalize(raw,db.nph,param,prior);
    std::fprintf(stderr,"nch=%d  ncogn=%ld  nlang=%ld\n",
       (int)db.dat.size(), (long)db.dat[0].rows(), (long)db.dat[0].cols());
    for(size_t c=0;c<db.dat.size();++c){
      long filled=0, total=db.dat[c].size();
      for(int a=0;a<db.dat[c].rows();++a)
        for(int b=0;b<db.dat[c].cols();++b)
          if(db.dat[c](a,b)>=0) ++filled;
      std::fprintf(stderr,"  ch%zu: nph=%d  filled=%ld/%ld (%.1f%%)\n",
         c, db.nph[c], filled, total, 100.0*filled/total);
    }

    Rng rng(cli.seed);
    SMCResult res=SMCbruit(db.dat,param,prior,rng,cli.ncores);
    save_result(res,cli.out);
    std::cout<<"done: "<<res.particles.size()<<" particles -> "<<cli.out<<"\n";
  }catch(const std::exception& e){
    std::cerr<<"error: "<<e.what()<<"\n"; return 1;
  }
  return 0;
}

