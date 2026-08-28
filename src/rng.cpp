#include "phylo/rng.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace phylo {

double Rng::runif(double a, double b){ std::uniform_real_distribution<> d(a,b); return d(gen_); }
double Rng::rexp(double rate){ std::exponential_distribution<> d(rate); return d(gen_); }
double Rng::rnorm(double m,double s){ std::normal_distribution<> d(m,s); return d(gen_); }
double Rng::rgamma(double sh,double rt){ std::gamma_distribution<> d(sh,1.0/rt); return d(gen_); }
int    Rng::rpois(double l){ std::poisson_distribution<> d(l); return d(gen_); }
int    Rng::rbinom(int n,double p){ if(n<=0) return 0; std::binomial_distribution<> d(n,p); return d(gen_); }
double Rng::rbeta(double a,double b){ double x=rgamma(a,1.0), y=rgamma(b,1.0); return x/(x+y); }

std::vector<int> Rng::sample(int n,int k,bool replace,const std::vector<double>* w){
  std::vector<int> out; out.reserve(k);
  if(w){
    if(replace){
      std::discrete_distribution<> d(w->begin(), w->end());
      for(int i=0;i<k;++i) out.push_back(d(gen_));
      return out;
    }
    std::vector<std::pair<double,int>> key(n);
    for(int i=0;i<n;++i){ double u=runif(); key[i]={std::pow(u,1.0/std::max((*w)[i],1e-300)),i}; }
    std::partial_sort(key.begin(),key.begin()+k,key.end(),std::greater<>());
    for(int i=0;i<k;++i) out.push_back(key[i].second);
    return out;
  }
  std::vector<int> idx(n); std::iota(idx.begin(),idx.end(),0);
  if(replace){ std::uniform_int_distribution<> d(0,n-1); for(int i=0;i<k;++i) out.push_back(idx[d(gen_)]); }
  else { std::shuffle(idx.begin(),idx.end(),gen_); out.assign(idx.begin(),idx.begin()+k); }
  return out;
}

double Rng::dgamma(double x,double sh,double rt){
  if(x<=0) return -INFINITY;
  return sh*std::log(rt)+(sh-1)*std::log(x)-rt*x-std::lgamma(sh);
}
double Rng::dbeta(double x,double a,double b){
  if(x<=0||x>=1) return -INFINITY;
  return (a-1)*std::log(x)+(b-1)*std::log1p(-x)+std::lgamma(a+b)-std::lgamma(a)-std::lgamma(b);
}
double Rng::dpois(int k,double l){
  if(l<=0) return k==0?0.0:-INFINITY;
  return k*std::log(l)-l-std::lgamma(k+1.0);
}
double Rng::dbinom(int k,int n,double p){
  if(k<0||k>n) return -INFINITY;
  if(p<=0) return k==0?0.0:-INFINITY;
  if(p>=1) return k==n?0.0:-INFINITY;
  return std::lgamma(n+1.0)-std::lgamma(k+1.0)-std::lgamma(n-k+1.0)+k*std::log(p)+(n-k)*std::log1p(-p);
}
double Rng::dnorm(double x,double m,double s){
  double z=(x-m)/s; return -0.5*z*z-std::log(s)-0.5*std::log(2*M_PI);
}

static double gammp(double a,double x){
  if(x<=0) return 0.0;
  if(x<a+1){
    double ap=a,sum=1.0/a,del=sum;
    for(int n=0;n<300;++n){ ap+=1; del*=x/ap; sum+=del; if(std::fabs(del)<std::fabs(sum)*1e-15) break; }
    return sum*std::exp(-x+a*std::log(x)-std::lgamma(a));
  }
  double b=x+1-a,c=1e300,d=1.0/b,h=d;
  for(int i=1;i<=300;++i){
    double an=-i*(i-a); b+=2; d=an*d+b; if(std::fabs(d)<1e-300) d=1e-300;
    c=b+an/c; if(std::fabs(c)<1e-300) c=1e-300; d=1.0/d; double del=d*c; h*=del;
    if(std::fabs(del-1)<1e-15) break;
  }
  return 1.0-std::exp(-x+a*std::log(x)-std::lgamma(a))*h;
}
double Rng::pgamma(double x,double sh,double rt){ return gammp(sh,rt*x); }
double Rng::qgamma(double p,double sh,double rt){
  if(p<=0) return 0.0;
  double lo=0,hi=1; while(pgamma(hi,sh,rt)<p && hi<1e12) hi*=2;
  for(int it=0;it<200;++it){ double m=0.5*(lo+hi); (pgamma(m,sh,rt)<p?lo:hi)=m; }
  return 0.5*(lo+hi);
}
double Rng::pnorm(double x,double m,double s){ return 0.5*std::erfc(-(x-m)/(s*M_SQRT2)); }

Vec rdirichlet(Rng& rng,const Vec& a){
  Vec x(a.size());
  for(int i=0;i<a.size();++i) x[i]=rng.rgamma(std::max(a[i],1e-9),1.0);
  return x/x.sum();
}

std::pair<Tree,double> rcoal(Rng& rng,int n,const std::vector<std::string>& labels){
  Tree tr; tr.tip_label = labels.empty() ? std::vector<std::string>() : labels;
  if(tr.tip_label.empty()) for(int i=1;i<=n;++i) tr.tip_label.push_back("t"+std::to_string(i));
  double corr=0.0;
  std::vector<double> h(2*n, 0.0);
  std::vector<int> pool(n); std::iota(pool.begin(),pool.end(),1);
  int nextnode = 2*n-1;
  double height=0.0;
  for(int i=0;i<n-1;++i){
    int m = static_cast<int>(pool.size());
    double x = rng.rexp(static_cast<double>(m)*(m-1)/2.0);
    corr += Rng::dgamma(x, 1.0, static_cast<double>(m)*(m-1)/2.0); // ~ dexp
    height += x;
    auto pick = rng.sample(m,2,false);
    corr += std::log(1.0/m);
    int a=pool[pick[0]], b=pool[pick[1]];
    tr.edge.push_back({nextnode,a}); tr.edge_length.push_back(height-h[a]);
    tr.edge.push_back({nextnode,b}); tr.edge_length.push_back(height-h[b]);
    h[nextnode]=height;
    std::vector<int> np; for(int k=0;k<m;++k) if(k!=pick[0]&&k!=pick[1]) np.push_back(pool[k]);
    np.push_back(nextnode); pool.swap(np); --nextnode;
  }
  tr.Nnode = n-1;
  tr.root_nodes = {n+1};
  for(auto& e:tr.edge) if(e[0]==n+1) tr.root_children.push_back(e[1]);
  return {tr,corr};
}

} // namespace phylo
