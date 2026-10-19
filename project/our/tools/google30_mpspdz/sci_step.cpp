// Validation harness for the frozen SCI integer graph; not a timing baseline.
#include "library_fixed.h"
#include <fstream>
#include <vector>
#include <stdexcept>
using namespace std;
int party=1,port=19580,num_threads=1;
int32_t bitlength=32;
string address="127.0.0.1";
extern void reconstruct(int64_t*,int64_t*,int32_t,int32_t,int);
void show(string name,vector<int64_t>& a,int bits=16){
 vector<int64_t> rec(a.size());reconstruct(a.data(),rec.data(),1,a.size(),bits);
 if(party==2)for(size_t i=0;i<a.size();++i)cout<<"VALUE "<<name<<" "<<i<<" "<<rec[i]<<"\n";
}
void read_private(ifstream& in,vector<int64_t>& v,int owner){
 if(party==owner)for(auto& x:v)if(!(in>>x))throw runtime_error("private input ended");
}
int main(int argc,char**argv){
 if(argc!=6)throw runtime_error("usage: sci_step party port input-prefix historical|independent cases");
 party=stoi(argv[1]);port=stoi(argv[2]);bool old=string(argv[4])=="historical";int cases=stoi(argv[5]);
 ifstream input(string(argv[3])+"-P"+to_string(party-1)+"-0");
 vector<int64_t>W1(512),W2(1600),U1(3500),U2(3500),Bg(100),Bh(100);
 for(auto*v:{&W1,&W2,&U1,&U2,&Bg,&Bh})read_private(input,*v,1);
 initialize();
 for(int cse=0;cse<cases;++cse){
  vector<int64_t>x(32),h(100),a(16),b(100),c(35),d(100),e(100),f(100),g(100),hx(100),candidate(100),gh(100),ng(100),zg(100),zn(100),prod(100),out(100),tmp(100);
  read_private(input,x,2);read_private(input,h,2);
  MatMul(1,32,16,1,2,5,0,1,16,8,32,16,x.data(),W1.data(),a.data(),tmp.data());show("tmp19",a);
  MatMul(1,16,100,old?2:32,old?4:64,4,0,1,16,old?8:16,32,16,a.data(),W2.data(),b.data(),tmp.data());show("tmp21",b);
  MatMul(1,100,35,1,1,6,1,256,16,8,32,8,h.data(),U1.data(),c.data(),tmp.data());show("tmp23",c,8);
  MatMul(1,35,100,1,1,old?0:6,old?6:0,1,8,old?8:16,old?16:32,16,c.data(),U2.data(),d.data(),tmp.data());show("tmp25",d);
  MatAdd(1,100,1,old?2:8,1,1,16,16,16,16,b.data(),d.data(),e.data());show("tmp26",e);
  MatAdd(1,100,old?32:1,old?1:8,1,1,16,old?8:16,16,16,e.data(),Bg.data(),f.data());
  if(old)AdjustScaleShl(1,100,32,f.data());show("tmp27",f);
  Sigmoid(1,100,128,16384,16,16,f.data(),g.data());show("tmp28",g);
  MatAdd(1,100,1,8,1,1,16,16,16,16,e.data(),Bh.data(),hx.data());show("tmp29",hx);
  TanH(1,100,128,128,16,16,hx.data(),candidate.data());show("tmp30",candidate);
  MulCir(1,100,128,128,1,16,16,32,16,g.data(),h.data(),gh.data());show("tmp31",gh);
  MatSubBroadCastA(1,100,1,1,1,1,16,16,16,16,16384,g.data(),ng.data());show("tmp33",ng);
  ScalarMul(1,100,128,128,1,16,16,32,16,16654,ng.data(),zg.data());show("tmp34",zg);
  MatAddBroadCastB(1,100,1,64,1,1,16,16,16,16,zg.data(),26006,zn.data());show("tmp35",zn);
  MulCir(1,100,8,16,1,16,16,32,16,zn.data(),candidate.data(),prod.data());show("tmp36",prod);
  MatAdd(1,100,1,2,1,1,16,16,16,16,gh.data(),prod.data(),out.data());show("H",out);
 }
 finalize();
}
