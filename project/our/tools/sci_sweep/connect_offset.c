#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <string.h>
#include <sys/socket.h>
int connect(int fd,const struct sockaddr *addr,socklen_t len){
 static int (*real_connect)(int,const struct sockaddr*,socklen_t);
 if(!real_connect)real_connect=dlsym(RTLD_NEXT,"connect");
 const char* delta=getenv("SWEEP_CONNECT_OFFSET");
 if(delta&&addr->sa_family==AF_INET&&len>=sizeof(struct sockaddr_in)){
  struct sockaddr_in copy;memcpy(&copy,addr,sizeof copy);copy.sin_port=htons(ntohs(copy.sin_port)+atoi(delta));
  return real_connect(fd,(const struct sockaddr*)&copy,sizeof copy);
 }
 return real_connect(fd,addr,len);
}
