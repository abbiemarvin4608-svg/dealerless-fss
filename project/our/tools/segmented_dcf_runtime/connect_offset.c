#define _GNU_SOURCE
#include <dlfcn.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>

int connect(int fd, const struct sockaddr *address, socklen_t length) {
    static int (*real_connect)(int, const struct sockaddr *, socklen_t);
    if (!real_connect) real_connect = dlsym(RTLD_NEXT, "connect");
    const char *offset = getenv("SWEEP_CONNECT_OFFSET");
    if (offset && address->sa_family == AF_INET && length >= sizeof(struct sockaddr_in)) {
        struct sockaddr_in redirected;
        memcpy(&redirected, address, sizeof redirected);
        redirected.sin_port = htons(ntohs(redirected.sin_port) + atoi(offset));
        return real_connect(fd, (const struct sockaddr *)&redirected, sizeof redirected);
    }
    return real_connect(fd, address, length);
}
