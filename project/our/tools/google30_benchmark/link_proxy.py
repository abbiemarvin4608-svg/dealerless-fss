#!/usr/bin/env python3
"""Real TCP relay with modeled one-way propagation and shared full-duplex rate caps.

This is an application-payload link model, not tc, TCP congestion, or an Internet route.
Each direction shares one serialization clock across all configured SCI sockets.
"""
import argparse
import heapq
import selectors
import signal
import socket
import time

class Relay:
    def __init__(self, mappings, rtt_ms, mbps):
        self.selector=selectors.DefaultSelector(); self.delay=rtt_ms/2000
        self.rate=mbps*1000000/8; self.free=[0.0,0.0]; self.queue=[]; self.serial=0
        self.buffers={}; self.peer={}; self.eof=set(); self.pending={}; self.running=True; self.connecting={}
        for listen,target in mappings:
            sock=socket.socket(); sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            sock.bind(('127.0.0.1',listen)); sock.listen(); sock.setblocking(False)
            self.selector.register(sock,selectors.EVENT_READ,('listen',target))
    def stop(self,*_): self.running=False
    def close(self):
        for key in list(self.selector.get_map().values()): key.fileobj.close()
        self.selector.close()
    def update(self,sock):
        if sock.fileno()<0 or sock in self.connecting:return
        events=(0 if sock in self.eof else selectors.EVENT_READ)
        if self.buffers[sock]:events|=selectors.EVENT_WRITE
        try:self.selector.unregister(sock)
        except KeyError:pass
        if events:self.selector.register(sock,events,('stream',None))
    def run(self):
        while self.running:
            for backend,(target,deadline) in list(self.connecting.items()):
                import errno
                code=backend.connect_ex(('127.0.0.1',target))
                if code in (0,errno.EISCONN):
                    del self.connecting[backend]; self.update(backend)
                elif code not in (errno.EINPROGRESS,errno.EALREADY,errno.EWOULDBLOCK):
                    # A refused nonblocking socket must be recreated; keep queued bytes/peer identity.
                    client,direction=self.peer.pop(backend); data=self.buffers.pop(backend); pending=self.pending.pop(backend)
                    backend.close(); del self.connecting[backend]
                    replacement=socket.socket(); replacement.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1);replacement.setblocking(False)
                    self.peer[replacement]=(client,direction);self.peer[client]=(replacement,1-direction)
                    self.buffers[replacement]=data;self.pending[replacement]=pending
                    self.queue=[(due,seq,replacement if dst is backend else dst,data) for due,seq,dst,data in self.queue];heapq.heapify(self.queue)
                    if time.monotonic()>deadline:raise TimeoutError('backend listener unavailable: '+str(target))
                    self.connecting[replacement]=(target,deadline)
            now=time.monotonic()
            while self.queue and self.queue[0][0]<=now:
                _,_,target,data=heapq.heappop(self.queue)
                self.pending[target]-=1;self.buffers[target].extend(data);self.update(target)
            timeout=min(.001 if self.connecting else .1,max(0,self.queue[0][0]-now)) if self.queue else (.001 if self.connecting else .1)
            for key,mask in self.selector.select(timeout):
                sock=key.fileobj; kind,arg=key.data
                if kind=='listen':
                    client,_=sock.accept(); backend=socket.socket()
                    backend.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1); backend.setblocking(False)
                    client.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1); client.setblocking(False)
                    # Delayed listeners occur in dFSS: raw sockets finish before its OT listeners start.
                    # Connect asynchronously so waiting for the next listener never stalls live channels.
                    for conn in [client,backend]:
                        self.buffers[conn]=bytearray(); self.pending[conn]=0
                    self.peer[client]=(backend,0); self.peer[backend]=(client,1)
                    self.update(client)
                    self.connecting[backend]=(arg,time.monotonic()+10)
                    continue
                if mask&selectors.EVENT_READ:
                    try:data=sock.recv(65536)
                    except BlockingIOError:data=None
                    if data:
                        target,direction=self.peer[sock];now=time.monotonic()
                        finish=max(now,self.free[direction])+len(data)/self.rate
                        self.free[direction]=finish;self.serial+=1;self.pending[target]+=1
                        heapq.heappush(self.queue,(finish+self.delay,self.serial,target,data))
                    elif data==b'':self.eof.add(sock);self.update(sock)
                if mask&selectors.EVENT_WRITE:
                    try:n=sock.send(self.buffers[sock]);del self.buffers[sock][:n]
                    except BlockingIOError:pass
                    self.update(sock)
            for sock in list(self.eof):
                target,_=self.peer[sock]
                if not self.pending[target] and not self.buffers[target]:
                    try:target.shutdown(socket.SHUT_WR)
                    except OSError:pass
            # Finished channels are retained until stop, so counters/lifecycle cannot drop in-flight bytes.

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ports',required=True)
    p.add_argument('--rtt-ms',type=float,required=True);p.add_argument('--mbps',type=float,required=True)
    a=p.parse_args()
    if a.rtt_ms<0 or a.mbps<=0:p.error('nonnegative delay and positive rate required')
    relay=Relay([tuple(map(int,pair.split(':'))) for pair in a.ports.split(',')],a.rtt_ms,a.mbps)
    signal.signal(signal.SIGTERM,relay.stop);signal.signal(signal.SIGINT,relay.stop)
    print('READY',flush=True)
    try:relay.run()
    finally:relay.close()
