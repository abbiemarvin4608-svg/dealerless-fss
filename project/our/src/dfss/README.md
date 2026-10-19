# Paper-Owned dFSS Sources

The files here contain the correlated-DPF changes used by this paper,
including DPF-based DCF evaluation, keyed correlated hashing, Beaver-backed
control-word generation, batched COT support, and the length-reduced
Gilboa-COT Beaver triple generation. They mirror the reference
`dealerless-FSS_public/src` layout (`utils/net_io_channel.h` overrides the
EzPC SCI header). `project/our/CMakeLists.txt` copies these files over the
reference library's generated build overlay, so the public checkout can
remain an unmodified dependency.
