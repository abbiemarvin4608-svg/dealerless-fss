# WSL 中断与内存保护

## 2026-10-15 核查

Windows 系统日志记录了北京时间 14:47:59 和 15:36:16 的 User32 1074 重启事件，均由开始菜单进程触发，随后出现正常事件日志停止和启动记录。最近 6 小时未查到 Kernel-Power 41、异常关机 6008、资源耗尽 2004，或提及 WSL、vmmem、HCS 的应用崩溃记录。Hyper-V Compute 日志当前无读取权限。

Linux 前次启动日志没有记录 OOM kill 或 kernel panic。协议监控最后记录约 4.8 GiB 可用内存、全局 oom_kill=0。ResNet-50 SCI 双方同时 RSS 采样峰值约 885 MiB；当时在运行的轨迹分析器 RSS 约 18 MiB，但大量临时文件读写还会使用内核文件缓存。

Windows 的一次实时采样仅有约 1.9 GiB 可用物理内存，后续约 3.1 GiB。可以确认宿主机发生了重启，无法据现有记录确定重启之前是否因内存压力卡顿。没有通过重复触发整机崩溃验证原因。

## 已实施的限制

`tools/imagenet_resnet/memory_guard.py` 使用独立 cgroup v1 运行完整实验进程树：

- RAM 上限 2 GiB，RAM 加 swap 总上限 2.5 GiB，包含归属该组的文件缓存。
- 监控器位于限制组之外，协议、轨迹分析器及其后代继承同一上限。
- Linux 可用内存连续 3 次低于 512 MiB，或 Windows 可用物理内存连续 2 次低于 768 MiB、可用虚拟内存连续 2 次低于 1 GiB 时，停止该实验组并保留记录。Linux 每 2 秒采样，Windows 每约 15 秒采样。
- 终止时重复检查该组后代；启动失败和清理失败写入明确状态。
- 多组完整密态协议串行执行。两个模型的 50,000 张精度验证已完成，不再并行运行精度任务。

保护仅限制实验进程，不限制 Windows 上其他应用的内存，也不保证消除所有宿主机卡顿原因。cgroup 在 WSL 重启后需要重新配置。

## 实际验证

`memory_protection/limit_integration.json` 是刻意缩小到 128 MiB、禁止 swap 的保护测试：申请 192 MiB 的测试进程收到 SIGKILL，组内 OOM 计数增加 1，WSL 保持运行。这次人为测试的 OOM 应从后续实验增量中区分。

`descendant_integration.json` 验证后代继承限制；`launch_failure_integration.json` 验证启动失败不会遗留 running 状态；`descendant_cleanup_integration.json` 验证后代被清理且组为空。正式额度已恢复为 2 GiB RAM 加 512 MiB swap。

## 测量记录

已完成的 SCI ResNet-50 协议已核对全部 1,000 个输出整数，时间与双方发送字节从小型原始 JSON 恢复至 `COST_MEASUREMENT.json`，无需重新读取约 31 GiB 的详细轨迹。该原始协议未施加上述 cgroup 限制，恢复统计不能改变这一事实；保留 `round_trace=on`，因果轮数和 LAN/WAN 投影为缺失值。

新的 `*_cost_q1` 批次关闭详细轨迹，保留真实 CPU、墙钟、发送字节和最终整数核对。其正式入口必须验证当前启动周期中的实际 cgroup 成员关系与限额，拒绝无保护或过期保护报告的整网运行。新计划记录 `resource_guard`；两种轨迹模式不直接计算墙钟比值。CPU 和墙钟还需结合实际 swap 和限额回收情况判断。
