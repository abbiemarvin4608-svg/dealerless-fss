# Google-30 历史回放与 SCI 契约工具

这是工单 01 的入口。历史程序与工单 02 的新独立质量程序分开保存。

依赖：Python 3（contract 导出另需 NumPy）、g++、已有 EzPC SCI 安装及 Eigen/OpenSSL/GMP。输入 CSV 为历史的完整 6,835 条输入；大文件保留在外部只读 reproduction 目录。小型源程序、量化头文件和完整测试顺序保存在结果目录，已打包 SCI 原始源码快照。生成的 `build/` 可删除后重建。

从仓库根执行，输出目录可以换成新的路径；下述命令不会写历史目录：

```bash
python3 project/our/tools/google30_replay/replay.py \
  --archive /home/jiang/dealerless-lunwen/reproduction_results/sirnn_google30_20261007 \
  --runtime /home/jiang/EzPC/SIRNN/reproduction \
  --output project/our/results/ticket01_native_replay/reference

python3 project/our/tools/google30_replay/trace_sci.py \
  --archive /home/jiang/dealerless-lunwen/reproduction_results/sirnn_google30_20261007 \
  --runtime /home/jiang/EzPC/SIRNN/reproduction \
  --sci /home/jiang/EzPC/SCI \
  --source-snapshot project/our/results/ticket01_native_replay/sources/sci_src.tar.gz \
  --output project/our/results/ticket01_native_replay/sci_trace \
  --row 0 --threads 4 --jobs 2

python3 project/our/tools/google30_replay/freeze_contract.py \
  --archive /home/jiang/dealerless-lunwen/reproduction_results/sirnn_google30_20261007 \
  --runtime /home/jiang/EzPC/SIRNN/reproduction \
  --sci /home/jiang/EzPC/SCI \
  --output project/our/results/ticket01_native_replay

python3 -m unittest discover -s project/our/tools/google30_replay/tests -v
```

`replay.py --rows 0,16,28` 可快速回算三个已知样本。`trace_sci.py --skip-build` 仅用于相同源码/配置的既有隔离构建。统计可以离线重建：

```bash
python3 project/our/tools/google30_replay/summarize_trace.py \
  --input project/our/results/ticket01_native_replay/sci_trace/party1_trace.csv.gz \
  --output /tmp/google30-primitive-totals.json
```

第一次建立新快照时可省略 `--source-snapshot` 使用当前 SCI 源；freeze_contract 从 `--sci` 导出新源码快照，所以应只在有意更新来源时执行。固定版本重放使用保存的快照和匹配 SCI 安装。历史命令的实际编译参数在 `sci_trace/build_commands.json` 中保留，其中当时的临时构建绝对路径只作运行记录，重建时由脚本生成新路径。

MSB / compare / equality / LUT 的日志不含秘密值；这些是嵌套 API，统计时保持各家族独立。`variant` 对 compare 表示严格方向/equality 标志，对截断表示 shift，其余为 0。时间步 0–98 为循环，99 为分类头；各 op 名对应生成程序中的有序调用。

工单 05 请先读 [接口交接](../../results/ticket01_native_replay/HANDOFF_TICKET05.md)。尤其不要将 SeeDot FLOATEXP 的中间结果当作 SCI LUT 的逐位 oracle。
