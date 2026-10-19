#!/usr/bin/env python3
"""Run a complete experiment tree in a bounded Linux memory cgroup."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/sys/fs/cgroup/memory')
MIB = 1 << 20
POWERSHELL = Path('/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe')


def group_path(uid):
    return ROOT / ('imagenet-resnet-%d' % uid)


def members(group):
    return [int(value) for value in (group / 'cgroup.procs').read_text().split()]


def configure(args):
    if os.geteuid() != 0:
        raise PermissionError('configure requires root; run only this setup step as root')
    group = group_path(args.uid)
    group.mkdir(exist_ok=True)
    if members(group):
        raise RuntimeError('experiment cgroup is occupied')
    memory, combined = args.memory_mib * MIB, (args.memory_mib + args.swap_mib) * MIB
    if memory < 128 * MIB or args.swap_mib < 0:
        raise ValueError('memory must be at least 128 MiB and swap nonnegative')
    # Raise the combined limit first, then lower it after the memory limit.
    old_combined = int((group / 'memory.memsw.limit_in_bytes').read_text())
    (group / 'memory.memsw.limit_in_bytes').write_text(str(max(old_combined, combined)))
    (group / 'memory.limit_in_bytes').write_text(str(memory))
    (group / 'memory.memsw.limit_in_bytes').write_text(str(combined))
    (group / 'memory.swappiness').write_text('10')
    (group / 'memory.oom_control').write_text('0')
    (group / 'memory.max_usage_in_bytes').write_text('0')
    (group / 'memory.memsw.max_usage_in_bytes').write_text('0')
    os.chown(group / 'cgroup.procs', args.uid, args.gid)
    print(json.dumps(dict(cgroup=str(group), memory_bytes=memory, memory_and_swap_bytes=combined)))


def sample(group):
    values = {key: int(value) for key, value in
              (line.split() for line in (group / 'memory.stat').read_text().splitlines())}
    oom = {key: int(value) for key, value in
           (line.split() for line in (group / 'memory.oom_control').read_text().splitlines())}
    system = {line.split(':')[0]: int(line.split()[1]) * 1024
              for line in Path('/proc/meminfo').read_text().splitlines()
              if line.startswith(('MemAvailable:', 'SwapFree:'))}
    return dict(epoch=time.time(), memory_bytes=int((group / 'memory.usage_in_bytes').read_text()),
                memory_and_swap_bytes=int((group / 'memory.memsw.usage_in_bytes').read_text()),
                kernel_peak_memory_bytes=int((group / 'memory.max_usage_in_bytes').read_text()),
                kernel_peak_memory_and_swap_bytes=int((group / 'memory.memsw.max_usage_in_bytes').read_text()),
                rss_bytes=values.get('total_rss'), cache_bytes=values.get('total_cache'),
                limit_hits=int((group / 'memory.failcnt').read_text()), oom_kill=oom.get('oom_kill', 0),
                linux_available_bytes=system['MemAvailable'], linux_swap_free_bytes=system['SwapFree'])


def windows_memory():
    if not POWERSHELL.is_file():
        return dict(status='unavailable')
    command = ('$m=Get-CimInstance Win32_OperatingSystem; '
               '@{free_physical_bytes=[long]$m.FreePhysicalMemory*1024; '
               'free_virtual_bytes=[long]$m.FreeVirtualMemory*1024} | ConvertTo-Json -Compress')
    try:
        result = subprocess.run([str(POWERSHELL), '-NoProfile', '-NonInteractive', '-Command', command],
                                capture_output=True, text=True, timeout=8, check=True)
        return dict(status='available', **json.loads(result.stdout))
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return dict(status='unavailable', reason=type(error).__name__)


def stop_group(group):
    for signum in (signal.SIGTERM, signal.SIGKILL):
        deadline = time.monotonic() + 5
        while members(group) and time.monotonic() < deadline:
            for pid in members(group):
                try:
                    os.kill(pid, signum)
                except ProcessLookupError:
                    pass
            time.sleep(.1)
    return members(group)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def run_background(args):
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        raise ValueError('a command is required after --')
    if args.output.exists():
        raise FileExistsError('choose a new guard output to preserve previous evidence')
    log_path = args.log or args.output.with_suffix('.log')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('x') as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'run',
            '--output', str(args.output.resolve()), '--', *command],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if args.output.exists():
            record = json.loads(args.output.read_text())
            print(json.dumps(dict(guard_pid=process.pid, status=record['status'],
                output=str(args.output.resolve()), log=str(log_path.resolve()))), flush=True)
            return 0 if record['status'] in ('running', 'complete') else 1
        if process.poll() is not None:
            raise RuntimeError('background guard failed to start; inspect ' + str(log_path))
        time.sleep(.1)
    raise RuntimeError('background guard startup not confirmed; inspect ' + str(log_path))


def run(args):
    if args.background:
        return run_background(args)
    if args.log:
        raise ValueError('--log requires --background')
    group = group_path(os.getuid())
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        raise ValueError('a command is required after --')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError('choose a new guard output to preserve previous evidence')
    with Path('/tmp/imagenet-resnet-memory-%d.lock' % os.getuid()).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if members(group):
            raise RuntimeError('experiment cgroup is occupied')
        memory = int((group / 'memory.limit_in_bytes').read_text())
        combined = int((group / 'memory.memsw.limit_in_bytes').read_text())
        if memory > 2 * 1024 * MIB or combined > 2560 * MIB:
            raise ValueError('configure an experiment limit of at most 2048 MiB RAM + 512 MiB swap')
        before = sample(group)
        record = dict(status='running', guard_pid=os.getpid(),
                      boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                      command=command, cgroup=str(group), memory_limit_bytes=memory,
                      memory_and_swap_limit_bytes=combined, started_epoch=time.time(),
                      before=before, samples=0, peak_memory_bytes=0, peak_memory_and_swap_bytes=0,
                      minimum_linux_available_bytes=before['linux_available_bytes'],
                      note='The cgroup limit includes anonymous memory and charged file cache for all descendants. Top-level peaks are sampled; kernel peaks cover the time since cgroup configuration.')
        write_json(args.output, record)

        def attach():
            (group / 'cgroup.procs').write_text(str(os.getpid()))

        process = None
        low_linux = low_windows = 0
        next_host = 0
        try:
            environment = dict(os.environ, IMAGENET_MEMORY_GUARD_REPORT=str(args.output.resolve()))
            process = subprocess.Popen(command, start_new_session=True, preexec_fn=attach, env=environment)
            record['command_pid'] = process.pid
            while True:
                current = sample(group)
                record.update(last=current, samples=record['samples'] + 1,
                              peak_memory_bytes=max(record['peak_memory_bytes'], current['memory_bytes']),
                              peak_memory_and_swap_bytes=max(record['peak_memory_and_swap_bytes'], current['memory_and_swap_bytes']),
                              minimum_linux_available_bytes=min(record['minimum_linux_available_bytes'], current['linux_available_bytes']))
                low_linux = low_linux + 1 if current['linux_available_bytes'] < 512 * MIB else 0
                if time.monotonic() >= next_host:
                    host = windows_memory()
                    record['windows_memory'] = host
                    if host['status'] == 'available':
                        low = host['free_physical_bytes'] < 768 * MIB or host['free_virtual_bytes'] < 1024 * MIB
                        low_windows = low_windows + 1 if low else 0
                    next_host = time.monotonic() + 15
                if low_linux >= 3 or low_windows >= 2:
                    record.update(status='resource_stopped', reason='sustained low system memory')
                    stop_group(group)
                write_json(args.output, record)
                if process.poll() is not None:
                    break
                time.sleep(2)
        except BaseException as error:
            record.update(status='interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed',
                          reason=type(error).__name__)
            stop_group(group)
            raise
        finally:
            remaining = []
            if members(group):
                remaining = stop_group(group)
            if process is not None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    record.update(status='cleanup_failed', reason='experiment process did not exit after SIGKILL')
            if remaining:
                record.update(status='cleanup_failed', remaining_pids=remaining)
            record.update(exit_code=process.returncode if process is not None else None,
                          finished_epoch=time.time(), last=sample(group))
            record['oom_kills'] = record['last']['oom_kill'] - before['oom_kill']
            if record['status'] == 'running':
                record['status'] = 'complete' if record['exit_code'] == 0 and not record['oom_kills'] else 'failed'
            write_json(args.output, record)
        return 0 if record['status'] == 'complete' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    setup = sub.add_parser('configure')
    setup.add_argument('--uid', type=int, required=True)
    setup.add_argument('--gid', type=int, required=True)
    setup.add_argument('--memory-mib', type=int, default=2048)
    setup.add_argument('--swap-mib', type=int, default=512)
    launch = sub.add_parser('run')
    launch.add_argument('--output', type=Path, required=True)
    launch.add_argument('--background', action='store_true', help='detach the guard and retain its log')
    launch.add_argument('--log', type=Path, help='background log path; defaults to output with .log suffix')
    launch.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return configure(args) if args.mode == 'configure' else run(args)


if __name__ == '__main__':
    raise SystemExit(main())
