"""Process, build and resource management for local/CI regression runs.

This module intentionally depends only on Python's standard library. It never
executes shell configuration and never deletes an existing build or checkpoint.
"""

import contextlib
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import signal
import socket
import subprocess
import threading
import time
import uuid

_active_processes = set()
_process_mutex = threading.Lock()


def terminate_all():
    with _process_mutex:
        processes = list(_active_processes)
    for process in processes:
        _terminate_group(process)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    os.replace(str(tmp), str(path))


def unique_run_id():
    return (
        datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "-"
        + uuid.uuid4().hex[:8]
    )


def safe_case_path(root, case_id):
    parts = case_id.split("/")
    if not parts or any(p in ("", ".", "..") or "\\" in p for p in parts):
        raise ValueError("Unsafe case identifier: " + case_id)
    candidate = Path(root).joinpath(*parts)
    if Path(root).resolve() not in candidate.resolve().parents:
        raise ValueError("Case path escapes run directory")
    return candidate


def command_text(argv):
    return " ".join(shlex.quote(str(a)) for a in argv)


def _terminate_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    # The leader may already have exited while a descendant ignored SIGTERM.
    # Its original process group must still be killed before a GPU lease ends.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_process(argv, *, cwd, log_path, timeout, env=None, headers=None):
    """Run argv without a shell; timeout includes the entire process group."""
    argv = [str(a) for a in argv]
    if not argv or timeout <= 0:
        raise ValueError("A command and a positive timeout are required")
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = {
        "argv": argv,
        "cwd": str(cwd),
        "log": str(log_path),
        "status": "ERROR",
        "headers": dict(headers or {}),
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with log_path.open("w") as log:
        for key, value in result["headers"].items():
            log.write("[{}] {}\n".format(key, value))
        log.write("[COMMAND] " + command_text(argv) + "\n")
        log.write("[ARGV_JSON] " + json.dumps(argv, ensure_ascii=False) + "\n")
        log.write("[STARTED_AT] " + result["started_at"] + "\n")
        log.write("[CWD] " + str(cwd) + "\n")
        log.flush()
        try:
            process = subprocess.Popen(
                argv,
                cwd=str(cwd),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            result.update(reason=str(exc), returncode=None, elapsed_seconds=0)
            log.write(str(exc) + "\n")
            return result
        with _process_mutex:
            _active_processes.add(process)
        try:
            returncode = process.wait(timeout=timeout)
            result.update(
                returncode=returncode, status="PASS" if returncode == 0 else "FAIL"
            )
            if returncode:
                result["reason"] = "Process exited with status {}".format(returncode)
        except subprocess.TimeoutExpired:
            _terminate_group(process)
            result.update(
                returncode=process.returncode,
                status="ERROR",
                reason="timeout after {}s".format(timeout),
            )
        except BaseException:
            _terminate_group(process)
            raise
        finally:
            # A failed launcher can exit before its rank processes. No child
            # owned by this invocation may outlive the invocation's resources.
            _terminate_group(process)
            with _process_mutex:
                _active_processes.discard(process)
    result["elapsed_seconds"] = time.monotonic() - started
    result["completed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with log_path.open("a") as log:
        log.write("[COMPLETED_AT] " + result["completed_at"] + "\n")
        log.write("[EXIT_CODE] " + str(result.get("returncode")) + "\n")
    return result


def probe(argv):
    try:
        output = subprocess.check_output(argv, stderr=subprocess.DEVNULL, timeout=15)
        return output.decode("utf-8", errors="replace").strip()
    except (OSError, subprocess.SubprocessError):
        return None


def discover_gpus():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return [
            s.strip() for s in visible.split(",") if s.strip() and s.strip() != "-1"
        ]
    output = probe(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"])
    return output.splitlines() if output else []


def environment_fingerprint(repo, build_dir, backend):
    cache = Path(build_dir) / "CMakeCache.txt"
    options = {}
    if cache.exists():
        for line in cache.read_text(errors="replace").splitlines():
            if ":" in line and "=" in line and not line.startswith(("//", "#")):
                left, value = line.split("=", 1)
                name = left.split(":", 1)[0]
                if name in {
                    "CMAKE_BUILD_TYPE",
                    "CMAKE_CXX_COMPILER",
                    "CMAKE_CXX_FLAGS",
                    "CMAKE_CUDA_FLAGS",
                    "CMAKE_CUDA_COMPILER",
                    "USE_CUDA",
                    "USE_NCCL",
                    "USE_OMP",
                    "PROFILE_MODE",
                }:
                    options[name] = value
    gpu = probe(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    cpu = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu = next(
            (
                s.split(":", 1)[1].strip()
                for s in cpuinfo.read_text().splitlines()
                if s.startswith("model name")
            ),
            "unknown",
        )
    result = {
        "backend": backend,
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu": cpu,
        "gpu_inventory": gpu.splitlines() if gpu else [],
        "gpu_topology": (
            probe(["nvidia-smi", "topo", "-m"]) if backend == "cuda" else None
        ),
        "cuda_compiler": probe(["nvcc", "--version"]) if backend == "cuda" else None,
        "compiler": probe(["c++", "--version"]),
        "build_options": options,
        "nccl": (
            probe(["pkg-config", "--modversion", "nccl"]) if backend == "cuda" else None
        ),
        "container_image": os.environ.get("INFINITRAIN_TEST_IMAGE", "unspecified"),
    }
    return result


def source_metadata(repo):
    repo = Path(repo).resolve()
    git = [
        "git",
        "-c",
        "safe.directory=" + str(repo),
        "-c",
        "safe.directory=" + str(Path(repo) / ".git"),
        "-C",
        str(repo),
    ]
    listed = probe(
        git
        + [
            "ls-files",
            "-co",
            "--exclude-standard",
            "--",
            "infini_train",
            "example",
            "tests",
            "cmake",
            "tools",
            "CMakeLists.txt",
        ]
    )
    tree_hash = hashlib.sha256()
    for relative in sorted(set((listed or "").splitlines())):
        path = Path(repo) / relative
        if relative.startswith("tests/regression/") or not path.is_file():
            continue
        tree_hash.update(relative.encode() + b"\0" + file_sha256(path).encode() + b"\0")
    return {
        "commit": probe(git + ["rev-parse", "HEAD"]),
        "dirty": bool(probe(git + ["status", "--porcelain"])),
        "diff_sha256": hashlib.sha256(
            (probe(git + ["diff", "HEAD"]) or "").encode()
        ).hexdigest(),
        "compiled_source_sha256": tree_hash.hexdigest(),
        "hostname": socket.gethostname(),
    }


def file_sha256(path, cache_dir=None):
    """Hash bytes, recording an audit cache without trusting filesystem times.

    Network/container filesystems can preserve all stat fields across a content
    change. The caller deduplicates fixture paths within one preflight; a new
    invocation must verify bytes again to make its baseline identity trustworthy.
    """
    path = Path(path).resolve()
    stat = path.stat()
    identity = [
        str(path),
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    ]
    key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    cached = Path(cache_dir) / (key + ".json") if cache_dir else None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    if cached:
        try:
            atomic_json(cached, {"identity": identity, "sha256": value})
        except OSError:
            # The cache is optional; network/container ownership may make it
            # read-only even though the source file is readable.
            pass
    return value


class ResourcePool:
    """Atomic GPU-set leases shared with other regression runner processes."""

    def __init__(self, gpu_ids, lock_dir, wait_seconds=60):
        if len(gpu_ids) != len(set(gpu_ids)):
            raise ValueError("GPU identifiers must be unique")
        self.gpu_ids = list(gpu_ids)
        inventory = probe(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"]
        )
        aliases = {}
        for line in (inventory or "").splitlines():
            if "," in line:
                index, gpu_uuid = (part.strip() for part in line.split(",", 1))
                aliases[index] = gpu_uuid
                aliases[gpu_uuid] = gpu_uuid
        self.identities = {gpu: aliases.get(gpu, gpu) for gpu in self.gpu_ids}
        if len(set(self.identities.values())) != len(self.gpu_ids):
            raise ValueError("GPU aliases refer to duplicate physical devices")
        self.lock_dir = Path(lock_dir)
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self.wait_seconds = wait_seconds
        self.mutex = threading.Lock()

    @contextlib.contextmanager
    def lease(self, count, exclusive=False):
        count = len(self.gpu_ids) if exclusive and count else count
        if count < 0 or count > len(self.gpu_ids):
            raise RuntimeError(
                "Need {} GPUs; only {} configured".format(count, len(self.gpu_ids))
            )
        opened, selected = [], []
        deadline = time.monotonic() + self.wait_seconds
        while count:
            with self.mutex:
                for gpu in self.gpu_ids:
                    name = hashlib.sha256(self.identities[gpu].encode()).hexdigest()[
                        :16
                    ]
                    handle = (self.lock_dir / ("gpu-" + name + ".lock")).open("a+")
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        opened.append(handle)
                        selected.append(gpu)
                    except BlockingIOError:
                        handle.close()
                    if len(opened) == count:
                        break
                if len(opened) == count:
                    break
                for handle in opened:
                    fcntl.flock(handle, fcntl.LOCK_UN)
                    handle.close()
                opened, selected = [], []
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "GPU lease timeout; requested devices are used by another regression run"
                )
            time.sleep(0.2)
        try:
            yield selected
        finally:
            for handle in opened:
                fcntl.flock(handle, fcntl.LOCK_UN)
                handle.close()


@contextlib.contextmanager
def build_lock(build_dir):
    path = Path(build_dir)
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".regression-build.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def build_project(
    repo, build_dir, backend, output_dir, jobs=8, timeout=3600, extra_options=None,
    *, profile_mode=False, build_tests=True
):
    if backend not in ("cpu", "cuda"):
        raise ValueError(
            "External backends require --no-build and a provider build directory"
        )
    options = [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_TEST=" + ("ON" if build_tests else "OFF"),
        "-DPROFILE_MODE=" + ("ON" if profile_mode else "OFF"),
        "-DUSE_CUDA=" + ("ON" if backend == "cuda" else "OFF"),
        "-DUSE_NCCL=" + ("ON" if backend == "cuda" else "OFF"),
    ]
    options += list(extra_options or [])
    with build_lock(build_dir):
        before = source_metadata(repo)
        if not before["commit"]:
            return {
                "status": "ERROR",
                "reason": "Cannot verify source commit; check repository ownership and git availability",
                "log": str(Path(output_dir) / "configure.log"),
            }
        result = run_process(
            ["cmake", "-S", str(repo), "-B", str(build_dir)] + options,
            cwd=repo,
            log_path=Path(output_dir) / "configure.log",
            timeout=timeout,
        )
        if result["status"] != "PASS":
            return result
        result = run_process(
            ["cmake", "--build", str(build_dir), "--parallel", str(jobs)],
            cwd=repo,
            log_path=Path(output_dir) / "build.log",
            timeout=timeout,
        )
        if result["status"] == "PASS":
            after = source_metadata(repo)
            if before["compiled_source_sha256"] != after["compiled_source_sha256"]:
                result.update(
                    status="ERROR",
                    reason="Sources changed while building; rebuild before using provenance",
                )
                return result
            binaries = {}
            for name in ("gpt2", "llama3", "qwen3", "mixtral", "infini_run"):
                path = Path(build_dir) / name
                if path.is_file():
                    binaries[name] = file_sha256(path)
            atomic_json(
                Path(build_dir) / "build-manifest.json",
                {
                    "schema_version": 1,
                    "source": after,
                    "binaries": binaries,
                    "environment": environment_fingerprint(repo, build_dir, backend),
                },
            )
        return result


def verify_build_manifest(build_dir, binary_names, expected_commit=None):
    path = Path(build_dir) / "build-manifest.json"
    if not path.is_file():
        return {"verified": False, "reason": "No build-manifest.json"}
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1 or not data.get("source", {}).get("commit"):
        raise ValueError("Invalid build provenance manifest: " + str(path))
    if expected_commit and data["source"]["commit"] != expected_commit:
        raise ValueError("Reference commit differs from build provenance")
    for name in binary_names:
        if data.get("binaries", {}).get(name) != file_sha256(Path(build_dir) / name):
            raise ValueError("Build binary changed after provenance recording: " + name)
    return {"verified": True, "manifest": data, "path": str(path)}
