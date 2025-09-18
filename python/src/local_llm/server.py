"""Programmatic llama-server launcher.

Provides start_server() returning a ServerHandle which manages a subprocess
running 'llama-server'. All configuration is passed as keyword arguments.
"""
from __future__ import annotations

import os, signal, subprocess, threading, time, shutil, urllib.request, textwrap
from dataclasses import dataclass
from typing import Iterable

class ServerLaunchError(RuntimeError):
    """Raised when server fails to start or exits prematurely."""
    pass

@dataclass
class ServerHandle:
    process: subprocess.Popen
    port: int
    log_thread: threading.Thread | None = None

    def is_alive(self) -> bool:
        return self.process.poll() is None

    def terminate(self, timeout: float = 10.0):
        if not self.is_alive():
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
            self.process.wait(timeout=timeout)
        except Exception:
            self.kill()

    def kill(self):
        if self.is_alive():
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except Exception:
                pass

    def wait(self):
        return self.process.wait()

    def __enter__(self):  # context manager support
        return self

    def __exit__(self, exc_type, exc, tb):
        self.terminate()

def _build_args(
    *,
    model_path: str,
    n_ctx: int,
    n_gpu_layers: int,
    port: int,
    host: str,
    threads: int | None,
    http_threads: int | None,
    slots: int | None,
    cors: str | None,
    log_disable: bool,
    log_colors: bool | None,
    verbose: bool,
    api_key: str | None,
    extra_args: Iterable[str] | None,
) -> list[str]:
    binpath = shutil.which("llama-server")
    if not binpath:
        raise FileNotFoundError("'llama-server' not found in PATH. Install it or add to PATH.")
    args: list[str] = [
        binpath,
        "-m", model_path,
        "-c", str(n_ctx),
        "-ngl", str(n_gpu_layers),
        "--port", str(port),
        "--host", host,
    ]
    if threads is not None:      args += ["-t", str(threads)]
    if http_threads is not None: args += ["--threads-http", str(http_threads)]
    if slots is not None:        args += ["--slots", str(slots)]
    if cors:                     args += ["--cors", cors]
    if log_disable:              args += ["--log-disable"]
    if log_colors is False:      args += ["--log-colors=0"]
    if verbose:                  args += ["-v"]
    if api_key:                  args += ["--api-key", api_key]
    if extra_args:               args += list(map(str, extra_args))
    return args

def _wait_health(proc: subprocess.Popen, port: int, timeout: float) -> None:
    url = f"http://127.0.0.1:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            try:
                tail_lines = proc.stdout.readlines()[-200:]  # type: ignore
                tail = "".join(tail_lines)
            except Exception:
                tail = "(no output)"
            raise ServerLaunchError(f"llama-server exited early before healthy. Last output:\n{tail}")
        try:
            with urllib.request.urlopen(url, timeout=0.5) as r:  # nosec
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.25)
    raise TimeoutError(f"Server did not report healthy within {timeout:.1f}s on port {port}")

def _stream_logs(proc: subprocess.Popen):
    for line in iter(proc.stdout.readline, ''):  # type: ignore
        if not line:
            break
        print(line, end="")

def start_server(
    *,
    model_path: str,
    n_ctx: int,
    n_gpu_layers: int,
    port: int = 8080,
    host: str = "127.0.0.1",
    threads: int | None = None,
    http_threads: int | None = None,
    slots: int | None = None,
    cors: str | None = None,
    log_disable: bool = False,
    log_colors: bool | None = None,
    verbose: bool = False,
    api_key: str | None = None,
    extra_args: list[str] | None = None,
    env: dict[str, str] | None = None,
    stream_logs: bool = True,
    health_timeout: float = 30.0,
    ensure_homebrew_path: bool = True,
) -> ServerHandle:
    """Launch a llama-server process and return a handle.

    Validation performed before spawning the process to fail fast on obvious
    user errors (bad paths, negative numbers, etc.).
    """
    # --- validation ---
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("model_path must be a non-empty string")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model_path does not exist: {model_path}")
    for name, val in {"n_ctx": n_ctx, "n_gpu_layers": n_gpu_layers, "port": port}.items():
        if not isinstance(val, (int,)) or val <= 0:
            raise ValueError(f"{name} must be a positive integer (got {val!r})")
    if port > 65535:
        raise ValueError("port must be <= 65535")
    for opt_name, opt_val in {
        "threads": threads,
        "http_threads": http_threads,
        "slots": slots,
    }.items():
        if opt_val is not None and (not isinstance(opt_val, int) or opt_val <= 0):
            raise ValueError(f"{opt_name} must be a positive integer if provided (got {opt_val!r})")
    if extra_args is not None:
        if not isinstance(extra_args, (list, tuple)) or not all(isinstance(x, (str, int, float)) for x in extra_args):
            raise TypeError("extra_args must be a list/tuple of str/int/float")
    if env is not None:
        if not isinstance(env, dict) or not all(isinstance(k, str) for k in env.keys()):
            raise TypeError("env must be a dict[str,str]")
    if ensure_homebrew_path and os.uname().sysname == "Darwin":  # type: ignore[attr-defined]
        os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ.get("PATH", "")
    if env:
        for k, v in env.items():
            os.environ[str(k)] = str(v)
    args = _build_args(
        model_path=model_path,
        n_ctx=int(n_ctx),
        n_gpu_layers=int(n_gpu_layers),
        port=int(port),
        host=host,
        threads=threads,
        http_threads=http_threads,
        slots=slots,
        cors=cors,
        log_disable=log_disable,
        log_colors=log_colors,
        verbose=verbose,
        api_key=api_key,
        extra_args=extra_args,
    )
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    try:
        _wait_health(proc, port=int(port), timeout=health_timeout)
    except Exception:
        try:
            if proc.stdout:
                leftover = proc.stdout.read()
                if leftover:
                    print(textwrap.shorten(leftover, 4000, placeholder="..."))
        finally:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                pass
        raise
    log_thread = None
    if stream_logs and not log_disable:
        log_thread = threading.Thread(target=_stream_logs, args=(proc,), daemon=True)
        log_thread.start()
    return ServerHandle(process=proc, port=int(port), log_thread=log_thread)
