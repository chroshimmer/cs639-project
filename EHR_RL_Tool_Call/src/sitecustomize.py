from __future__ import annotations

import builtins
import importlib.metadata
from typing import Any

_original_import = builtins.__import__
_original_metadata_version = importlib.metadata.version
_patched_sglang_http_server = False
_patched_sglang_patch_torch = False


def _patch_sglang_http_server(module: Any) -> None:
    global _patched_sglang_http_server
    if _patched_sglang_http_server or hasattr(module, "_launch_subprocesses"):
        _patched_sglang_http_server = True
        return
    try:
        from sglang.srt.entrypoints.engine import Engine
    except Exception:
        return

    def _launch_subprocesses(*args: Any, **kwargs: Any):
        return Engine._launch_subprocesses(*args, **kwargs)

    module._launch_subprocesses = _launch_subprocesses
    _patched_sglang_http_server = True


def _patch_sglang_patch_torch(module: Any) -> None:
    global _patched_sglang_patch_torch
    if _patched_sglang_patch_torch:
        return
    if not hasattr(module, "_modify_tuple"):
        _patched_sglang_patch_torch = True
        return

    def _modify_tuple_safe(t: tuple[Any, ...], index: int, modifier: Any) -> tuple[Any, ...]:
        if index >= len(t) or index < -len(t):
            return t
        return (*t[:index], modifier(t[index]), *t[index + 1 :])

    module._modify_tuple = _modify_tuple_safe
    _patched_sglang_patch_torch = True


def _import_with_sglang_compat(name: str, globals=None, locals=None, fromlist=(), level: int = 0):
    module = _original_import(name, globals, locals, fromlist, level)
    http_server_name = "sglang.srt.entrypoints.http_server"
    if name == http_server_name or (name == "sglang.srt.entrypoints" and "http_server" in fromlist):
        try:
            import sglang.srt.entrypoints.http_server as http_server
        except Exception:
            return module
        _patch_sglang_http_server(http_server)
    patch_torch_name = "sglang.srt.utils.patch_torch"
    if name == patch_torch_name or (name == "sglang.srt.utils" and "patch_torch" in fromlist):
        try:
            import sglang.srt.utils.patch_torch as patch_torch
        except Exception:
            return module
        _patch_sglang_patch_torch(patch_torch)
    return module


builtins.__import__ = _import_with_sglang_compat


def _metadata_version_with_sglang_alias(distribution_name: str) -> str:
    if distribution_name == "sgl-kernel":
        try:
            return _original_metadata_version("sglang-kernel")
        except importlib.metadata.PackageNotFoundError:
            pass
    return _original_metadata_version(distribution_name)


importlib.metadata.version = _metadata_version_with_sglang_alias
