"""Ensure tensorrt_libs DLLs are loaded before nvvfx can override them.

Nvidia-only DLL-ordering workaround; AMD/Intel/CPU installs have neither
package. When nvvfx IS present the ordering guarantee is load-bearing, so a
broken tensorrt_libs must still abort collection instead of silently skipping.
"""
import importlib.util

try:
    import tensorrt_libs  # noqa: F401 — locks in tensorrt_libs nvinfer_10.dll before nvvfx
except ImportError:
    if importlib.util.find_spec("nvvfx") is not None:
        raise
