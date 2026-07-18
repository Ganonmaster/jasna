# Running Jasna on an Intel Arc GPU (Linux server)

Jasna supports Intel Arc GPUs (tested target: **Arc Pro B50**, Battlemage) through
`torch.xpu` for the tensor pipeline, **OpenVINO** for the RF-DETR detection model,
and **Quick Sync (QSV)** for video decode/encode. This document is the bring-up
runbook for a headless Linux server.

Scope of Intel support (as of the initial port):
- CLI batch export (`jasna --input … --output …`) and streaming (`--stream`).
- Detection: RF-DETR (via OpenVINO GPU) and YOLO models (eager PyTorch on xpu).
- Restoration: BasicVSR++ in eager PyTorch (fp32 — see the deform-conv note below).
- Not available on Intel: the GUI, `--secondary-restoration rtx-super-res`
  (Nvidia Maxine hardware SDK) and `--secondary-restoration unet-4x` (TensorRT-only
  for now). `tvai` (external Topaz ffmpeg) works.

## Host requirements

| Component | Requirement |
|---|---|
| GPU | Arc B-series (Battlemage) or A-series; 16 GB VRAM recommended |
| Kernel | **6.12+** (`xe` driver drives Battlemage) |
| Firmware | **Resizable BAR enabled in BIOS — mandatory** (the media driver crashes without it) |
| OS | Ubuntu 24.04+ (Intel's B-series guidance: Ubuntu 25.04, kernel 6.14, Mesa 25.0.7) |
| Packages | `intel-compute-runtime` (NEO), `level-zero-loader` (a.k.a. `libze1`), `intel-media-driver`, `vpl-gpu-rt` (oneVPL runtime), `libva-utils` (for `vainfo`) |

Verification commands after driver install:

```bash
vainfo                       # must list H264/HEVC/AV1 decode + encode entrypoints
sycl-ls 2>/dev/null || true  # if oneAPI basekit present: should list the GPU
python -c "import torch; print(torch.xpu.is_available(), torch.xpu.get_device_name(0))"

# 10-bit pack sanity: the P010 converter relies on mod-2^16 float->int16 cast
# semantics (verified on CUDA and x86 CPU; run this once on xpu):
python - <<'EOF'
import torch
from jasna.media.rgb_to_p010 import chw_rgb_to_p010_bt709_limited
white = torch.full((3, 64, 64), 255, dtype=torch.uint8, device="xpu")
v = chw_rgb_to_p010_bt709_limited(white)[:64].view(torch.uint16)[0, 0].item()
print("OK" if v == 60160 else f"FAIL: got {v}, expected 60160 — xpu int16 cast saturates")
EOF
```

## Install

```bash
uv venv --python 3.13 .venv && source .venv/bin/activate
uv pip install ".[intel]" --extra-index-url https://download.pytorch.org/whl/xpu
```

Unlike the Nvidia dev setup, do **not** pass `--no-build-isolation` here: a fresh
Python 3.13 venv has no `setuptools`, so the build fails with
`ModuleNotFoundError: No module named 'setuptools'`. There is nothing to compile
natively on the Intel path, so standard build isolation is correct. (If you do
need the flag, run `uv pip install setuptools wheel` first.)

This installs `torch`/`torchvision` **+xpu** wheels, `pytorch-triton-xpu`, and
`openvino` (>= 2026.2, the first release supporting the Arc Pro B50). Do not
install the `nvidia` extra in the same environment.

## Run

```bash
jasna --list-devices                 # should show xpu:0 <GPU name>
jasna --input in.mp4 --output out.mkv --device xpu:0
```

`--device auto` (the default) selects `xpu:0` on an Intel install. On first run
Jasna warms up the OpenVINO engine cache (`model_weights/ov_cache/`) — the
"Compiling the OpenVINO engine cache" step; later runs start fast.

### fp16 note (BasicVSR++ / deform_conv2d)

`torchvision.ops.deform_conv2d` — the core alignment op of BasicVSR++ — has no
XPU kernel yet (RFC pytorch/vision#8679). PyTorch transparently falls back to
the CPU kernel for that op, which has no half-precision support. Jasna therefore
defaults to **fp32** on xpu (`--fp16` is off unless forced). Expect this op to be
the throughput bottleneck until the planned grid_sample-based XPU implementation
lands (`JASNA_DEFORM_BACKEND` selects the implementation once it exists).

## Known issues and workarounds

### torch.xpu aborts via Level Zero on Battlemage (intel/compute-runtime#922)

On some compute-runtime versions (observed 26.05–26.18) `torch.xpu` can abort
the process during resource initialization on B-series GPUs. Jasna's preflight
probes the GPU **in a subprocess**, and on failure automatically retries via the
OpenCL adapter; when that works it exports `ONEAPI_DEVICE_SELECTOR=opencl:gpu`
for the rest of the run. To force it manually:

```bash
ONEAPI_DEVICE_SELECTOR=opencl:gpu jasna --input … --output … --device xpu:0
```

If neither backend works, pin a known-good `intel-compute-runtime` version and
reboot after driver changes. OpenVINO and QSV are unaffected (they use
OpenCL/VA-API directly).

### QSV not detected

Jasna probes QSV by opening an `h264_qsv` encoder session at startup. If the
probe fails it logs a warning and falls back to software decode/encode
(libx264/libx265 — slow but correct). Checklist:

1. `vainfo` lists encode entrypoints (`VAEntrypointEncSlice*`).
2. `vpl-gpu-rt` (oneVPL GPU runtime) is installed.
3. ReBAR is enabled (media driver requirement).
4. The PyAV wheel's FFmpeg includes QSV codecs:
   `python -c "import av; print('h264_qsv' in av.codecs_available)"`.
   PyPI PyAV wheels ≥ 16.1 include them; the self-built Linux wheel from
   `build_linux.sh` must be configured with `--enable-libvpl`.
5. Streaming uses the **system** ffmpeg (bundled `tools/ffmpeg` in releases) —
   that binary needs QSV support too (`ffmpeg -encoders | grep qsv`).

### Performance expectations

The B50 is a 224 GB/s, ~21 TFLOPS-fp16 card; BasicVSR++ is bandwidth-bound.
Expect a large FPS gap versus mid/high-end Nvidia GPUs regardless of software —
treat the Arc server as a capacity/offload node. Decode/encode via QSV and
detection via OpenVINO are not the bottleneck; restoration is.

## What runs where (Intel path)

| Stage | Backend |
|---|---|
| Decode | `*_qsv` copy-back decoders → pinned host buffer → xpu upload → torch YUV→RGB |
| Detection (RF-DETR) | OpenVINO GPU plugin, fp16 hint, blob cache in `model_weights/ov_cache/` |
| Detection (YOLO) | ultralytics eager PyTorch on xpu |
| Restoration (BasicVSR++) | eager PyTorch on xpu, fp32 (deform_conv2d falls back to CPU) |
| Blend / VR180 / color / LUT | torch ops on xpu |
| Encode | `*_qsv` (nv12/p010le system-memory frames), software encoders as fallback |
| VRAM offloading | `torch.xpu.mem_get_info`-driven, same spill logic as CUDA |
