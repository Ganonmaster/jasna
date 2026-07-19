# Running Jasna on an Intel Arc GPU (Linux server)

Jasna supports Intel Arc GPUs (tested target: **Arc Pro B50**, Battlemage) through
`torch.xpu` for the tensor pipeline, **OpenVINO** for the RF-DETR detection model,
and **Quick Sync (QSV)** for video decode/encode. This document is the bring-up
runbook for a headless Linux server.

Scope of Intel support (as of the initial port):
- CLI batch export (`jasna --input … --output …`) and streaming (`--stream`).
- Detection: RF-DETR (via OpenVINO GPU) and YOLO models (eager PyTorch on xpu).
- Restoration: BasicVSR++ in eager PyTorch (fp32 — see the deform-conv note below).
- Not available on Intel: the GUI, and `--secondary-restoration` of any kind.
  As on the AMD build, secondary restoration is gated to NVIDIA for now —
  `rtx-super-res` (Nvidia Maxine SDK) and `unet-4x` (TensorRT) have no Intel
  equivalent, and `tvai` (external Topaz ffmpeg) is currently blocked by the same
  gate. Primary BasicVSR++ restoration is the Intel path.

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

# 10-bit pack sanity (regression check). XPU's float->int16 conversion
# saturates rather than wraps, so the P010 packer wraps explicitly; this
# confirms bright samples survive on this driver/torch combination:
python - <<'EOF'
import torch
from jasna.media.rgb_to_p010 import chw_rgb_to_p010_bt709_limited
white = torch.full((3, 64, 64), 255, dtype=torch.uint8, device="xpu")
v = chw_rgb_to_p010_bt709_limited(white)[:64].view(torch.uint16)[0, 0].item()
print("OK" if v == 60160 else f"FAIL: got {v}, expected 60160")
EOF
```

## Install

```bash
uv venv --python 3.13 .venv && source .venv/bin/activate
uv pip install -e ".[intel]" --extra-index-url https://download.pytorch.org/whl/xpu
```

`-e` (editable) is recommended for a git-checkout deployment: `git pull` takes
effect without reinstalling (re-run the install only when dependencies change).
A regular `uv pip install ".[intel]" …` works too.

Unlike the Nvidia dev setup, do **not** pass `--no-build-isolation` here: a fresh
Python 3.13 venv has no `setuptools`, so the build fails with
`ModuleNotFoundError: No module named 'setuptools'`. There is nothing to compile
natively on the Intel path, so standard build isolation is correct. (If you do
need the flag, run `uv pip install setuptools wheel` first.)

This installs `torch`/`torchvision` **+xpu** wheels, `pytorch-triton-xpu`, and
`openvino` (>= 2026.2, the first release supporting the Arc Pro B50). Do not
install the `nvidia` extra in the same environment.

## Run

The `--device` default is `cuda:0` (upstream's default); Intel users pass
`--device xpu:0` explicitly:

```bash
jasna --input in.mp4 --output out.mkv --device xpu:0
```

On first run Jasna warms up the OpenVINO engine cache under the model directory
(the "Preparing OpenVINO model cache (this may take several minutes)…" step);
later runs reuse the cache and start fast.

### Restoration precision (BasicVSR++ / deform_conv2d)

`torchvision.ops.deform_conv2d` — the core alignment op of BasicVSR++ — has no
XPU kernel (RFC pytorch/vision#8679), so on xpu Jasna substitutes a numerically
equivalent grid_sample composition that runs natively on the GPU (parity vs the
torchvision kernel verified to ~1e-5 fp32; whole-net agreement < 2e-3).
`JASNA_DEFORM_BACKEND=torchvision` reverts to the CPU-fallback behavior for
A/B comparison.

`--fp16` defaults to **on** (matching NVIDIA/AMD), and the native deform path
makes it safe on xpu — this is the recommended mode: fp16 roughly halves
restoration memory traffic, the bottleneck on a 224 GB/s card. Pass `--no-fp16`
to force fp32 if you want to A/B a clip for quality.

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

Jasna probes QSV by opening an `h264_qsv` encoder session at startup. **Decode**
falls back to FFmpeg software decoding if QSV is unavailable (slow but correct);
**encode requires a working QSV encoder** and raises if it is missing (mirroring
the AMD/AMF path — there is no software-encode fallback). Checklist:

1. `vainfo` lists encode entrypoints (`VAEntrypointEncSlice*`).
2. `vpl-gpu-rt` (oneVPL GPU runtime) is installed.
3. ReBAR is enabled (media driver requirement).
4. The PyAV wheel's FFmpeg includes QSV codecs:
   `python -c "import av; print('h264_qsv' in av.codecs_available)"`.
   PyPI PyAV wheels ≥ 16.1 include them; the self-built Linux wheel from
   `build_linux.sh` must be configured with `--enable-libvpl`.
5. Streaming uses the **system** ffmpeg (bundled `tools/ffmpeg` in releases) —
   that binary needs QSV support too (`ffmpeg -encoders | grep qsv`).

### Detection throughput (RF-DETR)

RF-DETR runs on the OpenVINO GPU plugin and is the pipeline's pace-setter on
Intel (restoration has spare capacity behind it). The main lever left to speed it
up is **INT8 quantization** (planned): the B50's INT8 throughput (170 TOPS) is
~8× its fp16, so an INT8 RF-DETR is the route to push detection past ~30 fps.

### Optional: torch.compile the restoration graph (~1.3x, opt-in)

BasicVSR++ runs eager on xpu (the TensorRT path is NVIDIA-only). Opting in to
torch.compile recovers part of that gap — measured ~1.3x on full clips on the
B50 (1068 -> 834 ms/clip):

```bash
sudo apt install libze-dev   # Level Zero headers; triton-xpu JIT needs them
JASNA_TORCH_COMPILE=1 jasna --input in.mp4 --output out.mkv --device xpu:0
```

The first run per machine compiles for ~30 minutes during startup (the same
one-time cost as the NVIDIA TensorRT engine build); compiled kernels persist in
`<model dir>/torchinductor_cache`, so later runs start in ~1 minute. Only
full-length clips run compiled — shorter tail clips run eager (each distinct
clip length would trigger a fresh multi-minute compile). Any compile or runtime
failure falls back to eager with a warning; it cannot break a run.
`JASNA_COMPILE_MODE=max-autotune` trades a longer compile for potentially
faster kernels (unmeasured).

### Profiling the deform-conv share (native-kernel ROI)

`torchvision.ops.deform_conv2d` has no XPU kernel, so xpu runs a grid_sample
composition (see above). To measure what that costs — and what a fused
Triton/SYCL kernel or `torch.compile` could recover — run the opt-in profile:

```bash
jasna --benchmark --benchmark-filter deform --device xpu:0
```

It reports deform's share of a 60-frame clip, a per-stage breakdown of the
composition, a measured bandwidth floor (the fused-kernel ceiling), and a
`torch.compile` probe (skip with `JASNA_PROFILE_COMPILE=0`).

### Performance expectations

The B50 is a 224 GB/s, ~21 TFLOPS-fp16 card; BasicVSR++ is bandwidth-bound.
Expect a large FPS gap versus mid/high-end Nvidia GPUs regardless of software —
treat the Arc server as a capacity/offload node. Decode/encode via QSV and
detection via OpenVINO are not the bottleneck; restoration is.

## What runs where (Intel path)

| Stage | Backend |
|---|---|
| Decode | `*_qsv` copy-back decoders → pinned host buffer → xpu upload → torch YUV→RGB |
| Detection (RF-DETR) | OpenVINO GPU plugin, fp16 hint, blob cache in `<model>.openvino/<precision>-<target>/` next to the ONNX |
| Detection (YOLO) | ultralytics eager PyTorch on xpu |
| Restoration (BasicVSR++) | eager PyTorch on xpu (deform via native grid_sample composition; fp32 default, `--fp16` opt-in) |
| Blend / VR180 / color / LUT | torch ops on xpu |
| Encode | `*_qsv` (nv12/p010le system-memory frames); no software-encode fallback (raises if QSV is missing) |
| VRAM offloading | `torch.xpu.mem_get_info`-driven, same spill logic as CUDA |
