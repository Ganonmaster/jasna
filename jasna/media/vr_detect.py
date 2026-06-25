"""
Automatic VR format detection for the restoration pipeline.

Classifies a video's stereo layout, projection and angular coverage so the rest of
the pipeline can pick sane defaults (and decide whether/how to remap) without the
user hand-ticking boxes per file. The result feeds the unified VR config.

Detection is ranked, most-authoritative first:

  1. Embedded metadata  - ffprobe side_data ("Spherical Mapping" projection +
                          "Stereo 3D" layout), plus a binary scan for the Google
                          Spherical Video V1 ('GSpherical:' / uuid) blob. Rare on
                          re-encoded JAV rips (none of our test files carry it),
                          but cheap and authoritative when present.
  2. Filename           - HereSphere / DeoVR tag convention (_LR/_TB/_180/_MKX200
                          ...) plus a small, *empirically observed* studio-prefix
                          table. The studio table is deliberately tiny: only what
                          we have actually verified on disk. Do NOT pad it with
                          guesses - build it from your own corpus.
  3. Geometry           - per-eye aspect ratio (after the candidate stereo split)
                          plus a max-projection content envelope to read off the
                          projection. Three per-eye shapes are distinguished:
                            * fills the eye edge-to-edge        -> equirect (full)
                            * 180 content centred in a wider     -> equirect (padded:
                              canvas, black side pillars,           180 deg of a 360
                              ~constant-width rectangle             deg-wide canvas)
                            * bright disc that pinches toward    -> fisheye
                              the poles (black corners/pillars)

Layout/coverage are resolved first, then projection (the projection envelope test
needs to know which half of the frame is one eye).
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from jasna.os_utils import resolve_executable, subprocess_no_window_kwargs

logger = logging.getLogger(__name__)

# Layout / projection vocab (plain strings so they serialize cleanly into CLI args,
# AppSettings and presets without enum-import churn).
LAYOUT_SBS = "sbs"
LAYOUT_TB = "tb"
LAYOUT_MONO = "mono"
PROJ_EQUIRECT = "equirect"
PROJ_FISHEYE = "fisheye"
PROJ_FLAT = "flat"


@dataclass(frozen=True)
class VRFormat:
    """What a file *is* (the input side). The processing config is resolved from this.

    coverage  the CONTENT's angular coverage in degrees (180 for VR180, 360 for full
              sphere; 0 for flat). VR180 is 180 regardless of how it is packed.
    padded    equirect only: the 180 deg content occupies the centre 180 deg of a
              360 deg-wide eye, with black side pillars (vs. filling the eye). This
              changes the longitude scale used by the fisheye remap.
    """

    is_vr: bool
    layout: str          # sbs | tb | mono
    projection: str      # equirect | fisheye | flat
    coverage: int        # content angular coverage: 180 | 360 (0 for flat)
    padded: bool         # equirect: 180 content centred in a 360-wide eye (pillars)
    fov_deg: float       # fisheye lens FOV (== coverage for equirect)
    source: str          # metadata | filename | geometry | mixed | none
    confidence: str      # high | medium | low
    width: int
    height: int
    notes: str = ""

    def summary(self) -> str:
        if not self.is_vr:
            return f"non-VR / flat ({self.width}x{self.height}) [{self.confidence}]"
        pad = " padded" if (self.projection == PROJ_EQUIRECT and self.padded) else ""
        return (
            f"{self.layout.upper()} {self.projection}{pad} {self.coverage}deg "
            f"({self.width}x{self.height}) via {self.source} [{self.confidence}]"
        )

    def to_dict(self) -> dict:
        return asdict(self)


EYE_BOTH = "both"
EYE_LEFT = "left"
EYE_RIGHT = "right"


@dataclass(frozen=True)
class VRConfig:
    """Resolved processing decisions for one video (the operational side).

    Produced by resolve_vr_config() from detection + user overrides, then threaded
    through the pipeline. The two booleans are *derived* from the resolved input /
    processing / output projections rather than set by hand:

      remap_on_decode      input projection -> processing projection (equirect->fisheye),
                           applied before detection on every decoded frame.
      reproject_on_encode  processing projection -> output projection (fisheye->equirect),
                           applied before encode (only on restored frames).

    layout / eye_hfov / fov_deg / eye describe the stereo geometry for the remap grids:
      layout    sbs | tb | mono   (the split axis)
      eye_hfov  the equirect eye's horizontal span in degrees (180 when the 180 deg
                content fills the eye; 360 when it is centred in a 360-wide canvas).
                Controls how longitude maps to source-x in the remap.
      fov_deg   fisheye lens FOV for the equidistant remap grid.
      eye       both | left | right (single-eye processing).
    """

    remap_on_decode: bool = False
    reproject_on_encode: bool = False
    layout: str = LAYOUT_SBS
    eye_hfov: float = 180.0
    fov_deg: float = 180.0
    eye: str = EYE_BOTH

    @property
    def active(self) -> bool:
        """Whether any VR-specific frame work happens (remap / reproject / eye select)."""
        return self.remap_on_decode or self.reproject_on_encode or self.eye != EYE_BOTH

    def describe(self) -> str:
        bits = [f"layout={self.layout}"]
        if self.remap_on_decode:
            pad = " padded" if self.eye_hfov >= 360 else ""
            bits.append(f"remap->fisheye(fov={self.fov_deg:.0f}{pad})")
        if self.reproject_on_encode:
            bits.append("reproject->source")
        if not self.remap_on_decode and not self.reproject_on_encode:
            bits.append("no-remap (restore in source projection)")
        if self.eye != EYE_BOTH:
            bits.append(f"eye={self.eye}")
        return ", ".join(bits)


# --------------------------------------------------------------------------- #
# Tier 1: ffprobe + V1 binary scan
# --------------------------------------------------------------------------- #

def _ffprobe_stream(path: str) -> dict:
    """First video stream as a dict, including side_data_list when present."""
    ffprobe = resolve_executable("ffprobe")
    cmd = [
        ffprobe, "-v", "quiet", "-print_format", "json",
        "-select_streams", "v:0", "-show_streams", path,
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, **subprocess_no_window_kwargs())
    except OSError as e:
        logger.warning("ffprobe not runnable for VR detect: %s", e)
        return {}
    if p.returncode != 0:
        logger.warning("ffprobe failed for VR detect (%s): %s", p.returncode,
                       (p.stderr or b"").decode(errors="replace")[:300])
        return {}
    try:
        streams = json.loads(p.stdout).get("streams", [])
    except json.JSONDecodeError:
        return {}
    return streams[0] if streams else {}


def _aspect(stream: dict) -> float:
    """Display aspect ratio of the *coded frame* (W:H), preferring DAR/SAR when sane."""
    w = int(stream.get("width", 0) or 0)
    h = int(stream.get("height", 0) or 0)
    if w <= 0 or h <= 0:
        return 0.0
    dar = stream.get("display_aspect_ratio")
    if dar and ":" in dar:
        try:
            dn, dd = (int(x) for x in dar.split(":"))
            if dn > 0 and dd > 0:
                return dn / dd
        except ValueError:
            pass
    return w / h


_STEREO3D_TO_LAYOUT = {
    "sidebyside": LAYOUT_SBS,
    "sidebyside (quincunx)": LAYOUT_SBS,
    "topbottom": LAYOUT_TB,
    "2d": LAYOUT_MONO,
}
_SPHERICAL_TO_PROJ = {
    "equirectangular": PROJ_EQUIRECT,
    "half equirectangular": PROJ_EQUIRECT,
    "fisheye": PROJ_FISHEYE,
}


def _signal_from_metadata(stream: dict, path: str) -> dict:
    """Partial signal from embedded spherical/stereo metadata (V2 side_data, then V1 scan)."""
    sig: dict = {}
    for sd in stream.get("side_data_list", []):
        sdt = (sd.get("side_data_type") or "").lower()
        if sdt == "stereo 3d":
            layout = _STEREO3D_TO_LAYOUT.get((sd.get("type") or "").lower())
            if layout:
                sig["layout"] = layout
        elif sdt == "spherical mapping":
            projn = (sd.get("projection") or "").lower()
            proj = _SPHERICAL_TO_PROJ.get(projn)
            if proj:
                sig["projection"] = proj
            if projn == "half equirectangular":
                sig["coverage"] = 180
            elif projn == "equirectangular":
                bl = float(sd.get("bound_left", 0) or 0)
                br = float(sd.get("bound_right", 0) or 0)
                w = int(stream.get("width", 0) or 0)
                if w and (bl + br) > 0:
                    hfov = 360.0 * (1.0 - (bl + br) / w)
                    sig["coverage"] = 180 if hfov <= 200 else 360
                else:
                    sig["coverage"] = 360
    if sig:
        sig["source"] = "metadata"
        return sig

    v1 = _scan_gspherical_v1(path)
    if v1:
        v1["source"] = "metadata"
    return v1


def _scan_gspherical_v1(path: str, window: int = 2 << 20) -> dict:
    """Bounded binary scan for the Google Spherical Video V1 RDF/XML blob."""
    try:
        size = Path(path).stat().st_size
        with open(path, "rb") as f:
            head = f.read(window)
            if size > window:
                f.seek(max(0, size - window))
                tail = f.read(window)
            else:
                tail = b""
    except OSError:
        return {}
    blob = head + tail
    if b"GSpherical:" not in blob and b"spherical-video" not in blob:
        return {}
    try:
        text = blob.decode("latin-1", errors="replace")
    except Exception:
        return {}
    sig: dict = {}
    m = re.search(r"StereoMode>\s*([\w-]+)", text)
    if m:
        sig["layout"] = {"left-right": LAYOUT_SBS, "top-bottom": LAYOUT_TB,
                         "mono": LAYOUT_MONO}.get(m.group(1).lower(), LAYOUT_MONO)
    if re.search(r"ProjectionType>\s*equirectangular", text, re.I):
        sig["projection"] = PROJ_EQUIRECT
    mc = re.search(r"CroppedAreaImageWidthPixels>\s*(\d+)", text)
    mf = re.search(r"FullPanoWidthPixels>\s*(\d+)", text)
    if mc and mf and int(mf.group(1)) > 0:
        hfov = 360.0 * int(mc.group(1)) / int(mf.group(1))
        sig["coverage"] = 180 if hfov <= 200 else 360
    return sig


# --------------------------------------------------------------------------- #
# Tier 2: filename tags + studio table
# --------------------------------------------------------------------------- #

# HereSphere / DeoVR fisheye lens tags -> (coverage, lens fov_deg)
_FISHEYE_TAGS = {
    "F180": (180, 180.0),
    "FISHEYE190": (180, 190.0),
    "RF52": (180, 190.0),       # Canon RF 5.2mm dual-fisheye (~190deg)
    "MKX200": (180, 200.0),
    "MKX220": (180, 220.0),
    "MKX22": (180, 220.0),
    "VRCA220": (180, 220.0),
}

# Per-studio defaults, applied as a tier-2 (filename) hint. Each entry only fills the
# fields it lists; anything omitted is left to geometry. Two groups:
#
#   verified on disk  - full geometry confirmed by frame analysis on actual files.
#   baseline          - domain knowledge that the studio delivers EQUIRECT footage with
#                       the mosaic baked in camera/fisheye space, so it needs remap->
#                       fisheye for detection. Only 'projection' is asserted (which also
#                       guards against the envelope test mis-reading a frame as a fisheye
#                       disc); layout/coverage/padded are NOT verified, so they are left
#                       to geometry rather than guessed. Extend from your own corpus.
_STUDIO_TABLE: dict[str, dict] = {
    # verified on disk
    "VRKM":  {"layout": LAYOUT_SBS, "projection": PROJ_EQUIRECT, "coverage": 180, "padded": False, "fov_deg": 180.0},
    "3DSVR": {"layout": LAYOUT_TB,  "projection": PROJ_EQUIRECT, "coverage": 180, "padded": True,  "fov_deg": 180.0},
    "VOVS":  {"layout": LAYOUT_TB,  "projection": PROJ_EQUIRECT, "coverage": 180, "padded": True,  "fov_deg": 180.0},
    # baseline: known to need fisheye conversion for detection (equirect, camera-space mosaic)
    "SAVR":  {"projection": PROJ_EQUIRECT},
    "DSVR":  {"projection": PROJ_EQUIRECT},
    "PXVR":  {"projection": PROJ_EQUIRECT},
    "HUNVR": {"projection": PROJ_EQUIRECT},
}


def _signal_from_filename(path: str) -> dict:
    stem = Path(path).stem
    upper = stem.upper()
    tokens = set(re.split(r"[\s_\-.]+", upper))
    sig: dict = {}

    if tokens & {"LR", "RL", "SBS"}:
        sig["layout"] = LAYOUT_SBS
    elif tokens & {"TB", "BT", "OU"}:
        sig["layout"] = LAYOUT_TB
    elif tokens & {"MONO", "2D"}:
        sig["layout"] = LAYOUT_MONO

    for tag, (cov, fov_deg) in _FISHEYE_TAGS.items():
        if tag in tokens:
            sig["projection"], sig["coverage"], sig["fov_deg"] = PROJ_FISHEYE, cov, fov_deg
            sig["padded"] = False
            sig.setdefault("layout", LAYOUT_SBS)
            break
    else:
        if "EAC360" in tokens:
            # equi-angular cubemap: labelled/detected but has no remap path (unsupported)
            sig["projection"], sig["coverage"] = "eac", 360
        elif "360" in tokens:
            sig["projection"], sig["coverage"] = PROJ_EQUIRECT, 360
            sig.setdefault("layout", LAYOUT_MONO)
        elif "180" in tokens:
            sig["projection"], sig["coverage"] = PROJ_EQUIRECT, 180
            sig.setdefault("layout", LAYOUT_MONO)

    if sig:
        sig["source"] = "filename"

    m = re.match(r"^([A-Za-z0-9]+)[-_ ]\d", stem)
    if m:
        entry = _STUDIO_TABLE.get(m.group(1).upper())
        if entry:
            for key, val in entry.items():
                if val is not None:
                    sig.setdefault(key, val)
            sig["source"] = "filename"
    return sig


# --------------------------------------------------------------------------- #
# Tier 3: geometry (aspect + frame content)
# --------------------------------------------------------------------------- #

def _classify_aspect(ar: float) -> str:
    if 1.9 <= ar <= 2.1:
        return "2:1"
    if 0.92 <= ar <= 1.08:
        return "1:1"
    if 3.7 <= ar <= 4.3:
        return "4:1"
    return "flat"


def _extract_frames(path: str, duration: float, n: int = 8, width: int = 640) -> list:
    """Grab n evenly-spaced frames (scaled down) as BGR arrays via ffmpeg->png->cv2."""
    import cv2
    import numpy as np

    ffmpeg = resolve_executable("ffmpeg")
    frames = []
    fracs = [(i + 1) / (n + 1) for i in range(n)]
    for fr in fracs:
        ts = max(0.0, duration * fr) if duration and duration > 1 else 0.0
        cmd = [
            ffmpeg, "-v", "quiet", "-ss", f"{ts:.2f}", "-i", path,
            "-frames:v", "1", "-vf", f"scale={width}:-2",
            "-f", "image2pipe", "-vcodec", "png", "pipe:1",
        ]
        try:
            p = subprocess.run(cmd, capture_output=True, **subprocess_no_window_kwargs())
        except OSError:
            continue
        if p.returncode != 0 or not p.stdout:
            continue
        img = cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img)
    return frames


def _primary_eye(img, layout: str):
    """Crop one eye from a full stereo frame for the projection test."""
    h, w = img.shape[:2]
    if layout == LAYOUT_TB:
        return img[: h // 2, :]
    if layout == LAYOUT_SBS:
        return img[:, : w // 2]
    return img


def _eye_envelope(path: str, layout: str, duration: float):
    """Max-projection luma of one eye over several frames (geometric content envelope).

    Black padding (no sensor data) stays ~0 in every frame, so the max over frames is
    the clean content region regardless of any single frame's brightness. Returns a 2D
    float array or None if no frames could be read.
    """
    import numpy as np
    frames = _extract_frames(path, duration, n=8, width=640)
    if not frames:
        return None
    eyes = [_primary_eye(f, layout).astype(np.float32) for f in frames]
    h = min(e.shape[0] for e in eyes)
    w = min(e.shape[1] for e in eyes)
    env = np.max([e[:h, :w] for e in eyes], axis=0)
    return env.mean(axis=2)  # luma


def _classify_projection(env, per_eye_ar: float) -> tuple[str, bool, int]:
    """(projection, padded, coverage) from a max-projection eye envelope.

      - content fills the eye edge to edge        -> equirect (full)
      - 180 content centred in a 360-wide eye      -> equirect (padded)
        (black side pillars, ~constant-width band)
      - bright disc pinching toward the poles      -> fisheye
    """
    import numpy as np
    H, W = env.shape
    eb = max(2, int(W * 0.05))
    # MEAN (not max) of the edge band on the max-projection envelope: a structural black
    # side-pillar averages near 0 even when a stray bright pixel spikes the max, whereas
    # content that fills the eye edge-to-edge averages bright (measured: padded pillars
    # ~0-13, full-equirect edges ~55-100).
    pillars = float(env[:, :eb].mean()) < 30.0 and float(env[:, -eb:].mean()) < 30.0
    if not pillars:
        return PROJ_EQUIRECT, False, (360 if per_eye_ar >= 1.6 else 180)

    mask = env > 8.0

    def row_width(frac: float) -> int:
        y = min(H - 1, max(0, int(H * frac)))
        cols = np.where(mask[y])[0]
        return int(cols[-1] - cols[0]) if len(cols) else 0

    mid = max(row_width(0.45), row_width(0.5), row_width(0.55))
    edge = min(row_width(0.05), row_width(0.95))  # most-pinched near-pole rows
    if mid <= 0:
        return PROJ_EQUIRECT, True, 180
    # rectangle (constant width to the poles) -> padded equirect;
    # circular disc (pinches to a chord near the poles) -> fisheye.
    if edge / mid >= 0.55:
        return PROJ_EQUIRECT, True, 180
    return PROJ_FISHEYE, False, 180


def _signal_from_geometry(stream: dict, path: str, probe_frame: bool) -> dict:
    w = int(stream.get("width", 0) or 0)
    h = int(stream.get("height", 0) or 0)
    if w <= 0 or h <= 0:
        return {}
    cls = _classify_aspect(_aspect(stream))
    sig: dict = {"source": "geometry"}

    if cls == "flat":
        sig.update(layout=LAYOUT_MONO, projection=PROJ_FLAT, coverage=0, padded=False, is_vr=False)
        return sig

    if cls == "1:1":
        sig["layout"] = LAYOUT_TB           # square -> top/bottom (sbs would give portrait eyes)
    elif cls == "4:1":
        sig["layout"] = LAYOUT_SBS          # each eye 2:1
    else:  # 2:1 -- ambiguous between mono-360-equirect and VR180-SBS; needs a stereo hint
        sig["layout"] = LAYOUT_SBS
        sig["_ambiguous_2to1"] = True

    layout = sig["layout"]
    per_eye_ar = (w / (h / 2)) if layout == LAYOUT_TB else \
                 ((w / 2) / h) if layout == LAYOUT_SBS else (w / h)

    if not probe_frame:
        sig["_frame_sampled"] = False
        return sig

    duration = float(stream.get("duration", 0) or 0)
    env = _eye_envelope(path, layout, duration)
    sig["_frame_sampled"] = env is not None
    if env is None:
        return sig

    proj, padded, cov = _classify_projection(env, per_eye_ar)
    sig["projection"] = proj
    sig["padded"] = padded
    sig["coverage"] = cov
    return sig


# --------------------------------------------------------------------------- #
# Resolver: detection -> VRFormat
# --------------------------------------------------------------------------- #

_RANK = {"metadata": 3, "filename": 2, "geometry": 1, None: 0}


def detect_vr_format(path: str, *, probe_frame: bool = True) -> VRFormat:
    """Detect a file's VR layout/projection/coverage. Never raises; degrades to 'low'/none."""
    stream = _ffprobe_stream(path)
    w = int(stream.get("width", 0) or 0)
    h = int(stream.get("height", 0) or 0)

    geom = _signal_from_geometry(stream, path, probe_frame) if stream else {}
    name = _signal_from_filename(path)
    meta = _signal_from_metadata(stream, path) if stream else {}

    layout = projection = None
    coverage = fov_deg = None
    padded = None
    set_by: dict[str, str] = {}
    for sig in (geom, name, meta):  # weakest first; stronger tiers override
        src = sig.get("source")
        if "layout" in sig:
            layout, set_by["layout"] = sig["layout"], src
        if "projection" in sig:
            projection, set_by["projection"] = sig["projection"], src
        if "coverage" in sig:
            coverage, set_by["coverage"] = sig["coverage"], src
        if "padded" in sig:
            padded, set_by["padded"] = sig["padded"], src
        if "fov_deg" in sig:
            fov_deg, set_by["fov_deg"] = sig["fov_deg"], src

    notes: list[str] = []

    if geom.get("is_vr") is False and "layout" not in name and "layout" not in meta:
        return VRFormat(False, LAYOUT_MONO, PROJ_FLAT, 0, False, 0.0, "geometry", "high",
                        w, h, "flat aspect ratio; no VR signal")

    if layout is None:
        layout = LAYOUT_SBS
        notes.append("layout defaulted to SBS")
    if projection is None:
        projection = PROJ_EQUIRECT
        notes.append("projection defaulted to equirect")
    if coverage is None:
        coverage = 180
    if padded is None:
        padded = False
    if fov_deg is None:
        fov_deg = float(coverage) if projection == PROJ_EQUIRECT else 180.0

    if geom.get("_ambiguous_2to1") and set_by.get("layout") == "geometry":
        notes.append("2:1 frame is ambiguous (mono-360 vs SBS-180); assumed SBS-180 "
                     "- override if this is mono 360")
    if geom.get("_frame_sampled") is False and not meta and "projection" not in name:
        notes.append("could not sample a frame; projection is a guess")
    if projection == PROJ_FISHEYE and set_by.get("fov_deg") != "filename":
        notes.append(f"fisheye lens FOV assumed {fov_deg:.0f}deg; tune per-studio if remap looks off")

    decisive = min(_RANK[set_by.get("layout")], _RANK[set_by.get("projection")])
    confidence = {3: "high", 2: "medium"}.get(decisive, "low")

    agreeing = list(dict.fromkeys(
        sig["source"] for sig in (geom, name, meta)
        if sig.get("source") and sig.get("layout") == layout
        and sig.get("projection") == projection
    ))
    if len(agreeing) >= 2:
        source = "+".join(agreeing)
        if confidence == "medium":
            confidence = "high"
    elif len(agreeing) == 1:
        source = agreeing[0]
    else:
        contributing = {set_by.get(k) for k in ("layout", "projection")} - {None}
        source = contributing.pop() if len(contributing) == 1 else \
            ("mixed" if contributing else "none")
        if source == "none":
            confidence = "low"

    is_vr = layout in (LAYOUT_SBS, LAYOUT_TB) or projection in (PROJ_EQUIRECT, PROJ_FISHEYE)

    return VRFormat(is_vr, layout, projection, int(coverage), bool(padded), float(fov_deg),
                    source, confidence, w, h, "; ".join(notes))


# --------------------------------------------------------------------------- #
# Resolve detection + user overrides into the operational VRConfig
# --------------------------------------------------------------------------- #

def resolve_vr_config(
    path: str,
    *,
    mode: str = "auto",                 # auto | on | off
    input_projection: str = "auto",     # auto | equirect | fisheye
    output_projection: str = "source",  # source | equirect | fisheye
    layout: str = "auto",               # auto | sbs | tb | mono
    fov: str = "auto",                  # auto | 180 | 360  (content coverage)
    eye: str = EYE_BOTH,                # both | left | right
    mosaic_space: str = "auto",         # auto | fisheye | projected
    fov_deg: float | None = None,       # None/auto -> from detection
    probe_frame: bool = True,
) -> tuple[VRConfig | None, VRFormat | None]:
    """Resolve detection + user overrides into a VRConfig. Returns (config, detected).

    config is None when VR processing is disabled (mode=off, or mode=auto on a non-VR
    input). detected is the raw detection (None when nothing needed detecting).
    """
    if mode == "off":
        return None, None

    need_detect = mode == "auto" or fov_deg is None or any(
        v == "auto" for v in (input_projection, layout, fov, mosaic_space)
    )
    detected = detect_vr_format(path, probe_frame=probe_frame) if need_detect else None

    if mode == "auto" and (detected is None or not detected.is_vr):
        return None, detected  # not VR -> no special handling

    in_proj = input_projection if input_projection != "auto" else \
        (detected.projection if detected else PROJ_EQUIRECT)
    lay = layout if layout != "auto" else (detected.layout if detected else LAYOUT_SBS)
    cov = int(fov) if fov in ("180", "360") else (detected.coverage if detected else 180)
    padded = bool(detected.padded) if detected else False
    if fov_deg is not None:
        lens_fov = float(fov_deg)
    elif detected is not None:
        lens_fov = detected.fov_deg
    else:
        lens_fov = 180.0

    # Where the mosaic is rectangular -> the space we must restore in. Default
    # 'fisheye': JAV VR mosaics are typically baked in camera (fisheye) space, so
    # equirect-delivered footage (tight or padded) must be un-warped to fisheye first.
    msp = mosaic_space if mosaic_space != "auto" else PROJ_FISHEYE
    proc_proj = PROJ_FISHEYE if msp == PROJ_FISHEYE else in_proj
    out_proj = in_proj if output_projection == "source" else output_projection

    remap_on_decode = (proc_proj == PROJ_FISHEYE and in_proj == PROJ_EQUIRECT)
    reproject_on_encode = remap_on_decode and (out_proj != PROJ_FISHEYE)

    if cov == 360 and remap_on_decode:
        logger.warning("360 equirect->fisheye remap is not supported yet; "
                       "restoring in the source projection instead")
        remap_on_decode = reproject_on_encode = False

    # equirect eye horizontal span: 360 when the 180 content is padded into a wider
    # canvas, else 180 (content fills the eye).
    eye_hfov = 360.0 if (in_proj == PROJ_EQUIRECT and padded) else 180.0

    return VRConfig(
        remap_on_decode=remap_on_decode,
        reproject_on_encode=reproject_on_encode,
        layout=lay,
        eye_hfov=eye_hfov,
        fov_deg=lens_fov,
        eye=eye,
    ), detected


# --------------------------------------------------------------------------- #
# CLI:  python -m jasna.media.vr_detect FILE [FILE ...]
# --------------------------------------------------------------------------- #

def _main(argv: list[str]) -> int:
    if not argv:
        print("usage: python -m jasna.media.vr_detect FILE [FILE ...]")
        return 2
    for f in argv:
        try:
            fmt = detect_vr_format(f)
        except Exception as e:  # noqa: BLE001 - CLI convenience
            print(f"{Path(f).name}: ERROR {e}")
            continue
        print(f"{Path(f).name}: {fmt.summary()}")
        if fmt.notes:
            print(f"    note: {fmt.notes}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv[1:]))
