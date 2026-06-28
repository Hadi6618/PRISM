from google.colab import drive
from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import sys
import time

import torch

assert torch.cuda.is_available(), "No GPU found. Use Runtime -> Change runtime type -> GPU."
print("GPU:", torch.cuda.get_device_name(0))
print("Torch:", torch.__version__)

drive.mount("/content/drive")

REPO_URL = "https://github.com/AnaFilipaNogueira/Abnormal-Human-Behaviour-Detection-using-Normalising-Flows-and-Attention-Mechanisms.git"
REPO_DIR = Path("/content/DA-STG-NF")
ALPHAPOSE_DIR = Path("/content/AlphaPose")
# Pin to the AlphaPose commit observed in the matching Colab diagnostic so defaults do not drift.
ALPHAPOSE_COMMIT = "c60106d19afb443e964df6f06ed1842962f5f1f7"
LOCAL_POSE_WORK = Path("/content/stg_nf_alphapose_work")

# Change this if your original ShanghaiTech data is extracted somewhere else.
ORIGINAL_DATA_ROOT = Path("/content/shanghaitech")

# Default original-data candidates. Adjust manually if your folders differ.
TRAIN_SOURCE_ROOT = ORIGINAL_DATA_ROOT / "training/videos"
TEST_SOURCE_ROOT = ORIGINAL_DATA_ROOT / "testing/frames"

# Use "video" for .avi/.mp4 clips. Use "images" if each clip is already a folder of jpg/png frames.
TRAIN_SOURCE_MODE = "video"  # "video" or "images"
TEST_SOURCE_MODE = "images"

# Paper-style extraction: STG-NF paper says AlphaPose with YOLOX detector, then PoseFlow/pose tracking.
# Set False only if you intentionally want the literal repo gen_data.py command where YOLOX is commented out.
USE_YOLOX_DETECTOR = True
YOLOX_X_WEIGHTS_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth"

DRIVE_ROOT = Path("/content/drive/MyDrive/STG-NF/original_shanghaitech")
DRIVE_POSE_TRAIN = DRIVE_ROOT / "pose/train"
DRIVE_POSE_TEST = DRIVE_ROOT / "pose/test"
DRIVE_LOG_DIR = DRIVE_ROOT / "logs"

for directory in [LOCAL_POSE_WORK, DRIVE_POSE_TRAIN, DRIVE_POSE_TEST, DRIVE_LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

print("Repo target:", REPO_DIR)
print("AlphaPose target:", ALPHAPOSE_DIR)
print("Train source root:", TRAIN_SOURCE_ROOT)
print("Test source root:", TEST_SOURCE_ROOT)
#print("Source mode:", SOURCE_MODE)
print("Drive pose root:", DRIVE_ROOT)

!tar -xzvf "/content/drive/MyDrive/shanghaitech.tar.gz" -C "/content/"
if not REPO_DIR.exists():
    subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
os.chdir(REPO_DIR)
subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=True)
print("DA-STG-NF ready at", REPO_DIR)

!sed -i 's/args = parser.parse_args()/args, _ = parser.parse_known_args()/g' /content/STG-NF/models/STG_NF/attention.py
!sed -i 's/args = parser.parse_args()/args, _ = parser.parse_known_args()/g' /content/STG-NF/models/STG_NF/stgcn.py
# 1. (Optional) If it complains about it already existing, you can ignore the clone step if you already ran it.
!git clone https://github.com/orhir/STG-NF.git /tmp/STG-NF-original

# 2. Make sure the target data directory exists
!mkdir -p /content/STG-NF/data

# 3. Copy the CONTENTS of the data folder (Notice the '/.' at the end!)
!cp -a /tmp/STG-NF-original/data/. /content/STG-NF/data/

!apt-get -qq update
!apt-get -qq install -y libyaml-dev ffmpeg
!pip -q install gdown cython git+https://github.com/samson-wang/cython_bbox.git halpecocotools pycocotools munkres natsort easydict yacs pyyaml scipy tensorboardX terminaltables loguru thop

if not ALPHAPOSE_DIR.exists():
    subprocess.run(["git", "clone", "https://github.com/MVIG-SJTU/AlphaPose.git", str(ALPHAPOSE_DIR)], check=True)
if ALPHAPOSE_COMMIT:
    subprocess.run(["git", "fetch", "--all", "--tags"], cwd=ALPHAPOSE_DIR, check=True)
    subprocess.run(["git", "checkout", ALPHAPOSE_COMMIT], cwd=ALPHAPOSE_DIR, check=True)

# Compatibility patch for current Colab NumPy.
for py_file in ALPHAPOSE_DIR.rglob("*.py"):
    text = py_file.read_text(errors="ignore")
    if "np.float" in text or "np.int" in text or "np.bool" in text:
        text = re.sub(r"np\.float(?!\d)", "float", text)
        text = re.sub(r"np\.int(?!\d)", "int", text)
        text = re.sub(r"np\.bool(?!\w)", "bool", text)
        py_file.write_text(text)

build_marker = ALPHAPOSE_DIR / ".build_develop_done"
if not build_marker.exists():
    subprocess.run([sys.executable, "setup.py", "build", "develop", "--user"], cwd=ALPHAPOSE_DIR, check=True)
    build_marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
else:
    print("AlphaPose build already done")

try:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ALPHAPOSE_DIR, text=True).strip()
    print("AlphaPose commit:", commit)
except Exception as exc:
    print("Could not read AlphaPose commit:", exc)
print("AlphaPose ready at", ALPHAPOSE_DIR)

import gdown

POSE_MODEL_ID = "1kfyedqyn8exjbbNmYq8XGd2EooQjPtF9"
YOLO_MODEL_ID = "1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC"
REID_MODEL_ID = "1myNKfr2cXqiHZVXaaG8ZAq_U2UpeOLfG"

def download_drive_file(file_id, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        print("Already exists:", target)
        return target
    gdown.download(id=file_id, output=str(target), quiet=False)
    assert target.exists() and target.stat().st_size > 0, f"Download failed: {target}"
    return target

def resolve_tracker_weight_path():
    cfg_path = ALPHAPOSE_DIR / "trackers/tracker_cfg.py"
    fallback = ALPHAPOSE_DIR / "trackers/weights/osnet_x0_25_msmt17.pt"
    if not cfg_path.exists():
        return fallback
    text = cfg_path.read_text(errors="ignore")
    matches = re.findall(r"loadmodel\s*=\s*['\"]([^'\"]+)['\"]", text)
    if not matches:
        return fallback
    configured = matches[-1]
    configured = configured[2:] if configured.startswith("./") else configured
    configured_path = Path(configured)
    return configured_path if configured_path.is_absolute() else ALPHAPOSE_DIR / configured_path

POSE_CKPT = ALPHAPOSE_DIR / "pretrained_models/fast_421_res152_256x192.pth"
YOLO_WEIGHTS = ALPHAPOSE_DIR / "detector/yolo/data/yolov3-spp.weights"
YOLOX_X_WEIGHTS = ALPHAPOSE_DIR / "detector/yolox/data/yolox_x.pth"
REID_WEIGHTS = resolve_tracker_weight_path()
ALPHAPOSE_CFG = ALPHAPOSE_DIR / "configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml"

download_drive_file(POSE_MODEL_ID, POSE_CKPT)
download_drive_file(YOLO_MODEL_ID, YOLO_WEIGHTS)
if USE_YOLOX_DETECTOR:
    YOLOX_X_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    if YOLOX_X_WEIGHTS.exists() and YOLOX_X_WEIGHTS.stat().st_size > 0:
        print("Already exists:", YOLOX_X_WEIGHTS)
    else:
        subprocess.run(["wget", "-O", str(YOLOX_X_WEIGHTS), YOLOX_X_WEIGHTS_URL], check=True)
    assert YOLOX_X_WEIGHTS.exists() and YOLOX_X_WEIGHTS.stat().st_size > 0, f"YOLOX-X download failed: {YOLOX_X_WEIGHTS}"
download_drive_file(REID_MODEL_ID, REID_WEIGHTS)
assert ALPHAPOSE_CFG.exists(), f"AlphaPose config not found: {ALPHAPOSE_CFG}"

print("Config:", ALPHAPOSE_CFG)
print("Pose checkpoint:", POSE_CKPT)
print("YOLOv3 weights:", YOLO_WEIGHTS)
print("YOLOX-X weights:", YOLOX_X_WEIGHTS if USE_YOLOX_DETECTOR else "disabled")
print("ReID weights:", REID_WEIGHTS)

os.chdir(REPO_DIR)

print("Original STG-NF gen_data.py command shape has YOLOX commented out:")
print("python scripts/demo_inference.py --cfg <cfg> --checkpoint <ckpt> --outdir <outdir> --video/--indir <source> --sp --pose_track")
print("Paper-style command adds: --detector yolox-x")
print("Notebook USE_YOLOX_DETECTOR:", USE_YOLOX_DETECTOR)

commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ALPHAPOSE_DIR, text=True).strip()
print("\nAlphaPose commit:", commit)
print("Expected pinned commit:", ALPHAPOSE_COMMIT)
assert commit == ALPHAPOSE_COMMIT, f"AlphaPose commit mismatch: {commit} != {ALPHAPOSE_COMMIT}"

script_path = ALPHAPOSE_DIR / "scripts/demo_inference.py"
script_text = script_path.read_text(errors="ignore")
print("\nRelevant demo_inference.py argparse defaults:")
for line in script_text.splitlines():
    low = line.lower()
    if "add_argument" in line and any(token in low for token in ["conf", "thres", "nms", "detector", "qsize", "min_box", "posebatch", "detbatch", "pose_track", "sp"]):
        print(line.strip())

import yaml
cfg_data = yaml.safe_load(Path(ALPHAPOSE_CFG).read_text())

def get_nested(obj, dotted):
    cur = obj
    for part in dotted.split("."):
        cur = cur[part]
    return cur

print("\nAlphaPose detector settings from YAML config file:")
for key in ["DETECTOR.NAME", "DETECTOR.CONFIG", "DETECTOR.WEIGHTS", "DETECTOR.NMS_THRES", "DETECTOR.CONFIDENCE"]:
    print(f"{key}: {get_nested(cfg_data, key)}")

tracker_cfg = ALPHAPOSE_DIR / "trackers/tracker_cfg.py"
print("\nTracker settings:")
if tracker_cfg.exists():
    for line in tracker_cfg.read_text(errors="ignore").splitlines():
        low = line.lower()
        if any(token in low for token in ["loadmodel", "conf_thres", "nms_thres", "iou_thres"]):
            print(line.strip())
else:
    print("tracker_cfg.py not found")

example = command_for_source(train_sources[0], LOCAL_POSE_WORK / "settings_audit_example", source_mode=TRAIN_SOURCE_MODE) if "train_sources" in globals() and train_sources else None
if example:
    print("\nExample command:")
    print(" ".join(example))
    assert "--qsize" not in example
    if USE_YOLOX_DETECTOR:
        assert "--detector" in example and "yolox-x" in example
        assert YOLOX_X_WEIGHTS.exists() and YOLOX_X_WEIGHTS.stat().st_size > 0
print("\nStrict settings audit complete.")

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

def clip_id_from_source(source, source_mode):
    source = Path(source)
    return source.stem if source_mode == "video" else source.name

def list_video_sources(root):
    root = Path(root)
    assert root.exists(), f"Video root does not exist: {root}"
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS], key=lambda p: str(p))

def list_image_clip_dirs(root):
    root = Path(root)
    assert root.exists(), f"Image root does not exist: {root}"
    clip_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        if any(Path(name).suffix.lower() in IMAGE_EXTS for name in filenames):
            clip_dirs.append(Path(dirpath))
    return sorted(set(clip_dirs), key=lambda p: str(p))

def list_sources(root, source_mode):
    if source_mode == "video":
        return list_video_sources(root)
    if source_mode == "images":
        return list_image_clip_dirs(root)
    raise ValueError(f"Unknown SOURCE_MODE: {source_mode}")

train_sources = list_sources(TRAIN_SOURCE_ROOT, source_mode=TRAIN_SOURCE_MODE)
test_sources = list_sources(TEST_SOURCE_ROOT, source_mode=TEST_SOURCE_MODE)
assert train_sources, f"No train sources found under {TRAIN_SOURCE_ROOT}"
assert test_sources, f"No test sources found under {TEST_SOURCE_ROOT}"

print("Train sources:", len(train_sources))
print("Test sources:", len(test_sources))
print("First train sources:", [str(p) for p in train_sources[:5]])
print("First test sources:", [str(p) for p in test_sources[:5]])
print("First train clip IDs:", [clip_id_from_source(p, source_mode=TRAIN_SOURCE_MODE) for p in train_sources[:5]])
print("First test clip IDs:", [clip_id_from_source(p, source_mode=TEST_SOURCE_MODE) for p in test_sources[:5]])

from tqdm import tqdm

# Original STG-NF gen_data.py conversion behavior.
def convert_data_format(data, split="None"):
    if split == "testing":
        num_digits = 3
    elif split == "training":
        num_digits = 4
    elif split == "None":
        num_digits = 4
    else:
        num_digits = 4

    data_new = dict()
    for item in data:
        frame_idx_str = item["image_id"][:-4]
        frame_idx_str = frame_idx_str.zfill(num_digits)
        person_idx_str = str(item["idx"])
        keypoints = item["keypoints"]
        scores = item["score"]
        if person_idx_str not in data_new:
            data_new[person_idx_str] = {frame_idx_str: {"keypoints": keypoints, "scores": scores}}
        else:
            data_new[person_idx_str][frame_idx_str] = {"keypoints": keypoints, "scores": scores}
    return data_new

def raw_json_path(out_dir, clip_id):
    return Path(out_dir) / f"{clip_id}_alphapose-results.json"

def tracked_json_path(out_dir, clip_id):
    return Path(out_dir) / f"{clip_id}_alphapose_tracked_person.json"

def in_progress_path(out_dir, clip_id):
    return Path(out_dir) / f"{clip_id}.in_progress"

def manifest_path(out_dir):
    return Path(out_dir) / "pose_extraction_manifest.jsonl"

def is_json_readable(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with path.open("r") as handle:
            json.load(handle)
        return True
    except Exception:
        return False

def tracked_json_has_stg_format(path):
    path = Path(path)
    if not is_json_readable(path):
        return False
    with path.open("r") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        return False
    if not data:
        return True
    first_person = next(iter(data.values()))
    if not isinstance(first_person, dict) or not first_person:
        return False
    first_record = next(iter(first_person.values()))
    return isinstance(first_record, dict) and "keypoints" in first_record and "scores" in first_record

def append_manifest(out_dir, record):
    path = manifest_path(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")

def convert_raw_to_tracked(raw_path, tracked_path, split="None"):
    with Path(raw_path).open("r") as handle:
        data = json.load(handle)
    converted = convert_data_format(data, split=split)
    with Path(tracked_path).open("w") as handle:
        json.dump(converted, handle)
    assert tracked_json_has_stg_format(tracked_path), f"Bad tracked JSON after conversion: {tracked_path}"

def command_for_source(source, out_dir, source_mode):
    # Base command
    cmd = [
        sys.executable,
        "scripts/demo_inference.py",
        "--cfg", str(ALPHAPOSE_DIR / "configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml"),
        "--checkpoint", str(ALPHAPOSE_DIR / "pretrained_models/fast_421_res152_256x192.pth"),
        "--outdir", str(out_dir),
    ]

    # Detector settings
    if USE_YOLOX_DETECTOR:
        cmd.extend(["--detector", "yolox-x"])

    # Input source mode
    if source_mode == "video":
        cmd.extend(["--video", str(source)])
    elif source_mode == "images":
        cmd.extend(["--indir", str(source)])
    else:
        raise ValueError(f"Unknown source mode: {source_mode}")

    # STG-NF tracking flags
    cmd.extend(["--sp", "--pose_track"])

    # ADD THESE MEMORY EFFICIENCY FLAGS:
    # ----------------------------------
    # --detbatch 1 : Runs the YOLOX detector on 1 frame at a time
    # --posebatch 32 : Limits pose estimation batch size (default is often 64-80)
    # --qsize 10 : Prevents the frame queue from flooding system memory
    cmd.extend(["--detbatch", "1", "--posebatch", "32", "--qsize", "10"])

    return [str(x) for x in cmd]


def run_command_streamed(command, cwd, timeout=None):
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    started = time.time()
    last_line_at = started
    try:
        while True:
            line = process.stdout.readline()
            if line:
                print(line, end="")
                last_line_at = time.time()
            elif process.poll() is not None:
                break
            elif timeout is not None and time.time() - started > timeout:
                process.kill()
                raise TimeoutError(f"Command timed out after {timeout} seconds; last output {time.time() - last_line_at:.1f}s ago")
            else:
                time.sleep(0.2)
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)
    finally:
        if process.poll() is None:
            process.kill()

def extract_one_source(source, drive_out_dir, split, source_mode, timeout=None, force=False):
    source = Path(source)
    drive_out_dir = Path(drive_out_dir)
    drive_out_dir.mkdir(parents=True, exist_ok=True)
    clip_id = clip_id_from_source(source, source_mode)
    drive_raw = raw_json_path(drive_out_dir, clip_id)
    drive_tracked = tracked_json_path(drive_out_dir, clip_id)
    marker = in_progress_path(drive_out_dir, clip_id)

    if not force and tracked_json_has_stg_format(drive_tracked) and is_json_readable(drive_raw):
        print(f"[skip] {clip_id}: raw and tracked JSON already exist in Drive")
        append_manifest(drive_out_dir, {"clip_id": clip_id, "source": str(source), "status": "skipped_complete"})
        return "skipped"

    if not force and is_json_readable(drive_raw) and not tracked_json_has_stg_format(drive_tracked):
        print(f"[convert] {clip_id}: raw Drive JSON exists; creating tracked JSON")
        convert_raw_to_tracked(drive_raw, drive_tracked, split=split)
        append_manifest(drive_out_dir, {"clip_id": clip_id, "source": str(source), "status": "converted_from_drive_raw"})
        return "converted"

    local_out_dir = LOCAL_POSE_WORK / drive_out_dir.name / clip_id
    if local_out_dir.exists():
        shutil.rmtree(local_out_dir)
    local_out_dir.mkdir(parents=True, exist_ok=True)

    marker.write_text(json.dumps({"source": str(source), "started": time.strftime("%Y-%m-%d %H:%M:%S")}))
    command = command_for_source(source, local_out_dir, source_mode=source_mode)
    print(f"[run] {clip_id}")
    print("$", " ".join(command))
    started = time.time()
    try:
        run_command_streamed(command, cwd=ALPHAPOSE_DIR, timeout=timeout)
        local_raw = local_out_dir / "alphapose-results.json"
        assert is_json_readable(local_raw), f"AlphaPose did not create readable raw JSON for {clip_id}: {local_raw}"
        local_tracked = local_out_dir / f"{clip_id}_alphapose_tracked_person.json"
        convert_raw_to_tracked(local_raw, local_tracked, split=split)
        shutil.copy2(local_raw, drive_raw)
        shutil.copy2(local_tracked, drive_tracked)
        elapsed = time.time() - started
        append_manifest(drive_out_dir, {
            "clip_id": clip_id,
            "source": str(source),
            "status": "processed",
            "elapsed_sec": round(elapsed, 2),
            "raw": str(drive_raw),
            "tracked": str(drive_tracked),
        })
        print(f"[done] {clip_id}: {elapsed / 60:.1f} min; copied raw and tracked JSON to Drive")
        return "processed"
    except Exception as exc:
        append_manifest(drive_out_dir, {"clip_id": clip_id, "source": str(source), "status": "error", "error": repr(exc)})
        print(f"[error] {clip_id}: {exc}")
        raise
    finally:
        shutil.rmtree(local_out_dir / "poseflow", ignore_errors=True)
        if marker.exists() and tracked_json_has_stg_format(drive_tracked):
            marker.unlink()

def extract_sources(sources, drive_out_dir, split, source_mode, limit=None, start=0, timeout=None, force=False, continue_on_error=True):
    selected = list(sources)[start:]
    if limit is not None:
        selected = selected[:limit]
    counts = {"processed": 0, "converted": 0, "skipped": 0, "error": 0}
    for source in tqdm(selected, desc=f"extract -> {Path(drive_out_dir).name}"):
        try:
            result = extract_one_source(source, drive_out_dir, split=split, source_mode=source_mode, timeout=timeout, force=force)
            counts[result] = counts.get(result, 0) + 1
        except Exception as exc:
            counts["error"] += 1
            if not continue_on_error:
                raise
            print(f"[continue] {source}: {exc}")
    print("Extraction counts:", counts)
    return counts

print("Extraction helpers ready. Default command follows paper-style YOLOX + pose tracking, with no --qsize and no custom thresholds.")
SMOKE_TIMEOUT = None  # Set seconds if you want a hard timeout.

smoke_train_source = train_sources[0]
smoke_test_source = test_sources[0]
print("Smoke train:", smoke_train_source, "->", clip_id_from_source(smoke_train_source, source_mode=TRAIN_SOURCE_MODE))
print("Smoke test:", smoke_test_source, "->", clip_id_from_source(smoke_test_source, source_mode=TEST_SOURCE_MODE))

extract_one_source(smoke_train_source, DRIVE_POSE_TRAIN, split="None", source_mode=TRAIN_SOURCE_MODE, timeout=SMOKE_TIMEOUT)
extract_one_source(smoke_test_source, DRIVE_POSE_TEST, split="None", source_mode=TEST_SOURCE_MODE, timeout=SMOKE_TIMEOUT)

for source, out_dir, mode in [(smoke_train_source, DRIVE_POSE_TRAIN, TRAIN_SOURCE_MODE), (smoke_test_source, DRIVE_POSE_TEST, TEST_SOURCE_MODE)]:
    clip_id = clip_id_from_source(source, source_mode=mode)
    assert is_json_readable(raw_json_path(out_dir, clip_id)), raw_json_path(out_dir, clip_id)
    assert tracked_json_has_stg_format(tracked_json_path(out_dir, clip_id)), tracked_json_path(out_dir, clip_id)
    print("validated", clip_id)

BATCH_LIMIT = None  # Set an integer for smaller Colab sessions.
START_AT = 0
EXTRACTION_TIMEOUT = None
CONTINUE_ON_ERROR = True

extract_sources(train_sources, DRIVE_POSE_TRAIN, split="None", source_mode=TRAIN_SOURCE_MODE, limit=BATCH_LIMIT, start=START_AT, timeout=EXTRACTION_TIMEOUT, continue_on_error=CONTINUE_ON_ERROR)
extract_sources(test_sources, DRIVE_POSE_TEST, split="None", source_mode=TEST_SOURCE_MODE, limit=BATCH_LIMIT, start=START_AT, timeout=EXTRACTION_TIMEOUT, continue_on_error=CONTINUE_ON_ERROR)

def output_clip_ids(out_dir, suffix):
    out_dir = Path(out_dir)
    return {p.name[:-len(suffix)] for p in out_dir.glob(f"*{suffix}")}

def verify_outputs(sources, out_dir, source_mode):
    expected = {clip_id_from_source(s, source_mode=source_mode) for s in sources}
    raw_ids = output_clip_ids(out_dir, "_alphapose-results.json")
    tracked_ids = output_clip_ids(out_dir, "_alphapose_tracked_person.json")
    missing_raw = sorted(expected - raw_ids)
    missing_tracked = sorted(expected - tracked_ids)
    extra_raw = sorted(raw_ids - expected)
    extra_tracked = sorted(tracked_ids - expected)
    print(out_dir)
    print("  expected:", len(expected))
    print("  raw:", len(raw_ids), "tracked:", len(tracked_ids))
    print("  missing raw:", missing_raw[:20], "count=", len(missing_raw))
    print("  missing tracked:", missing_tracked[:20], "count=", len(missing_tracked))
    print("  extra raw:", extra_raw[:20], "count=", len(extra_raw))
    print("  extra tracked:", extra_tracked[:20], "count=", len(extra_tracked))
    assert not missing_raw, f"Missing raw JSONs: {missing_raw[:20]}"
    assert not missing_tracked, f"Missing tracked JSONs: {missing_tracked[:20]}"
    assert not extra_raw, f"Unexpected raw JSONs: {extra_raw[:20]}"
    assert not extra_tracked, f"Unexpected tracked JSONs: {extra_tracked[:20]}"

verify_outputs(train_sources, DRIVE_POSE_TRAIN, source_mode=TRAIN_SOURCE_MODE)
verify_outputs(test_sources, DRIVE_POSE_TEST, source_mode=TEST_SOURCE_MODE)
print("All expected raw and tracked pose JSONs are present in Drive.")

print("Original STG-NF gen_data.py command has YOLOX commented out:")
print("python scripts/demo_inference.py --cfg <cfg> --checkpoint <ckpt> --outdir <outdir> --video/--indir <source> --sp --pose_track")
print("Paper-style STG-NF extraction adds --detector yolox-x.")
print("This notebook command example:")
example = command_for_source(train_sources[0], LOCAL_POSE_WORK / "example", source_mode=TRAIN_SOURCE_MODE)
print(" ".join(example))
#assert "--qsize" not in example
#print("No extra qsize or threshold flags are used. YOLOX is controlled by USE_YOLOX_DETECTOR.")

os.chdir(REPO_DIR)

def replace_if_present(path, old, new):
    path = Path(path)
    text = path.read_text()
    if old in text:
        path.write_text(text.replace(old, new))
        print("patched", path)
    else:
        print("already ok", path)

replace_if_present("dataset.py", "dtype=np.int)", "dtype=int)")
replace_if_present("utils/pose_utils.py", ".astype(np.int)", ".astype(int)")
replace_if_present("utils/pose_utils.py", "plt.style.use('seaborn-ticks')", "plt.style.use('seaborn-v0_8-ticks')")

training_path = Path("models/training.py")
training_text = training_path.read_text()
if "weights_only=False" not in training_text and "checkpoint = torch.load(filename)" in training_text:
    training_text = training_text.replace(
        "            checkpoint = torch.load(filename)",
        "            try:\n                checkpoint = torch.load(filename, map_location=self.args.device, weights_only=False)\n            except TypeError:\n                checkpoint = torch.load(filename, map_location=self.args.device)",
    )
    training_path.write_text(training_text)
    print("patched", training_path)
else:
    print("already ok", training_path)

print("Compatibility patch step complete")
os.chdir(REPO_DIR)

# Reload repo modules after compatibility patches.
for name in list(sys.modules):
    if name == "dataset" or name == "args" or name.startswith("utils"):
        del sys.modules[name]

from args import init_parser, init_sub_args
from dataset import get_dataset_and_loader
from utils.data_utils import trans_list

argv = [
    "--dataset", "ShanghaiTech",
    "--pose_path_train", str(DRIVE_POSE_TRAIN),
    "--pose_path_test", str(DRIVE_POSE_TEST),
    "--vid_path_train", str(TRAIN_SOURCE_ROOT),
    "--vid_path_test", str(TEST_SOURCE_ROOT),
    "--batch_size", "2",
    "--num_workers", "0",
    "--specific_clip", "0",
]
args = init_parser().parse_args(argv)
args, _ = init_sub_args(args)
dataset, loader = get_dataset_and_loader(args, trans_list=trans_list, only_test=False)
train_batch = next(iter(loader["train"]))
test_batch = next(iter(loader["test"]))
print("Train sample shape:", dataset["train"][0][0].shape)
print("Test sample shape:", dataset["test"][0][0].shape)
print("Train batch pose shape:", train_batch[0].shape)
print("Test batch pose shape:", test_batch[0].shape)
BATCH_SIZE = 256
NUM_WORKERS = 2

os.chdir(REPO_DIR)
!python train_eval.py \
  --dataset ShanghaiTech \
  --pose_path_train {DRIVE_POSE_TRAIN} \
  --pose_path_test {DRIVE_POSE_TEST} \
  --vid_path_train {TRAIN_SOURCE_ROOT} \
  --vid_path_test {TEST_SOURCE_ROOT} \
  --exp_dir {DRIVE_LOG_DIR} \
  --epochs 1 \
  --batch_size 256 \
  --num_workers {NUM_WORKERS}
os.chdir(REPO_DIR)
!python train_eval.py \
  --dataset ShanghaiTech \
  --pose_path_train {DRIVE_POSE_TRAIN} \
  --pose_path_test {DRIVE_POSE_TEST} \
  --vid_path_train {TRAIN_SOURCE_ROOT} \
  --vid_path_test {TEST_SOURCE_ROOT} \
  --exp_dir {DRIVE_LOG_DIR} \
  --batch_size 256 \
  --num_workers 2

