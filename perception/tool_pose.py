"""
Shared helpers for turning SurgeoNet YOLO-pose detections into tool poses.

Keypoint topology (12 x 2D, see data.yaml `kpt_shape: [12, 2]`):
the keypoints come in 6 left/right pairs running along the instrument,
from the tip (indices 0, 1) to the handle / finger rings (indices 10, 11).
The midpoints of these pairs trace the tool's centre line, so orientation is
computed from pair midpoints rather than from individual edge points (which,
for open scissors/clamps, can be almost perpendicular to the tool axis).
"""

import os
import glob
import numpy as np
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

NUM_KPTS = 12
TIP_PAIR = 0      # pair index of keypoints (0, 1)
HANDLE_PAIR = 5   # pair index of keypoints (10, 11)

# Storage tray geometry from surgical_env.xml (table_storage + tray_storage bodies).
# Tables are randomized by +/-5 cm at reset, so pass the live `site_tray_storage`
# position as `tray_center_xy` when bridging into a running simulation.
TRAY_CENTER_XY = (-0.5, 0.2)
TRAY_INNER_HALF_EXTENTS = (0.11, 0.14)   # inside of the tray walls (x, y)
TRAY_SURFACE_Z = 0.80                    # tray geoms are visual only; tools rest on the table top
TOOL_HALF_HEIGHT = 0.01                  # sim tool geoms are 2 cm thick


def find_best_model():
    """Locates the best available surgical YOLO model weights, robust to cwd."""
    candidate_paths = [
        os.path.join(SCRIPT_DIR, "weights/runs/pose/m_1152/weights/best.pt"),
        os.path.join(SCRIPT_DIR, "weights/runs/pose/s_640/weights/best.pt"),
        os.path.join(SCRIPT_DIR, "weights/yolo/runs/pose/m_1152/weights/best.pt"),
        os.path.join(SCRIPT_DIR, "weights/yolo/runs/pose/s_640/weights/best.pt"),
        os.path.join(SCRIPT_DIR, "runs/train/surgical_tool_model/weights/best.pt"),
    ]
    candidate_paths.extend(sorted(glob.glob(os.path.join(SCRIPT_DIR, "weights/**/best.pt"), recursive=True)))

    for path in candidate_paths:
        if os.path.exists(path):
            return os.path.abspath(path)

    raise FileNotFoundError(
        "No trained surgical YOLO-pose weights found. Run download_pretrained_models.py "
        "or train_segmentation.py first (the generic COCO 'yolov8s-pose.pt' cannot "
        "detect surgical instruments)."
    )


def tool_axis(kpts):
    """
    Returns (angle_deg, center_xy) of the tool's centre line in image coordinates.

    angle_deg is the direction the tool points (handle -> tip), measured with
    image axes (x right, y down), i.e. positive angles are clockwise on screen.
    center_xy is the mean of the pair midpoints, which lies on the instrument
    itself (unlike the bbox centre of a diagonal, opened clamp).
    """
    mids = np.asarray(kpts, dtype=float).reshape(NUM_KPTS // 2, 2, 2).mean(axis=1)
    d = mids[TIP_PAIR] - mids[HANDLE_PAIR]
    angle_deg = float(np.degrees(np.arctan2(d[1], d[0])))
    return angle_deg, mids.mean(axis=0)


def extract_tool_poses(result, skip_background=True):
    """Converts one ultralytics pose Result into a list of per-tool dicts."""
    tools = []
    if result.boxes is None or len(result.boxes) == 0:
        return tools

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    kpts_all = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else None

    for idx in range(len(boxes)):
        name = str(result.names.get(cls_ids[idx], f"class_{cls_ids[idx]}"))
        if skip_background and name.lower() == "background":
            continue

        x1, y1, x2, y2 = boxes[idx]
        tool = {
            "name": name,
            "confidence": float(confs[idx]),
            "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "bbox_center": [float((x1 + x2) / 2), float((y1 + y2) / 2)],
            "keypoints": None,
            "axis_center": None,
            "angle_deg": None,
        }
        # Keep all 12 keypoints in their semantic order. With 2D keypoints there is no
        # visibility score; points predicted outside the frame are clipped to the border
        # by ultralytics, so they must not be filtered by "> 0" (that shifts indices).
        if kpts_all is not None and idx < len(kpts_all) and kpts_all[idx].shape[0] == NUM_KPTS:
            kpts = kpts_all[idx]
            angle_deg, center = tool_axis(kpts)
            tool["keypoints"] = kpts.astype(float).tolist()
            tool["axis_center"] = center.tolist()
            tool["angle_deg"] = angle_deg
        tools.append(tool)
    return tools


def tool_color(name):
    name = name.lower()
    if "scalpel" in name:
        return (0, 230, 115)
    if "scissor" in name:
        return (240, 120, 40)
    return (70, 170, 255)


def draw_tool_pose(img, tool, axis_len=40):
    """Draws bbox, keypoint skeleton (both tool edges), centre point, pointing axis and label."""
    color = tool_color(tool["name"])
    x1, y1, x2, y2 = tool["bbox_xyxy"]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    if tool["keypoints"] is not None:
        kpts = np.asarray(tool["keypoints"])
        for side in (0, 1):  # left edge: 0,2,..,10 / right edge: 1,3,..,11
            chain = kpts[side::2].astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [chain], False, color, 1, cv2.LINE_AA)
        for i, (kx, ky) in enumerate(kpts):
            dot = (0, 0, 255) if i // 2 == TIP_PAIR else (0, 255, 255)
            cv2.circle(img, (int(kx), int(ky)), 3, dot, -1)

        cx, cy = tool["axis_center"]
        theta = np.radians(tool["angle_deg"])
        end = (int(cx + axis_len * np.cos(theta)), int(cy + axis_len * np.sin(theta)))
        cv2.circle(img, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        cv2.arrowedLine(img, (int(cx), int(cy)), end, (0, 0, 255), 2, tipLength=0.3)
    else:
        cx, cy = tool["bbox_center"]
        cv2.circle(img, (int(cx), int(cy)), 5, (0, 0, 255), -1)

    label = f"{tool['name']} {tool['confidence'] * 100:.1f}%"
    org = (x1, max(y1 - 8, 18))
    cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def pixel_to_tray(u, v, img_w, img_h, angle_deg=None, tray_center_xy=TRAY_CENTER_XY,
                  tray_half_extents=TRAY_INNER_HALF_EXTENTS):
    """
    Maps an image point (and optional image-space angle) onto the MuJoCo tray plane.

    Assumes an idealised top-down camera centred over the tray, image x -> world +x,
    image up -> world +y, with the image scaled uniformly (aspect preserved) to fit
    inside the tray walls. A real camera needs a calibrated homography
    (e.g. cv2.findHomography from the 4 tray corners) instead of this model.

    Returns (xyz, yaw_deg): world position of the tool centre resting on the tray and
    the world-frame yaw of the tool's pointing direction (None if angle_deg is None).
    """
    scale = min(2 * tray_half_extents[0] / img_w, 2 * tray_half_extents[1] / img_h)  # m / px
    x = tray_center_xy[0] + (u - img_w / 2) * scale
    y = tray_center_xy[1] - (v - img_h / 2) * scale  # image v grows downward, world y upward
    z = TRAY_SURFACE_Z + TOOL_HALF_HEIGHT
    yaw_deg = None if angle_deg is None else -angle_deg  # y-axis flip mirrors the angle
    return [x, y, z], yaw_deg
