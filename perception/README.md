# SurgiFlow Perception: Surgical Instrument Detection, Pose Estimation & Tray Management

This module implements the computer vision perception layer for surgical instrument identification, 12-keypoint topological pose estimation, and spatial coordinate mapping to bridge real-world cameras into the **SurgiFlow** MuJoCo reinforcement learning environment.

---

## 1. Quickstart

### Step 1: Run the End-to-End Perception Demo
To immediately run inference across validation surgical tray images and see 2D $\to$ 3D MuJoCo pose mapping:
```bash
python perception/run_full_pipeline_demo.py
# Or if you are inside the perception/ directory:
python run_full_pipeline_demo.py
```
This saves annotated detection images to `perception/demo_outputs/` and exports structured spatial coordinates to `perception/demo_outputs/demo_summary.json`.

### Step 2: Test on Any Custom Tray Image
```bash
python perception/predict_and_extract_pose.py --image path/to/your_image.jpg
```
The script auto-detects the best trained YOLO weights and plots bounding boxes, the 12-keypoint skeleton (both tool edges), the tool centre and its pointing direction (handle → tip). Shared post-processing lives in `perception/tool_pose.py`.

### Step 3: Train / Fine-Tune on Your Local GPU
To train the YOLO pose & detection model on the 17,258-image dataset:
```bash
python perception/train_segmentation.py --epochs 30 --batch 16 --device 0
```
This automatically produces:
* Model checkpoints (`best.pt`, `last.pt` in `perception/runs/train/surgical_tool_model/weights/`)
* Publication-ready evaluation plots: Confusion Matrix, Precision-Recall Curve, and F1 Curve.
* Metrics on both the `val` split (used for early stopping / `best.pt` selection) and the held-out `test` split — report the `test` numbers.

---

## 2. Dataset Overview

* **Benchmark:** SurgeoNet Surgical Instruments Dataset (GCPR 2024 / DFKI)
* **Dataset Config:** `perception/data.yaml`
* **Total Volume:** 17,258 high-resolution annotated images
  * `Train`: 13,786 images
  * `Validation`: 1,709 images
  * `Test`: 1,763 images
* **13 Clinical Classes:**
  `Scalpel`, `Metzenbaum Scissor`, `Surgical Scissor`, `Overholt Clamp`, `Needle Holder`, `Surgical Forceps`, `Atraumatic Forceps`, `Retractor`, `Hook`, `Ligament Clamp`, `Peripheral Clamp`, `Bowl`, `Tong`.
* **Topology:** 12 2D keypoints = 6 left/right edge pairs along the instrument, from the tip (`0, 1`) to the handle / finger rings (`10, 11`). Orientation is computed from the pair midpoints (the tool centre line), not from individual edge points.

---

## 3. Bridging Perception to SurgiFlow Reinforcement Learning

The perception module exports JSON coordinates (`demo_summary.json`):
```json
{
    "tool": "Sur. Scissor",
    "confidence": 0.98,
    "centroid_px": [606, 898],
    "normalized_pos": [0.526, 0.78],
    "angle_degrees": 177.3
}
```
* `centroid_px` is the mean of the keypoint-pair midpoints, so it lies on the instrument (the bbox centre of an opened clamp often does not).
* `angle_degrees` is the direction the tool points (handle → tip) in image axes (x right, y down, so positive is clockwise on screen).

In `surgical_tray_env.py`, SurgiFlow expects tool observations:
`[obj_pos, obj_rot, obj_velp, obj_velr, one_hot_state]`

`tool_pose.pixel_to_tray()` maps image points onto the Storage Tray plane of `surgical_env.xml` (centre $(-0.5, 0.2)$, inner area $0.22 \times 0.28$ m, tool resting height $z = 0.81$ m). It assumes an idealised top-down camera: aspect ratio is preserved, image-up is world $+y$, and the image yaw is negated accordingly:
$$\text{Vision 2D: } [0.526, 0.780] \implies \text{MuJoCo 3D Pose: } [-0.494, 0.139, 0.810],\ \text{yaw } -177.3^\circ$$

> **Note:** the SurgeoNet images are perspective views of desks, not top-down tray views, so the 3D poses printed by the demo are illustrative only. For a real camera, replace the idealised model with a calibrated homography (e.g. `cv2.findHomography` from the four tray corners) and pass the live `site_tray_storage` position, since the env randomises table positions by ±5 cm at reset.

This replaces the privileged simulator state vectors with vision perception, transitioning SurgiFlow into a true **Vision-Guided Robotic Scrub Nurse**.
