import os
import cv2
import json
import argparse
from ultralytics import YOLO

from tool_pose import SCRIPT_DIR, find_best_model, extract_tool_poses, draw_tool_pose

def analyze_and_extract_tool_poses(model_path=None, image_path=None, output_dir=None, conf_thresh=0.35):
    """
    Runs YOLO inference on a surgical tray image, extracts 2D tool locations,
    keypoints, and orientations for each detected instrument.

    This bridges Computer Vision to the SurgiFlow Robotic Control loop.
    """
    if model_path is None or not os.path.exists(model_path):
        model_path = find_best_model()

    if output_dir is None:
        output_dir = os.path.join(SCRIPT_DIR, "demo_outputs")

    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)

    # Resolve image path
    if not os.path.exists(image_path):
        # Try resolving relative to SCRIPT_DIR
        rel_img = os.path.join(SCRIPT_DIR, image_path)
        if os.path.exists(rel_img):
            image_path = rel_img
        else:
            raise FileNotFoundError(f"Could not find image at {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    orig_h, orig_w = img.shape[:2]

    # Run Inference (imgsz is taken from the checkpoint, e.g. 1152 for m_1152)
    results = model.predict(source=img, conf=conf_thresh, save=False, verbose=False)[0]

    detected_tools = []
    annotated_img = img.copy()

    for tool in extract_tool_poses(results):
        draw_tool_pose(annotated_img, tool)

        # Tool centre: on the instrument centre line if keypoints exist, else bbox centre
        cx, cy = tool["axis_center"] or tool["bbox_center"]
        detected_tools.append({
            "instrument": tool["name"],
            "confidence": tool["confidence"],
            "centroid_2d_pixels": [int(round(cx)), int(round(cy))],
            "normalized_centroid": [cx / orig_w, cy / orig_h],
            # Direction the tool points (handle -> tip), image axes (x right, y down)
            "orientation_degrees": tool["angle_deg"] if tool["angle_deg"] is not None else 0.0,
            "bbox_xyxy": tool["bbox_xyxy"],
            # All 12 keypoints in fixed semantic order: pairs (0,1) tip ... (10,11) handle
            "keypoints": tool["keypoints"] or []
        })

    # Save output visualization
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_vis_path = os.path.join(output_dir, f"{base_name}_detected.png")
    out_json_path = os.path.join(output_dir, f"{base_name}_tool_poses.json")

    cv2.imwrite(out_vis_path, annotated_img)
    with open(out_json_path, "w") as f:
        json.dump(detected_tools, f, indent=4)

    print("\n" + "=" * 60)
    print(f"RESULTS FOR: {os.path.basename(image_path)}")
    print(f"Model: {model_path}")
    print(f"Detected Instruments: {len(detected_tools)}")
    for t in detected_tools:
        print(f" -> [{t['instrument'].upper()}] Conf: {t['confidence']*100:.1f}% | Centroid: {t['centroid_2d_pixels']} | Angle: {t['orientation_degrees']:.1f}°")
    print(f"\nSaved visualization to: {out_vis_path}")
    print(f"Saved pose metadata to:  {out_json_path}")
    print("=" * 60)

    return detected_tools, out_vis_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 2D Spatial Poses from Surgical Tray Image")
    parser.add_argument("--model", type=str, default=None, help="Path to trained YOLO model .pt (auto-finds best if None)")
    parser.add_argument("--image", type=str, required=True, help="Path to input surgical tray image")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")

    args = parser.parse_args()
    analyze_and_extract_tool_poses(
        model_path=args.model,
        image_path=args.image,
        output_dir=args.output_dir,
        conf_thresh=args.conf
    )
