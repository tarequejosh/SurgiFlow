import os
import cv2
import json
import glob
from ultralytics import YOLO

from tool_pose import SCRIPT_DIR, find_best_model, extract_tool_poses, draw_tool_pose, pixel_to_tray

def run_pipeline_demo(num_samples=5):
    print("=" * 70)
    print("      SURGIFLOW PERCEPTION: FULL PIPELINE END-TO-END DEMO       ")
    print("=" * 70)

    model_path = find_best_model()
    print(f"Loading Model Weights from: {model_path}")
    model = YOLO(model_path)

    # Path resolution robust to invocation directory
    val_images_dir = os.path.join(SCRIPT_DIR, "datasets/surgical_tools/synthetic_yolo/val/images")
    output_dir = os.path.join(SCRIPT_DIR, "demo_outputs")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(val_images_dir):
        raise FileNotFoundError(f"Validation images not found at {val_images_dir}")

    # Sorted so the demo picks the same images on every OS / filesystem
    all_images = sorted(glob.glob(os.path.join(val_images_dir, "*.jpg")))
    if not all_images:
        raise FileNotFoundError(f"No .jpg images found in {val_images_dir}")

    sample_images = all_images[:num_samples]
    print(f"Selected {len(sample_images)} validation images for evaluation.")
    print("=" * 70)

    summary_results = []

    for img_idx, img_path in enumerate(sample_images):
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Could not read {img_path}, skipping.")
            continue
        h, w = img.shape[:2]

        # Run inference (imgsz is taken from the checkpoint, e.g. 1152 for m_1152)
        results = model.predict(source=img, conf=0.30, verbose=False)[0]

        annotated = img.copy()
        detected_in_image = []

        for tool in extract_tool_poses(results):
            draw_tool_pose(annotated, tool)

            # Tool centre: on the instrument centre line if keypoints exist, else bbox centre
            cx, cy = tool["axis_center"] or tool["bbox_center"]
            angle_deg = tool["angle_deg"] if tool["angle_deg"] is not None else 0.0
            detected_in_image.append({
                "tool": tool["name"],
                "confidence": tool["confidence"],
                "centroid_px": [int(round(cx)), int(round(cy))],
                "normalized_pos": [round(cx / w, 3), round(cy / h, 3)],
                # Direction the tool points (handle -> tip), image axes (x right, y down)
                "angle_degrees": round(angle_deg, 1)
            })

        out_img_path = os.path.join(output_dir, f"demo_result_{img_idx+1}_{img_name}")
        cv2.imwrite(out_img_path, annotated)

        print(f"\n[Image {img_idx+1}/{len(sample_images)}] {img_name}")
        print(f" -> Saved visualization: {out_img_path}")
        print(f" -> Tools detected: {len(detected_in_image)}")
        for d in detected_in_image:
            print(f"    * {d['tool'].upper()} (Conf: {d['confidence']*100:.1f}%) | Pos: {d['normalized_pos']} | Angle: {d['angle_degrees']}°")

        summary_results.append({
            "image": img_name,
            "image_size": [w, h],
            "output_path": out_img_path,
            "detections": detected_in_image
        })

    # Save JSON summary
    summary_json_path = os.path.join(output_dir, "demo_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_results, f, indent=4)

    print("\n" + "=" * 70)
    print("                     DEMO COMPLETED SUCCESSFULLY!                     ")
    print(f"Annotated Visuals Saved To: {output_dir}/")
    print(f"Structured JSON Exported:   {summary_json_path}")
    print("=" * 70)

    # Demonstrate Bridging into SurgiFlow
    print("\n--- SURGIFLOW STATE BRIDGING DEMONSTRATION ---")
    print("In surgical_tray_env.py, SurgiFlow expects tool observations:")
    print("Currently: [obj_pos, obj_rot, obj_velp, obj_velr, one_hot_state]")
    print("\nVision Module translates camera pixels -> Storage Tray plane (idealised top-down camera;")
    print("dataset images are not top-down tray views, so these poses are illustrative only):")
    for res in summary_results:
        w, h = res["image_size"]
        for d in res["detections"]:
            (sim_x, sim_y, sim_z), yaw = pixel_to_tray(*d["centroid_px"], w, h, angle_deg=d["angle_degrees"])
            print(f" Tool: {d['tool']:<18} | Vision 2D: {d['normalized_pos']} -> MuJoCo 3D Pose: "
                  f"[{sim_x:.3f}, {sim_y:.3f}, {sim_z:.3f}], yaw {yaw:.1f}°")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline_demo()
