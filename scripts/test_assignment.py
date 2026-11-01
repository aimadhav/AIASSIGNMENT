"""Local evaluator; simulator ground truth is used only here, never by the solution."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "A4" / "A4" / "src"
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".tools" / "ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
sys.path.insert(0, str(SRC))
os.chdir(str(SRC))

import numpy as np
import cv2
import open3d as o3d
import pybullet as bullet
import torch
from base import Environment
from control import Manipulator
from perception import Perception, DepthProcessor
from planning import Planner

torch.set_num_threads(2)


class TestClient:
    def __init__(self):
        self.bodies = []

    def connect(self, mode):
        return bullet.connect(bullet.DIRECT)

    def loadURDF(self, filename, *args, **kwargs):
        body = bullet.loadURDF(filename, *args, **kwargs)
        self.bodies.append((str(filename), body))
        return body

    def __getattr__(self, name):
        if name in {"getBasePositionAndOrientation", "getAABB", "getVisualShapeData", "getCollisionShapeData"}:
            raise AssertionError("Privileged simulator access by solution: " + name)
        return getattr(bullet, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["default", "moved", "near", "close", "partial"], default="default")
    parser.add_argument("--perception", action="store_true")
    parser.add_argument("--saved", action="store_true", help="Reuse sensed clusters for grasp debugging")
    args = parser.parse_args()
    state = json.loads((SRC / "init_state.json").read_text())
    if args.case == "moved":
        state["shelf_position"] = [0.2, 0.45, 0.05]
        locations = list(state["objects"].values())[:4]
        for label, location in zip(list(state["objects"])[:4], locations[1:] + locations[:1]):
            state["objects"][label] = location
    elif args.case == "near":
        state["objects"]["lemon"] = [0.08, -0.17, 0.07]
        state["objects"]["wood_block"] = [0.5, 0.23, 0.12]
    elif args.case == "close":
        state["objects"].update({"lemon": [0.38, -0.08, 0.07], "plum": [0.46, -0.16, 0.08], "wood_block": [0.5, 0.23, 0.12]})
    elif args.case == "partial":
        state["objects"]["plum"] = [0.14, 0.54, 0.32]
    output = ROOT / ".tools" / "assignment-checks" / args.case
    output.mkdir(parents=True, exist_ok=True)
    client = TestClient()
    started = time.monotonic()
    result = {
        "case": args.case, "saved_geometry": args.saved,
        "source_hashes": {name: hashlib.sha256((SRC / name).read_bytes()).hexdigest() for name in ("control.py", "perception.py", "planning.py")},
    }
    try:
        environment = Environment(client, state)
        arm = Manipulator(client, environment.get_panda_id())
        if args.case == "moved":
            client.loadURDF("./assets/Collection/orange.urdf", [0.41, -0.02, 0.08], globalScaling=0.1)
        perceptor = Perception(arm)
        detect = perceptor.obj_detector.detect_objects
        view_number = [0]
        def log_detection(rgb):
            labels, boxes = detect(rgb)
            cv2.imwrite(str(output / ("view-%02d.png" % view_number[0])), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            view_number[0] += 1
            return labels, boxes
        perceptor.obj_detector.detect_objects = log_detection
        processor = DepthProcessor()
        if args.saved:
            labels = json.loads((output / "objects.json").read_text())
            objects = {label: o3d.io.read_point_cloud(str(output / ("object-" + label + ".ply"))) for label in labels}
            arm.move_home()
        else:
            print("Scanning", args.case, flush=True)
            observations = perceptor.exlpore_scene(perceptor.generate_exploration_poses())
            print("Aligning", [len(p.points) for p in observations], flush=True)
            merged = processor.register_pointclouds(observations)
            o3d.io.write_point_cloud(str(output / "merged.ply"), merged)
            remaining = processor.remove_planes(merged)
            o3d.io.write_point_cloud(str(output / "segmented.ply"), remaining)
            clusters = processor.cluster_objects(remaining)
            print("Clusters", [(len(c.points), np.asarray(c.points).mean(0).round(3).tolist()) for c in clusters], flush=True)
            objects = perceptor.identify_objects(clusters)
            for label, cloud in objects.items():
                o3d.io.write_point_cloud(str(output / ("object-" + label + ".ply")), cloud)
            (output / "objects.json").write_text(json.dumps(list(objects)))
        result["objects"] = {
            label: {"points": len(cloud.points), "center": np.median(np.asarray(cloud.points), axis=0).tolist(), "pick": perceptor.get_object_pick_pose(label, objects).tolist()}
            for label, cloud in objects.items()
        }
        print("Objects", result["objects"], flush=True)
        planner = Planner(arm)
        plan = planner.get_clear_task_plan(objects)
        print("Plan", [(name, [np.asarray(a).round(3).tolist() for a in values]) for name, values in plan], flush=True)
        if not args.perception:
            for action, arguments in plan:
                print("Executing", action, [np.asarray(a).round(3).tolist() for a in arguments], flush=True)
                getattr(arm, action)(*arguments)
            arm._step(360)
            # Evaluator-only oracle: not exposed to perception, control, or planning.
            plate = next(body for name, body in client.bodies if name.endswith("plate.urdf"))
            destination = np.array(bullet.getBasePositionAndOrientation(plate)[0])
            placements = []
            for name, body in client.bodies:
                if "/Collection/" not in name or name.endswith(("plate.urdf", "small_shelf.urdf", "wood_block.urdf")):
                    continue
                location = np.array(bullet.getBasePositionAndOrientation(body)[0])
                placed = np.linalg.norm(location[:2] - destination[:2]) < 0.115 and location[2] > destination[2] - 0.02
                placements.append({"object": Path(name).stem, "position": location.tolist(), "on_plate": bool(placed)})
            result["placements"] = placements
            result["placed"] = sum(p["on_plate"] for p in placements)
            result["total"] = len(placements)
            if result["placed"] != result["total"]:
                raise AssertionError("Only %d/%d fruits reached the plate" % (result["placed"], result["total"]))
        result["status"] = "passed"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = repr(error)
        import traceback
        traceback.print_exc()
    finally:
        result["seconds"] = time.monotonic() - started
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        if bullet.isConnected():
            bullet.disconnect()
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
