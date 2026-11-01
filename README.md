# AIASSIGNMENT

RGB-D perception and fruit pick-and-place with a Panda arm in PyBullet.

The robot scans the table, reconstructs a point cloud, removes support planes,
clusters objects, associates YOLO detections with 3D geometry, and places fruit
on the perceived plate. Object positions come from the supplied RGB-D sensor,
not simulator ground-truth queries.

## Run

Use Linux with a display, or Ubuntu under WSL2 with WSLg. Install the supplied
environment without changing its package pins:

```bash
cd A4/A4
conda env create -f a4_2026.yml
conda activate a4_2026
cd src
python main.py
```

The YAML includes the original author's machine-specific `prefix`. If necessary,
pass an explicit installation prefix to Conda instead of editing package pins.
Run `main.py` from `src` so the provided asset paths resolve correctly. The
checkpoint is included at `src/weights/best.pt`; inference runs on CPU offline.

## Implementation

Only the three designated starter files implement the solution:

- `control.py`: bounded joint trajectories, inverse kinematics, gripper control,
  and pick/place execution with explicit simulation stepping.
- `perception.py`: RGB-D back-projection, frame transforms, conservative ICP,
  RANSAC plane removal, DBSCAN, and depth-consistent multiview label association.
- `planning.py`: identifies the destination plate, skips fruit already on it,
  and distributes placement locations within its measured footprint.

Repeated fruit classes keep separate instance keys. A tilted approach handles
the arm's near-base reach limit. The wooden block and shelf are obstacles; the
task in the supplied driver is to move the fruit onto the plate.

The supplied `base.py`, `main.py`, initial scene, environment specification,
assets, and weights are retained. No extra assignment dependencies are added.

## Local evaluation

After activating the environment, run from the repository root:

```bash
python scripts/test_assignment.py --case default
python scripts/test_assignment.py --case moved
python scripts/test_assignment.py --case near
python scripts/test_assignment.py --case close
python scripts/test_assignment.py --case partial
```

The evaluator uses DIRECT mode and saves results under `.tools/assignment-checks`.
It covers the default arrangement, a moved shelf with an extra fruit, a fruit
near the base, closely spaced fruit, and a fruit already on the plate. Simulator
ground truth is used only by this external evaluator to score final placement;
it is never provided to the perception or planning implementation.

The controller reports unreachable targets, timeouts, and empty grasps rather
than continuing indefinitely. This is a tabletop simulation solution, not a
general-purpose collision planner or a real-hardware safety controller.
