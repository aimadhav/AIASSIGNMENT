from pathlib import Path
import copy

import cv2
import numpy as np
import open3d as o3d
from ultralytics import YOLO

from base import RGBDSensor
from control import Manipulator


class ObjectDetector:
    def __init__(self):
        self.detector = YOLO(str(Path(__file__).parent / "weights" / "best.pt"))

    def detect_objects(self, rgbImage):
        image = cv2.cvtColor(np.asarray(rgbImage, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        result = self.detector.predict(image, conf=0.25, device="cpu", verbose=False)[0]
        labels = [result.names[int(index)] for index in result.boxes.cls.cpu().numpy()]
        return labels, result.boxes.xyxy.cpu().numpy()


class Perception:
    def __init__(self, manipulator: Manipulator):
        self.sensor = RGBDSensor()
        self.obj_detector = ObjectDetector()
        self.manipulator = manipulator
        self.depth_processor = DepthProcessor()
        self.observations = []

    def generate_exploration_poses(self):
        poses = []
        for pitch in (0, 0.22):
            for yaw in np.linspace(-np.pi / 2, np.pi / 2, 5):
                pose = np.array(self.manipulator.PANDA_HOME_POSITION)
                pose[0], pose[5] = yaw, pose[5] + pitch
                poses.append(pose.tolist())
        return poses

    def exlpore_scene(self, exlporation_poses):
        clouds = []
        self.observations = []
        self.manipulator.move_home()
        for pose in exlporation_poses:
            self.manipulator.move_joints(pose)
            rgb, depth, transform, intrinsics = self.sensor.get_observation(
                self.manipulator.p, self.manipulator.pandaId
            )
            self.observations.append((rgb, depth, transform, intrinsics))
            cloud = self.depth_processor.rgbd_to_pointcloud(rgb, depth, intrinsics, 640, 480)
            clouds.append(self.depth_processor.transform_pointcloud(cloud, transform))
        self.manipulator.move_home()
        return clouds

    def _associate(self, cluster, observation, detections):
        _, depth, transform, intrinsics = observation
        points = np.asarray(cluster.points)
        camera = points @ transform[:3, :3] - transform[:3, :3].T @ transform[:3, 3]
        visible = camera[:, 2] > 0
        pixels = np.full((len(points), 2), -1.0)
        pixels[visible] = camera[visible, :2] / camera[visible, 2, None] * intrinsics[:2] + intrinsics[2:]
        u, v = np.rint(pixels).astype(int).T
        visible &= (u >= 0) & (u < 640) & (v >= 0) & (v < 480)
        indices = np.flatnonzero(visible)
        visible[indices] &= np.abs(depth[v[indices], u[indices]] - camera[indices, 2]) < 0.015
        seeds, names, masks, score = [], [], [], 0
        for label, box in zip(*detections):
            inside = visible & np.all(pixels >= box[:2], axis=1) & np.all(pixels <= box[2:], axis=1)
            if inside.sum() < max(12, 0.025 * visible.sum()):
                continue
            seed = np.median(points[inside], axis=0)
            if any(np.linalg.norm(seed - other) < 0.02 for other in seeds):
                continue
            seeds.append(seed)
            names.append(label)
            masks.append(inside)
            score += inside.sum()
        return score, names, seeds, masks

    def identify_objects(self, pointcloud_clusters):
        views = [(view, self.obj_detector.detect_objects(view[0])) for view in self.observations]
        objects = {}
        for cluster in pointcloud_clusters:
            matches = [self._associate(cluster, view, detections) for view, detections in views]
            position = self.depth_processor.get_cluster_top_position(cluster)
            self.manipulator.move_home()
            try:
                self.manipulator.move_in_task_space(position.tolist(), [np.pi, 0, 0])
                view = self.sensor.get_observation(self.manipulator.p, self.manipulator.pandaId)
                matches.append(self._associate(cluster, view, self.obj_detector.detect_objects(view[0])))
            except RuntimeError:
                pass
            score, names, seeds, masks = max(matches, default=(0, [], [], []), key=lambda match: match[0])
            if not score:
                continue
            points = np.asarray(cluster.points)
            distances = np.linalg.norm(points[:, None, :] - np.array(seeds)[None, :, :], axis=2)
            if any(label.startswith("plate") for label in names):
                for index, label in enumerate(names):
                    if not label.startswith("plate"):
                        distances[~masks[index], index] = np.inf
            for index, label in enumerate(names):
                selected = np.flatnonzero(np.argmin(distances, axis=1) == index)
                if len(selected) < 20:
                    continue
                part = cluster.select_by_index(selected.tolist())
                for existing in objects:
                    same_class = existing == label or existing.startswith(label + "_")
                    distance = np.linalg.norm(np.median(np.asarray(objects[existing].points), axis=0) - seeds[index])
                    if same_class and distance < (0.18 if label.startswith("plate") else 0.065):
                        objects[existing] += part
                        break
                else:
                    key, duplicate = label, 2
                    while key in objects:
                        key, duplicate = label + "_" + str(duplicate), duplicate + 1
                    objects[key] = part
        plates = [key for key in objects if key.startswith("plate")]
        if plates:
            objects[plates[0]] = max((objects[key] for key in plates), key=lambda cloud: len(cloud.points))
            for key in plates[1:]:
                del objects[key]
        self.manipulator.move_home()
        return objects

    def get_object_pick_pose(self, obj_label, objs):
        points = np.asarray(objs[obj_label].points)
        low, high = np.percentile(points, [3, 97], axis=0)
        center = (low + high) / 2
        center[2] = low[2] + 0.4 * (high[2] - low[2])
        values, vectors = np.linalg.eigh(np.cov(points[:, :2].T))
        direction = vectors[:, -1]
        yaw = np.arctan2(direction[1], direction[0]) if values[-1] > 1.4 * values[0] else 0
        yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
        radius = np.linalg.norm(center[:2])
        pitch = np.clip(6 * (0.3 - radius), 0, 0.65)
        if pitch > 0:
            yaw = np.arctan2(center[1], center[0])
        return np.r_[center, np.pi, pitch, yaw]

    def get_obj_place_pose(self, obj_label, target_obj_label, objs):
        points = np.asarray(objs[obj_label].points)
        target = np.asarray(objs[target_obj_label].points)
        center = np.mean(np.percentile(target, [3, 97], axis=0), axis=0)
        height = np.percentile(points[:, 2], 97) - np.percentile(points[:, 2], 3)
        center[2] = np.percentile(target[:, 2], 95) + 0.5 * height + 0.015
        return np.r_[center, np.pi, 0, 0]


class DepthProcessor:
    def rgbd_to_pointcloud(self, rgb, depth, intrinsics, width, height):
        depth = np.asarray(depth)
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        fx, fy, cx, cy = intrinsics
        valid = np.isfinite(depth) & (depth > 0.12) & (depth < 1.8)
        xyz = np.stack(((u - cx) * depth / fx, (v - cy) * depth / fy, depth), axis=-1)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(xyz[valid])
        cloud.colors = o3d.utility.Vector3dVector(np.asarray(rgb)[valid].astype(float) / 255)
        return cloud.voxel_down_sample(0.004)

    def transform_pointcloud(self, pointcloud, transform):
        return copy.deepcopy(pointcloud).transform(np.asarray(transform))

    def register_pointclouds(self, pcds):
        if not pcds:
            raise ValueError("No observations were collected")
        scene = copy.deepcopy(pcds[0])
        for cloud in pcds[1:]:
            source = cloud.voxel_down_sample(0.008)
            target = scene.voxel_down_sample(0.008)
            target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))
            result = o3d.pipelines.registration.registration_icp(
                source, target, 0.015, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30),
            )
            correction = result.transformation
            aligned = copy.deepcopy(cloud)
            translation = np.linalg.norm(correction[:3, 3])
            rotation = np.linalg.norm(correction[:3, :3] - np.eye(3))
            if result.fitness > 0.3 and translation < 0.015 and rotation < 0.05:
                aligned.transform(correction)
            scene += aligned
            scene = scene.voxel_down_sample(0.004)
        points = np.asarray(scene.points)
        radius = np.linalg.norm(points[:, :2], axis=1)
        keep = (radius > 0.12) & (radius < 0.95) & (points[:, 2] > -0.65) & (points[:, 2] < 0.6)
        return scene.select_by_index(np.flatnonzero(keep).tolist())

    def visualize_pointcloud(self, pcds):
        o3d.visualization.draw_geometries(pcds)

    def remove_plane(self, scene_pcd):
        if len(scene_pcd.points) < 3:
            return copy.deepcopy(scene_pcd)
        _, inliers = scene_pcd.segment_plane(distance_threshold=0.005, ransac_n=3, num_iterations=1000)
        return scene_pcd.select_by_index(inliers, invert=True)

    def remove_planes(self, scene_pcd):
        o3d.utility.random.seed(0)
        scene = copy.deepcopy(scene_pcd)
        for _ in range(3):
            if len(scene.points) < 100:
                break
            plane, inliers = scene.segment_plane(distance_threshold=0.005, ransac_n=3, num_iterations=1000)
            support = np.asarray(scene.points)[inliers]
            if abs(plane[2]) < 0.9 or np.min(np.ptp(support[:, :2], axis=0)) < 0.35:
                break
            scene = scene.select_by_index(inliers, invert=True)
        return scene

    def cluster_objects(self, scene_pcd):
        if not len(scene_pcd.points):
            return []
        labels = np.array(scene_pcd.cluster_dbscan(eps=0.012, min_points=10))
        clusters = []
        for label in np.unique(labels[labels >= 0]):
            indices = np.flatnonzero(labels == label)
            points = np.asarray(scene_pcd.points)[indices]
            span = np.ptp(points, axis=0)
            compact = span[2] < 2 * max(span[:2])
            if len(indices) >= 80 and max(span) < 0.4 and span[2] < 0.2 and compact:
                clusters.append(scene_pcd.select_by_index(indices.tolist()))
        return clusters

    def get_cluster_top_position(self, cluster):
        points = np.asarray(cluster.points)
        position = np.median(points, axis=0)
        position[2] = max(0.5, np.percentile(points[:, 2], 98) + 0.26)
        return position

    def project_world_to_image(self, point_world, intrinsics, T_cb):
        camera = np.linalg.solve(np.asarray(T_cb), np.r_[point_world, 1])[:3]
        if camera[2] <= 0:
            return (np.nan, np.nan)
        fx, fy, cx, cy = intrinsics
        return (fx * camera[0] / camera[2] + cx, fy * camera[1] / camera[2] + cy)
