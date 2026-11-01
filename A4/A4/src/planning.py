import numpy as np
from perception import Perception


class Planner:
    def __init__(self, manipulator):
        self.perceptor = Perception(manipulator)

    def get_clear_task_plan(self, objs):
        plates = [label for label in objs if label.startswith("plate")]
        if not plates:
            raise ValueError("No plate was identified; acquire another view")
        plate = plates[0]
        support = np.asarray(objs[plate].points)
        center = np.mean(np.percentile(support, [3, 97], axis=0), axis=0)
        top = np.percentile(support[:, 2], 90)
        plan = [("move_home", ())]
        labels = [label for label in objs if label.startswith(("lemon", "orange", "peach", "plum"))]
        radius = min(0.065, 0.22 * np.min(np.ptp(support[:, :2], axis=0)))
        for index, label in enumerate(labels):
            points = np.asarray(objs[label].points)
            distance = np.linalg.norm(np.median(points[:, :2], axis=0) - center[:2])
            if distance < 0.075 and np.median(points[:, 2]) > top:
                continue
            steps = self.pick_place_object_plan(label, plate, objs)
            angle = 2 * np.pi * index / len(labels)
            steps[1][1][0][:2] += radius * np.array([np.cos(angle), np.sin(angle)])
            plan.extend(steps)
        return plan

    def pick_place_object_plan(self, obj_label, place_location_label, objs):
        pick = self.perceptor.get_object_pick_pose(obj_label, objs)
        place = self.perceptor.get_obj_place_pose(obj_label, place_location_label, objs)
        return [
            ("pick_object_from", (pick[:3], pick[3:])),
            ("place_object_at", (place[:3], place[3:])),
        ]
