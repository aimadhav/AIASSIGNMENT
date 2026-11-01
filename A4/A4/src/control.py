import time

import numpy as np
from base import PandaArm


class Manipulator(PandaArm):
    def __init__(self, p, pandaId):
        super().__init__(p, pandaId)
        self.p = p
        self.gui = p.getConnectionInfo()["connectionMethod"] == p.GUI
        self.holding = False
        p.setRealTimeSimulation(0)
        p.setTimeStep(1 / 240)
        for finger in self.PANDA_PRISMATIC_JOINTS:
            p.changeDynamics(pandaId, finger, lateralFriction=2, spinningFriction=0.02)

    def _step(self, count=1):
        for _ in range(count):
            self.p.stepSimulation()
            if self.gui:
                time.sleep(1 / 240)

    def _joints(self):
        return self.p.getJointStates(self.pandaId, self.REVOLUTE_JOINTS)

    def is_moving(self):
        return any(abs(state[1]) > 0.02 for state in self._joints())

    def move_joints(self, target_joints):
        target = np.asarray(target_joints, dtype=float)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError("Expected seven finite joint angles")
        if np.any(target < self.JOINT_LOWER_LIMITS) or np.any(target > self.JOINT_UPPER_LIMITS):
            raise ValueError("Joint target exceeds arm limits")
        start = np.array([state[0] for state in self._joints()])
        steps = max(24, int(np.max(np.abs(target - start)) / 0.006))
        for fraction in np.linspace(0, 1, steps):
            blend = fraction * fraction * (3 - 2 * fraction)
            self.p.setJointMotorControlArray(
                self.pandaId, self.REVOLUTE_JOINTS, self.p.POSITION_CONTROL,
                targetPositions=(start + blend * (target - start)).tolist(),
                forces=[87] * 4 + [12] * 3, positionGains=[0.2] * 7,
            )
            self._step()
        for _ in range(480):
            error = max(abs(s[0] - q) for s, q in zip(self._joints(), target))
            if error < 0.008 and not self.is_moving():
                return True
            self._step()
        raise RuntimeError("Arm did not settle at its joint target")

    def move_home(self):
        return self.move_joints(self.PANDA_HOME_POSITION)

    def _ik(self, position, orientation):
        quaternion = self.p.getQuaternionFromEuler(orientation) if len(orientation) == 3 else orientation
        lower = self.JOINT_LOWER_LIMITS + [0, 0]
        upper = self.JOINT_UPPER_LIMITS + [0.04, 0.04]
        rest = [s[0] for s in self._joints()] + [0.04, 0.04]
        joints = self.p.calculateInverseKinematics(
            self.pandaId, self.PANDA_END_EFFECTOR_INDEX, position, quaternion,
            lowerLimits=lower, upperLimits=upper,
            jointRanges=(np.array(upper) - lower).tolist(), restPoses=rest,
            maxNumIterations=300, residualThreshold=1e-5,
        )
        return np.clip(joints[:7], self.JOINT_LOWER_LIMITS, self.JOINT_UPPER_LIMITS).tolist()

    def move_in_task_space(self, target_position, target_orientation):
        self.move_joints(self._ik(target_position, target_orientation))
        actual = self.p.getLinkState(self.pandaId, self.PANDA_END_EFFECTOR_INDEX)[4]
        if np.linalg.norm(np.array(actual) - target_position) > 0.025:
            raise RuntimeError("Unreachable tool target: " + str(np.round(target_position, 3).tolist()))
        return True

    def command_gripper(self, open):
        target = self.PANDA_FINGER_OPEN if open else self.PANDA_FINGER_CLOSED
        self.p.setJointMotorControlArray(
            self.pandaId, self.PANDA_PRISMATIC_JOINTS, self.p.POSITION_CONTROL,
            targetPositions=[target, target], forces=[30, 30], positionGains=[0.25, 0.25],
        )
        self._step(120)
        fingers = self.p.getJointStates(self.pandaId, self.PANDA_PRISMATIC_JOINTS)
        width = sum(state[0] for state in fingers)
        return width > 0.07 if open else width > 0.003

    def pick_object_from(self, obj_location, obj_orientation):
        position = np.asarray(obj_location, dtype=float)
        above = position.copy()
        above[2] = max(0.45, position[2] + 0.18)
        self.command_gripper(True)
        self.move_in_task_space(above.tolist(), obj_orientation)
        self.move_in_task_space(position.tolist(), obj_orientation)
        if not self.command_gripper(False):
            self.move_in_task_space(above.tolist(), obj_orientation)
            raise RuntimeError("Gripper closed without an object")
        self.move_in_task_space(above.tolist(), obj_orientation)
        self.holding = True
        return True

    def place_object_at(self, target_location, target_orientation):
        if not self.holding:
            raise RuntimeError("Cannot place without a preceding grasp")
        position = np.asarray(target_location, dtype=float)
        above = position.copy()
        above[2] = max(0.5, position[2] + 0.16)
        self.move_in_task_space(above.tolist(), target_orientation)
        self.move_in_task_space(position.tolist(), target_orientation)
        self.command_gripper(True)
        self.holding = False
        self.move_in_task_space(above.tolist(), target_orientation)
        self.move_home()
        return True

    def execute_task_plan(self, plan):
        allowed = {"move_home", "pick_object_from", "place_object_at"}
        for action, arguments in plan:
            if action not in allowed:
                raise ValueError("Unsupported plan action: " + action)
            getattr(self, action)(*arguments)
        return True
