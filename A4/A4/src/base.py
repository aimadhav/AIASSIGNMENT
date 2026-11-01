from abc import ABC
import numpy as np
import pybullet as p
import pybullet_data
import math



class PandaArm(ABC):
    REVOLUTE_JOINTS = [0,1,2,3,4,5,6] 
    PANDA_PRISMATIC_JOINTS = [9,10]
    JOINT_LOWER_LIMITS = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973] 
    JOINT_UPPER_LIMITS = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973] 
    PANDA_END_EFFECTOR_INDEX = 11
    PANDA_HOME_POSITION = [0.002434193097806552, -0.7811217832664136, 0.009772560588829905, -2.062867921360752, -0.007282385107511599, 1.4879767515924243, 0.8049546850317025]
    PANDA_FINGER_OPEN = 0.04
    PANDA_FINGER_CLOSED = 0

    def __init__(self, p, pandaId):
        self.pybullet_client, self.pandaId = self.create_finger_constraint(p, pandaId)

    def create_finger_constraint(self, p, pandaId):
        # if you wish to change this implementation please do so in control.Manipulator
        finger_constraint = p.createConstraint(
            pandaId, 9, pandaId, 10,
            jointType = p.JOINT_GEAR,
            jointAxis = [1,0,0],
            parentFramePosition = [0,0,0],
            childFramePosition = [0,0,0])
        p.changeConstraint(finger_constraint, gearRatio=-1, erp=0.1, maxForce=50)
        return p, pandaId




class RGBDSensor:
    IMG_WIDTH, IMG_HEIGHT = 640, 480
    NEAR_VAL, FAR_VAL = 0.047, 2

    def __init__(self, ):
        pass

    def get_observation(self, p, pandaId):
        '''
        returns rgb and depth image captured from a camera mounted on hand
        args
            robotId : pybullet robot instance
        returns
            rgbImage : RGB Image captured by the camera at the current instant
            depthImage : single channel depth image captured concurrently
            T_cb : transformation of camera frame wrt base frame
            intrinsics : camera intrinsics, this can be used to convert scalar depth value to 3D spatial vector
        '''
        hand_position, hand_orientation, _, _, _, _ = p.getLinkState(pandaId, 11, computeForwardKinematics=True)
        rot_matrix = np.array(p.getMatrixFromQuaternion(hand_orientation)).reshape(3,3)
        init_camera_vector = np.array([0,0,1]) # camera is looking along z direction
        camera_vector = np.dot(rot_matrix, init_camera_vector)

        init_up_vector = np.array([0,1,0]) # orthogonal vector to the camera vector
        up_vector = np.dot(rot_matrix, init_up_vector)
        camera_target_pos = np.array(hand_position) + np.array([0.04, 0, -0.02]) # camera spatial position wrt panda hand link
        view_matrix = p.computeViewMatrix(cameraEyePosition=camera_target_pos,
                        cameraTargetPosition=camera_target_pos+0.1*camera_vector, cameraUpVector=up_vector)
        projection_matrix = p.computeProjectionMatrixFOV(fov=70, aspect=1,
                        nearVal=self.NEAR_VAL, farVal=self.FAR_VAL)

        _, _, rgbaImg, depthImg, _ = p.getCameraImage(width=self.IMG_WIDTH, height=self.IMG_HEIGHT,
                        viewMatrix=view_matrix, projectionMatrix=projection_matrix, 
                        lightDirection=[.5,2,1.5], shadow=1, renderer=p.ER_TINY_RENDERER)
        rgbImage = self.__rgba2rgb(rgbaImg)
        
        # convert depthBuffer to depth image using near and far values. 
        # See https://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl
        # and https://stackoverflow.com/questions/70955660/how-to-get-depth-images-from-the-camera-in-pybullet 
        depthImage = self.FAR_VAL * self.NEAR_VAL / (self.FAR_VAL - (self.FAR_VAL - self.NEAR_VAL) * depthImg)

        T_cb = self.__get_tf_wrt_base(view_matrix)
        intrinsics = self.__get_intrinsics_from_projection_matrix(projection_matrix)
        return rgbImage, depthImage, T_cb, intrinsics

    def __get_tf_wrt_base(self, view_matrix):
        '''
        returns a transformation of camera frame wrt the robot base frame
        args
            view_matrix : 
        '''
        Tc = np.array([[1,  0,  0,  0],
                    [0,  -1,  0,  0],
                    [0,  0,  -1,  0],
                    [0,  0,  0,  1]])
        T_cb = np.linalg.inv(np.array(view_matrix).reshape(4,4)).T @ Tc
        return T_cb

    def __get_intrinsics_from_projection_matrix(self, projection_matrix):
        '''
        returns intrinsics asuming stereo depth camera
        args
            projection_matrix : 
            width : 
            height :
        '''
        width, height = self.IMG_WIDTH, self.IMG_HEIGHT
        fx = projection_matrix[0] * width / 2
        fy = projection_matrix[5] * height / 2
        cx = projection_matrix[2] * width / 2 + width / 2
        cy = projection_matrix[6] * height / 2 + height / 2
        intrinsics = [fx, fy, cx, cy]
        return intrinsics

    def __rgba2rgb(self, rgba, background=(255, 255, 255)):
        """
        Convert rgba to rgb.

        Args:
            rgba (tuple):
            background (tuple):

        Returns:
            rgb (tuple):
        """
        row, col, ch = rgba.shape
        if ch == 3:
            return rgba
        assert ch == 4, 'RGBA image has 4 channels.'
        rgb = np.zeros((row, col, 3), dtype='float32')
        r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]
        a = np.asarray(a, dtype='float32') / 255.0
        R, G, B = background
        rgb[:, :, 0] = r * a + (1.0 - a) * R
        rgb[:, :, 1] = g * a + (1.0 - a) * G
        rgb[:, :, 2] = b * a + (1.0 - a) * B
        return np.asarray(rgb, dtype='uint8')



class Environment:
    def __init__(self, p, init_state):
        '''
        args
            p: provide the name given to the pybullet client
            init_state: dictionary containing initial state information
        returns
            p: pybullet api
            pandaId: manipulator id
        '''
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.8)
        p.setRealTimeSimulation(1)
        
        #Add plane, Table and Robot. The location of these will be fixed and will not change in the assignment evaluation.
        plane_position = np.array([0, 0, -0.58])
        p.loadURDF("plane.urdf", basePosition=plane_position)
        tablePosition = np.array([0.3,0.1,0]) + plane_position
        tableOrientation = p.getQuaternionFromEuler([0,0,-math.pi/2])
        p.loadURDF("table/table.urdf", tablePosition, tableOrientation)
        pandaId = p.loadURDF("franka_panda/panda.urdf", [0,0,0], p.getQuaternionFromEuler([0,0,0]), useFixedBase=True)
        
        #Adding plate and objects. The number of objects and thier position as well as the plate position will change in the assignment evaluation.
        # You can change the init_state.json to test your code with different object positions and number of objects. But make sure to reset it to the original state before submission.
        p.loadURDF('./assets/Collection/small_shelf.urdf', init_state["shelf_position"], globalScaling=.6)
        plate_position = np.array(init_state['shelf_position']) + np.array([0,0,0.26])
        p.loadURDF('./assets/Collection/plate.urdf', plate_position.tolist(), globalScaling=.1)
        
        
        for (object_name, position) in init_state["objects"].items():
            if object_name == 'wood_block':
                p.loadURDF("./assets/Collection/wood_block.urdf", position, globalScaling=.4)
            else:
                 p.loadURDF(f"./assets/Collection/{object_name}.urdf", position, globalScaling=.1)

        self.bullet_client = p
        self.pandaId = pandaId
    
    def get_panda_id(self):
        return self.pandaId
    
    def get_bullet_client(self):
        return self.bullet_client