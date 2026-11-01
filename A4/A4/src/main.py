
import pybullet as p

from base import Environment
from planning import Planner
from control import Manipulator
from perception import Perception, DepthProcessor
import json



if __name__=="__main__":
    ## 0. Initialize the environment with the initial state provided in init_state.json. You can test your code on various initial states by changing the json file.
    with open('./init_state.json', 'r') as f:
        init_state = json.load(f)

    env = Environment(p, init_state)
    p, pandaId = env.get_bullet_client(), env.get_panda_id()
    manipulator = Manipulator(p, pandaId)
    planner = Planner(manipulator)
    perceptor = Perception(manipulator)
    depth_processor = DepthProcessor()

    ## 1.1 Propose exploration poses using heuristics and/or current observation
    exploration_poses = perceptor.generate_exploration_poses()

    ## 1.2 Explore the scene with proposed exploration and collect spatial information
    pcds = perceptor.exlpore_scene(exploration_poses)

    ## 1.3 combine the observations
    merged_pcd = depth_processor.register_pointclouds(pcds)

    ## 1.4 to retrieve smaller, manipulable objects, assuming table top scene remove table and ground plane from the collected spatial information
    extracted_objects_scene = depth_processor.remove_planes(merged_pcd)

    ## 1.5 cluster remaining points to get pointclouds of dense objects (assuming objects are sparsely placed)
    objs_clusters = depth_processor.cluster_objects(extracted_objects_scene)

    ## 1.6 using clusters and object detector identify objects in 2D and assign corresponding object cluster the object label
    objs = perceptor.identify_objects(objs_clusters)

    ## 2.1 generate a plan for the task of clearing the fruits on the table to the plate on the side
    plan = planner.get_clear_task_plan(objs)

    ## 2.2 execute the generate plan to perform the task
    manipulator.execute_task_plan(plan)
