#Import pybullet
import pybullet as p

#import the PyBullet simulation instance
from panda_gym.pybullet import PyBullet

#import the classes for defining a custom Robot and Task
from panda_gym.envs.core import PyBulletRobot

#import the RobotTaskEnv class
from core import Task, RobotTaskEnv

#import panda_gym utilities
from panda_gym.utils import distance

from gymnasium import spaces

from typing import Any, Dict, Optional, Union
import numpy as np
import math
import os
import torch
import random

class Panda(PyBulletRobot):
    """Panda robot in PyBullet.

    Args:
        sim (PyBullet): Simulation instance.
        block_gripper (bool, optional): Whether the gripper is blocked. Defaults to False.
        base_position (np.ndarray, optional): Position of the base of the robot, as (x, y, z). Defaults to (0, 0, 0).
        control_type (str, optional): "ee" to control end-effector displacement or "joints" to control joint angles.
            Defaults to "ee".
    """

    def __init__(
        self,
        sim: PyBullet,
        block_gripper: bool = False,
        base_position: Optional[np.ndarray] = None,
        control_type: str = "ee",
    ) -> None:
        
        #set the base position to the specified (x,y,z) coordinates or to the default (0,0,0) if no coordinates are specified
        base_position = base_position if base_position is not None else np.zeros(3)
        self.block_gripper = block_gripper
        self.control_type = control_type

        n_action = 3 if self.control_type == "ee" else 7  # control (x, y z) of the end effector if "ee", else, control the 7 joints
        n_action += 0 if self.block_gripper else 1 #if the gripper is unblocked, add another action for controlling the gripper (total actions=4)
        action_space = spaces.Box(-1.0, 1.0, shape=(n_action,), dtype=np.float32) #define the action space

        super().__init__(
            sim,
            body_name="panda",
            file_name="franka_panda/panda.urdf",
            base_position=base_position,
            action_space=action_space,
            joint_indices=np.array([0, 1, 2, 3, 4, 5, 6, 9, 10]), #seven joints (0-6) plus the two gripper fingers (9,10)
            joint_forces=np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0]),
        )

        self.fingers_indices = np.array([9, 10])
        self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
        self.ee_link = 11
        #set the lateral friction and spinning friction for the two gripper fingers
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[0], lateral_friction=1.0)
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[1], lateral_friction=1.0)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[0], spinning_friction=0.001)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[1], spinning_friction=0.001)

    """
    Define the main methods
    """

    def set_action(self, action: np.ndarray) -> None:
        action = action.copy()  # ensure action don't change
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.control_type == "ee":
            ee_displacement = action[:3]
            target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement)
        else:
            arm_joint_ctrl = action[:7]
            target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)

        if self.block_gripper:
            target_fingers_width = 0
        else:
            fingers_ctrl = action[-1] * 0.2  # limit maximum change in position
            fingers_width = self.get_fingers_width()
            target_fingers_width = fingers_width + fingers_ctrl

        target_angles = np.concatenate((target_arm_angles, [target_fingers_width / 2, target_fingers_width / 2]))
        self.control_joints(target_angles=target_angles)

    def get_obs(self) -> np.ndarray:
        # end-effector position and velocity
        ee_position = np.array(self.get_ee_position())
        ee_velocity = np.array(self.get_ee_velocity())
        # fingers opening
        if not self.block_gripper:
            fingers_width = self.get_fingers_width()
            observation = np.concatenate((ee_position, ee_velocity, [fingers_width]))
        else:
            observation = np.concatenate((ee_position, ee_velocity))
        return observation

    def reset(self) -> None:
        self.set_joint_neutral()

    def ee_displacement_to_target_arm_angles(self, ee_displacement: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the end-effector displacement.

        Args:
            ee_displacement (np.ndarray): End-effector displacement, as (dx, dy, dy).

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        ee_displacement = ee_displacement[:3] * 0.05  # limit maximum change in position
        # get the current position and the target position
        ee_position = self.get_ee_position()
        target_ee_position = ee_position + ee_displacement
        # Clip the height target. For some reason, it has a great impact on learning
        target_ee_position[2] = np.max((0, target_ee_position[2]))
        # compute the new joint angles
        target_arm_angles = self.inverse_kinematics(
            link=self.ee_link, position=target_ee_position, orientation=np.array([1.0, 0.0, 0.0, 0.0])
        )
        target_arm_angles = target_arm_angles[:7]  # remove fingers angles
        return target_arm_angles

    def arm_joint_ctrl_to_target_arm_angles(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the arm joint control.

        Args:
            arm_joint_ctrl (np.ndarray): Control of the 7 joints.

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        arm_joint_ctrl = arm_joint_ctrl * 0.05  # limit maximum change in position
        # get the current position and the target position
        current_arm_joint_angles = np.array([self.get_joint_angle(joint=i) for i in range(7)])
        target_arm_angles = current_arm_joint_angles + arm_joint_ctrl
        return target_arm_angles
    
    """
    Define the helper methods
    """

    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        """
        #For collecting offline data, radomize some joint angles
        joint2 = np.random.uniform(0.3, 0.6)
        joint3 = np.random.uniform(-0.8, 0.8)
        joint4 = np.random.uniform(-2.1, -1.5)
        joint6 = np.random.uniform(1, 3)
        joint7 = np.random.uniform(-1, 1)
        random_joint_values = np.array([0.00, joint2, joint3, joint4, 0.00, joint6, joint7, 0.00, 0.00])
        """
        self.set_joint_angles(self.neutral_joint_values)

    def get_fingers_width(self) -> float:
        """Get the distance between the fingers."""
        finger1 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[0])
        finger2 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[1])
        return finger1 + finger2

    def get_ee_position(self) -> np.ndarray:
        """Returns the position of the end-effector as (x, y, z)"""
        return self.get_link_position(self.ee_link)

    def get_ee_velocity(self) -> np.ndarray:
        """Returns the velocity of the end-effector as (vx, vy, vz)"""
        return self.get_link_velocity(self.ee_link)
    
class VialHandling(Task):
    """Custom task modelled on the PickAndPlace task.

    Args:
        sim (PyBullet): Simulation instance.
        reward_type (str): Robot's reward mechanism. Two types:
                            Sparse: the environment returns a reward only if the task is completed
                            Dense: the closer the agent is to completing the task, the higher the reward
        distance_threshold (float): If the offset between the position of the object and the goal position is within this threshold,
            the task is successful.
        y_max (float):
        y_min (float): 
        x_max (float):
        x_min (float):
        z_holder (float):
        z_vial (float):
    """
    
    def __init__(
        self,
        sim: PyBullet,
        get_ee_position,
        reward_type: str = "sparse",
        distance_threshold: float = 0.02,
        x_max: float = 0.15,
        x_min: float = -0.35,
        y_max: float = 0.20,
        y_min: float = -0.30,
        z_holder: float = 0.0,
        z_vial: float = 0.004,
    ) -> None:
        
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.z_holder = z_holder
        self.z_vial = z_vial
        self.xy_threshold = 0.1
        #unsafe parameters
        self.unsafe_region1_radius = 0.15
        self.unsafe_region2_radius = 0.11
        self.unsafe_region3_radius = 0.06
        self.unsafe_region4_size = np.array([0.6, 0.05, 0.2])
        self.safe_distance = 0.05
        #workspace range
        self.range_low = np.array([x_min, y_min , 0])
        self.range_high = np.array([x_max, y_max , 0])

        with self.sim.no_rendering():
            self._create_scene()

    """
    Define the main methods
    """

    def reset(self) -> None:
        #reset the positions
        [self.rack_position, self.unsafe_region1]  = self._sample_rack()
        [self.flask_position, self.unsafe_region2]  = self._sample_flask()
        [self.bottle_position, self.unsafe_region3] = self._sample_bottle()
        [self.human_position, self.unsafe_region4] = self._sample_human()

        [self.object, self.holder] = self._sample_position()
        while self._object_in_unsafe_region():
            [self.object, self.holder] = self._sample_position()
        
        [self.goal, self.holder_goal] = self._sample_position()
        obj_target_dist = distance(self.object, self.goal)
        while self._target_in_unsafe_region() or obj_target_dist < self.xy_threshold:
            [self.goal, self.holder_goal] = self._sample_position()
            obj_target_dist = distance(self.object, self.goal)

        #reset the orientations
        r = math.pi
        self.orientation = np.array([math.cos(r/4), math.sin(r/4), math.sin(r/4), math.cos(r/4)])

        #set the base poses
        self.sim.set_base_pose("object", self.object, self.orientation) 
        self.sim.set_base_pose("target", self.goal, self.orientation)
        self.sim.set_base_pose("object_holder", self.holder, self.orientation)
        self.sim.set_base_pose("target_holder", self.holder_goal, self.orientation)
        self.sim.set_base_pose("human", self.human_position, self.orientation)
        self.sim.set_base_pose("rack", self.rack_position, self.orientation)
        self.sim.set_base_pose("flask", self.flask_position, self.orientation)
        self.sim.set_base_pose("bottle", self.bottle_position, self.orientation)
        self.sim.set_base_pose("unsafe_region1", self.unsafe_region1, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("unsafe_region2", self.unsafe_region2, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("unsafe_region3", self.unsafe_region3, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("unsafe_region4", self.unsafe_region4, np.array([0.0, 0.0, 0.0, 1.0]))

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object_position = self.sim.get_base_position("object")
        object_rotation = self.sim.get_base_rotation("object")
        object_velocity = self.sim.get_base_velocity("object")
        object_angular_velocity = self.sim.get_base_angular_velocity("object")
        end_effector_pos = self.get_end_effector_position().copy()
        unsafe_space1 = self.sim.get_base_position("unsafe_region1")
        unsafe_space2 = self.sim.get_base_position("unsafe_region2")
        unsafe_space3 = self.sim.get_base_position("unsafe_region3")
        unsafe_space4 = self.sim.get_base_position("unsafe_region4")
        observation = np.concatenate([object_position, object_rotation, object_velocity, object_angular_velocity,
                                        unsafe_space1, unsafe_space2, unsafe_space3, unsafe_space4])
        return observation
    
    def get_achieved_goal(self) -> np.ndarray:
        object_position = np.array(self.sim.get_base_position("object"))
        return object_position
    
    def get_end_effector_position(self):
        """Get the position of the end effector (SafeRL method)"""
        ee_position = np.array(self.get_ee_position())
        return ee_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)
    
    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32)
    
    def _object_in_unsafe_region(self):
        """Determine if the object is in the unsafe region"""
        object_pos = self.object
        # distance between object and unsafe space
        distance_to_unsafe1 = distance(object_pos, self.unsafe_region1)
        distance_to_unsafe2 = distance(object_pos, self.unsafe_region2)
        distance_to_unsafe3 = distance(object_pos, self.unsafe_region3)
        # lowest distance possible is sphere radius plus safe distance 
        # this is a mathematical apporximation
        min_distance_threshhold1 = self.unsafe_region1_radius + self.safe_distance
        min_distance_threshhold2 = self.unsafe_region2_radius + self.safe_distance
        min_distance_threshhold3 = self.unsafe_region3_radius + self.safe_distance
        return (distance_to_unsafe1 < min_distance_threshhold1) or (distance_to_unsafe2 < min_distance_threshhold2) or \
            (distance_to_unsafe3 < min_distance_threshhold3)
    
    def _target_in_unsafe_region(self):
        """Determine if the target is in the unsafe region"""
        target_pos = self.goal
        # distance between target_holder and unsafe space
        distance_to_unsafe1 = distance(target_pos, self.unsafe_region1)
        distance_to_unsafe2 = distance(target_pos, self.unsafe_region2)
        distance_to_unsafe3 = distance(target_pos, self.unsafe_region3)
        # lowest distance possible is sphere radius plus safe distance
        # this is a mathematical apporximation
        min_distance_threshhold1 = self.unsafe_region1_radius + self.safe_distance
        min_distance_threshhold2 = self.unsafe_region2_radius + self.safe_distance
        min_distance_threshhold3 = self.unsafe_region3_radius + self.safe_distance
        return (distance_to_unsafe1 < min_distance_threshhold1) or (distance_to_unsafe2 < min_distance_threshhold2) or \
            (distance_to_unsafe3 < min_distance_threshhold3)
    
    def _end_effector_in_unsafe_region(self):
        end_effector_pos = self.get_end_effector_position()
        # distance between end effector and unsafe space center
        distance_end_effector_unsafe1 = distance(end_effector_pos, self.unsafe_region1)
        distance_end_effector_unsafe2 = distance(end_effector_pos, self.unsafe_region2)
        distance_end_effector_unsafe3 = distance(end_effector_pos, self.unsafe_region3)
        
        unsafe_region4_center = self.unsafe_region4
        unsafe_region4_size = self.unsafe_region4_size
        y_dist_unsafe4 = abs(end_effector_pos[1] - unsafe_region4_center[1])
        z_dist_unsafe4 = abs(end_effector_pos[2] - unsafe_region4_center[2])
        if y_dist_unsafe4 < unsafe_region4_size[1] and z_dist_unsafe4 < unsafe_region4_size[2]:
            in_unsafe4 = True
        else:
            in_unsafe4 = False
       
        min_distance_treshhold1 = self.unsafe_region1_radius
        min_distance_treshhold2 = self.unsafe_region2_radius
        min_distance_treshhold3 = self.unsafe_region3_radius
        return (distance_end_effector_unsafe1 < min_distance_treshhold1) or (distance_end_effector_unsafe2 < min_distance_treshhold2) or \
            (distance_end_effector_unsafe3 < min_distance_treshhold3) or in_unsafe4
    
    def _check_rotation(self):
        r = math.pi
        rotation_deg = 20
        rotation_rad = rotation_deg * (r/180)
        orientation = self.sim.get_base_orientation("object")
        x_init = math.cos(r/4); y_init = math.sin(r/4)
        if abs(x_init - orientation[0]) > rotation_rad or abs(y_init - orientation[1]) > rotation_rad:
            angle_constraint = True
        else:
            angle_constraint = False
        return angle_constraint
    
    def _compute_cost_function(self):
        '''
        two possible contontitions to get a non 0 cost
        1 Two objects collide
        2 end effector enters any unsafe space
        '''
        if self._end_effector_in_unsafe_region():
            exp = "End effector in unsafe region"
            return 1.0, exp
    
        elif self._check_rotation():
            exp = "Object rotated by more than 20 deg"
            return 1.0, exp

        elif self._has_contacts_objects():
            exp = "Two objects collided"
            return 1.0, exp
        
        else:
            exp = "No constraints violated"
            return 0.0, exp

    def compute_cost(self) -> Union[np.ndarray, float]:
        [cost, exp] = self._compute_cost_function()
        return cost, exp
    
    def has_contacts(self):
        contact = False
        if len(self.sim.physics_client.getContactPoints(self.sim._bodies_idx['panda'], self.sim._bodies_idx['rack'])) or \
        len(self.sim.physics_client.getContactPoints(self.sim._bodies_idx['panda'], self.sim._bodies_idx['flask'])) or \
        len(self.sim.physics_client.getContactPoints(self.sim._bodies_idx['panda'], self.sim._bodies_idx['bottle'])) or \
        len(self.sim.physics_client.getContactPoints(self.sim._bodies_idx['panda'], self.sim._bodies_idx['human'])):
            contact = True
        return contact

    def _has_contacts_objects(self):
        contact = False
        if len(self.sim.physics_client.getContactPoints(self.sim._bodies_idx['object'], self.sim._bodies_idx['rack'])) or \
        len(self.sim.physics_client.getContactPoints(self.sim._bodies_idx['object'], self.sim._bodies_idx['flask'])) or \
        len(self.sim.physics_client.getContactPoints(self.sim._bodies_idx['object'], self.sim._bodies_idx['bottle'])):
            contact = True
        return contact
        
    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.2, width=0.7, height=0.4, x_offset=-0.3)

        #define the paths to the vial and vial holder .obj files
        vial_visual = "../Panda_Environments/Models/Github_objects/vial.obj" #visual shape file
        vial_collision = "../Panda_Environments/Models/Github_objects/vial_coll.obj" #physical shape file

        vial_holder_visual = "../Panda_Environments/Models/Github_objects/vial_holder.obj" #visual shape file
        vial_holder_collision = "../Panda_Environments/Models/Github_objects/vial_holder_coll.obj" #physical shape file

        human_visual = "../Panda_Environments/Models/Human/obj_urdf_file/human.obj"
        human_coll = "../Panda_Environments/Models/Human/obj_urdf_file/human.obj"

        rack_visual = "../Panda_Environments/Models/Vial/obj_urdf_file/vial_rack.obj"
        rack_collision = "../Panda_Environments/Models/Vial/obj_urdf_file/vial_rack_coll.obj"

        flask = "../Panda_Environments/Models/Flask/obj_urdf_file/flask.obj"
        bottle = "../Panda_Environments/Models/Bottle/obj_urdf_file/bottle.obj"

        #specify mesh scales to ensure the objects are an appropriate size
        meshScale = [0.03] * 3
        meshScale_human = [0.5] * 3
        meshScale_rack = [0.4] * 3
        meshScale_flask = None
        meshScale_bottle = [0.5] * 3

        #define the rgba colours for all objects
        rgbaColors = {'object' : [0.8, 0.8, 0.0, 0.15], 
                      'target': [0.4, 0.8, 0.0, 0.15], 
                      'object_holder': [0.0, 0.0, 0.5, 0.3], 
                      'target_holder' : [0.0, 0.0, 0.5, 0.3],
                      'items' : [0.8, 0.9, 0.9, 0.5],
                      'unsafe_region' : [0.9, 0.1, 0.1, 0.3]}

        #import the vial object into the environment
        self.sim._create_geometry(
            body_name='object',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass=1,
            position=np.array([0.0, -0.2, self.z_vial]),
            ghost=False,
            visual_kwargs={
                'fileName': vial_visual,
                'meshScale': meshScale,
                'rgbaColor': rgbaColors['object']},
            collision_kwargs={
                'fileName': vial_collision,
                'meshScale': meshScale}
        )
        #import the vial target into the environment
        self.sim._create_geometry(
            body_name='target',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass=0,
            position=np.array([0.0, 0.1, self.z_vial]),
            ghost=True,
            visual_kwargs={
                'fileName': vial_visual,
                'meshScale': meshScale,
                'rgbaColor': rgbaColors['target']},
            collision_kwargs={
                'fileName': vial_collision,
                'meshScale': meshScale}
        )
        #import the vial object holder into the environment
        self.sim._create_geometry(
            body_name='object_holder',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass=1000.0,
            position=np.array([0.0, -0.2, self.z_holder]),
            ghost=False,
            visual_kwargs={
                'fileName': vial_holder_visual,
                'meshScale': meshScale,
                'rgbaColor': rgbaColors['object_holder']},
            collision_kwargs={
                'fileName': vial_holder_collision,
                'meshScale': meshScale}
        )
        #create the import the vial target holder into the environment
        self.sim._create_geometry(
            body_name='target_holder',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass=1000.0,
            position=np.array([0.0, 0.1, self.z_vial]),
            ghost=False,
            visual_kwargs={
                'fileName': vial_holder_visual,
                'meshScale': meshScale,
                'rgbaColor': rgbaColors['target_holder']},
            collision_kwargs={
                'fileName': vial_holder_collision,
                'meshScale': meshScale}
        )
        #import a SMPL-X human mesh into the environment
        self.sim._create_geometry(
            body_name='human',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass=1000.0,
            position=np.array([0.0, 0.5, -0.4]),
            ghost=False,
            visual_kwargs={
                'fileName': human_visual,
                'meshScale': meshScale_human},
            collision_kwargs={
                'fileName': human_coll,
                'meshScale': meshScale_human}
        )
        #import the stocked vial rack into the environment
        self.sim._create_geometry(
            body_name="rack",
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass=1.0,
            position=np.array([-0.3, 0.25, 0.0]),
            ghost=False,
            visual_kwargs={
                'fileName': rack_visual,
                'meshScale': meshScale_rack,
                'rgbaColor': rgbaColors['items']},
            collision_kwargs={
                'fileName': rack_collision,
                'meshScale': meshScale_rack}
        )
        #import the solvent flask into the environment
        self.sim._create_geometry(
            body_name="flask",
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass=1.0,
            position=np.array([0.0, 0.0, 0.0]),
            ghost=False,
            visual_kwargs={
                'fileName': flask,
                'rgbaColor': rgbaColors['items']},
            collision_kwargs={
                'fileName': flask}
        )
        #import the solvent flask into the environment
        self.sim._create_geometry(
            body_name="bottle",
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass=1.0,
            position=np.array([0.0, 0.0, 0.0]),
            ghost=False,
            visual_kwargs={
                'fileName': bottle,
                'meshScale': meshScale_bottle,
                'rgbaColor': rgbaColors['items']},
            collision_kwargs={
                'fileName': bottle,
                'meshScale': meshScale_bottle}
        )
        #create the unsafe region(s)
        self.sim.create_sphere(
            body_name="unsafe_region1",
            radius=self.unsafe_region1_radius,
            mass=0.0,
            ghost=True,
            position=np.array([-0.3, 0.25, 0.0]),
            rgba_color=rgbaColors['unsafe_region']
        )

        self.sim.create_sphere(
            body_name="unsafe_region2",
            radius=self.unsafe_region2_radius,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, -0.25, 0.0]),
            rgba_color=rgbaColors['unsafe_region']
        )

        self.sim.create_sphere(
            body_name="unsafe_region3",
            radius=self.unsafe_region3_radius,
            mass=0.0,
            ghost=True,
            position=np.array([0.1, -0.25, 0.0]),
            rgba_color=rgbaColors['unsafe_region']
        )

        self.sim.create_box(
            body_name="unsafe_region4",
            half_extents=self.unsafe_region4_size,
            mass=0.0,
            ghost=True,
            position=np.array([0.1, -0.25, 0.0]),
            rgba_color=rgbaColors['unsafe_region']
        )

    def _sample_position(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([0.0, 0.0, 0.0])
        holder_position = np.array([0.0, 0.0, 0.0])
        noise = self.np_random.uniform(self.range_low, self.range_high)
        object_position += noise
        holder_position += noise
        object_position[2] = self.z_vial
        holder_position[2] = self.z_holder
        return object_position, holder_position
    
    def _sample_rack(self) -> np.ndarray:
        """sample the position of the vial rack"""
        rack_position = np.array([-0.3, 0.25, 0.0])
        rack_unsafe_region = rack_position.copy()
        return rack_position, rack_unsafe_region

    def _sample_human(self) -> np.ndarray:
        """sample the position of the vial rack"""
        human_position = np.array([-0.35, 0.60, -0.4])
        human_unsafe_region = np.array([-0.3, 0.375, 0.0])
        return human_position, human_unsafe_region
    
    def _sample_flask(self) -> np.ndarray:
        flask_position = np.array([-0.1, -0.20, 0.0])
        flask_unsafe_region = flask_position.copy()
        flask_unsafe_region[2] = flask_position[2] + 0.05
        return flask_position, flask_unsafe_region
    
    def _sample_bottle(self) -> np.ndarray:
        bottle_position = np.array([0.15, 0.0, 0.0])
        bottle_unsafe_region = bottle_position.copy()
        bottle_unsafe_region[2] = bottle_position[2] + 0.05
        return bottle_position, bottle_unsafe_region
    
class VialHandlingEnv(RobotTaskEnv):
    """Pick and Place task wih Panda robot.

    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        renderer (str, optional): Renderer, either "Tiny" or OpenGL". Defaults to "Tiny" if render mode is "human"
            and "OpenGL" if render mode is "rgb_array". Only "OpenGL" is available for human render mode.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targetting this postion, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Rool of the camera. Defaults to 0.
    """
    #for top view --> render_target_position = [0.2, 0, 0], render_yaw = 90, render_pitch = -70, render_roll = 0.0
    def __init__(
        self,
        render_mode: str = "human",
        reward_type: str = "sparse",
        control_type: str = "ee",
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.0,
        render_yaw: float = 45,
        render_pitch: float = -40,
        render_roll: float = 0.0,
    ) -> None:
        sim = PyBullet()
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = VialHandling(sim, reward_type=reward_type, get_ee_position=robot.get_ee_position)
        super().__init__(
            robot,
            task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        #VialHandlingEnv.seed(seed)