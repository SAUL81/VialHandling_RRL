from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding

from panda_gym.envs.core import PyBulletRobot
from panda_gym.pybullet import PyBullet

class Task(ABC):
    """Base class for tasks.
    Args:
        sim (PyBullet): Simulation instance.
    """

    def __init__(self, sim: PyBullet) -> None:
        self.sim = sim
        self.goal = None

    @abstractmethod
    def reset(self) -> None:
        """Reset the task: sample a new goal."""

    @abstractmethod
    def get_obs(self) -> np.ndarray:
        """Return the observation associated to the task."""

    @abstractmethod
    def get_achieved_goal(self) -> np.ndarray:
        """Return the achieved goal."""

    def get_goal(self) -> np.ndarray:
        """Return the current goal."""
        if self.goal is None:
            raise RuntimeError("No goal yet, call reset() first")
        else:
            return self.goal.copy()

    @abstractmethod
    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        """Returns whether the achieved goal match the desired goal."""

    @abstractmethod
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        """Compute reward associated to the achieved and the desired goal."""

    @abstractmethod
    def compute_cost(self) -> Union[np.ndarray, float]:
        """Compute cost associated  with constraint violations."""

    @abstractmethod
    def has_contacts(self) -> bool:
        """Check if a collision has occured."""


class RobotTaskEnv(gym.Env):
    """Robotic task goal env, as the junction of a task and a robot.

    Args:
        robot (PyBulletRobot): The robot.
        task (Task): The task.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targetting this postion, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Rool of the camera. Defaults to 0.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        robot: PyBulletRobot,
        task: Task,
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
        collisions: float = 0,
        timestep: float = 0,
        max_episode_steps: float = 50,
    ) -> None:
        assert robot.sim == task.sim, "The robot and the task must belong to the same simulation."
        self.sim = robot.sim
        self.render_mode = self.sim.render_mode
        self.metadata["render_fps"] = 1 / self.sim.dt
        self.collision_count = collisions
        self.max_episode_steps = max_episode_steps
        self.timestep = timestep
        self.robot = robot
        self.task = task
        observation, _ = self.reset()  # required for init; seed can be changed later
        observation_shape = observation.shape
        self.observation_space = spaces.Box(-10.0, 10.0, shape=observation_shape, dtype=np.float32)
        """
        observation_shape = observation["observation"].shape
        achieved_goal_shape = observation["achieved_goal"].shape
        desired_goal_shape = observation["achieved_goal"].shape
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(-10.0, 10.0, shape=observation_shape, dtype=np.float32),
                desired_goal=spaces.Box(-10.0, 10.0, shape=achieved_goal_shape, dtype=np.float32),
                achieved_goal=spaces.Box(-10.0, 10.0, shape=desired_goal_shape, dtype=np.float32),
            )
        )
        """
        self.action_space = self.robot.action_space
        self.compute_reward = self.task.compute_reward
        self._saved_goal = dict()  # For state saving and restoring

        self.render_width = render_width
        self.render_height = render_height
        self.render_target_position = (
            render_target_position if render_target_position is not None else np.array([0.0, 0.0, 0.0])
        )
        self.render_distance = render_distance
        self.render_yaw = render_yaw
        self.render_pitch = render_pitch
        self.render_roll = render_roll
        with self.sim.no_rendering():
            self.sim.place_visualizer(
                target_position=self.render_target_position,
                distance=self.render_distance,
                yaw=self.render_yaw,
                pitch=self.render_pitch,
            )

    def _get_obs(self) -> Dict[str, np.ndarray]:
        robot_obs = self.robot.get_obs().astype(np.float32)  # robot state
        task_obs = self.task.get_obs().astype(np.float32)  # object position, velocity, unsafe state locations etc...
        observation = np.concatenate([robot_obs, task_obs])
        achieved_goal = self.task.get_achieved_goal().astype(np.float32)
        return np.concatenate([observation, achieved_goal, self.task.get_goal().astype(np.float32)])

        '''
        return {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": self.task.get_goal().astype(np.float32),
        }
        '''

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.task.np_random, seed = seeding.np_random(seed)
        with self.sim.no_rendering():
            self.robot.reset()
            self.task.reset()
        observation = self._get_obs()
        info = {"is_success": self.task.is_success(observation[31:34], self.task.get_goal())}
        """
        info = {"is_success": self.task.is_success(observation["achieved_goal"], self.task.get_goal())}
        """
        self.collision_count = 0
        self.timestep = 0
        return observation, info

    def save_state(self) -> int:
        """Save the current state of the envrionment. Restore with `restore_state`.

        Returns:
            int: State unique identifier.
        """
        state_id = self.sim.save_state()
        self._saved_goal[state_id] = self.task.goal
        return state_id

    def restore_state(self, state_id: int) -> None:
        """Resotre the state associated with the unique identifier.

        Args:
            state_id (int): State unique identifier.
        """
        self.sim.restore_state(state_id)
        self.task.goal = self._saved_goal[state_id]

    def remove_state(self, state_id: int) -> None:
        """Remove a saved state.

        Args:
            state_id (int): State unique identifier.
        """
        self._saved_goal.pop(state_id)
        self.sim.remove_state(state_id)

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.robot.set_action(action)
        self.sim.step()
        observation = self._get_obs()
        self.timestep += 1
        # An episode is terminated if the agent has reached the target
        terminated = bool(self.task.is_success(observation[31:34], self.task.get_goal()))
        """
        terminated = bool(self.task.is_success(observation["achieved_goal"], self.task.get_goal()))
        """
        if self.timestep == self.max_episode_steps:
            truncated = True
        else:
            truncated = False
        [cost_value, exp] = self.task.compute_cost()

        collision_reward = 0
        if self.task.has_contacts():
            self.collision_count += 1
            collision_reward = float(-10)
        info = {"is_success": terminated, "cost": cost_value, "explanation": exp, "num_collisions": self.collision_count}
        reward = float(self.task.compute_reward(observation[31:34], self.task.get_goal(), info)) + collision_reward
        """
        reward = float(self.task.compute_reward(observation["achieved_goal"], self.task.get_goal(), info)) + collision_reward
        """
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self.sim.close()

    def render(self) -> Optional[np.ndarray]:
        """Render.

        If render mode is "rgb_array", return an RGB array of the scene. Else, do nothing and return None.

        Returns:
            RGB np.ndarray or None: An RGB array if mode is 'rgb_array', else None.
        """
        return self.sim.render(
            width=self.render_width,
            height=self.render_height,
            target_position=self.render_target_position,
            distance=self.render_distance,
            yaw=self.render_yaw,
            pitch=self.render_pitch,
            roll=self.render_roll,
        )