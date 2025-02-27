from typing import Any, List, Type

import cv2
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices


class ColorMazeEnv(VecEnv):
    def __init__(self, grid_size=100, margin_size=80, n_envs=1000):
        self.grid_size = grid_size
        self.margin_size = margin_size
        self.maze_size = grid_size - 2 * margin_size
        self.num_envs = n_envs
        self.window_size = 512  # The size of the PyGame window
        # Observation: ['dist_to_target_x', 'dist_to_target_y', 'color_current_cell', 'color_bottom', 'color_right',
        #               'color_top', 'color_left', 'dist_last_cell_x', 'dist_last_cell_y']
        self.obs_dim = 9

        self.observation_space = spaces.Box(
            np.ones(self.obs_dim) * -self.grid_size,
            np.ones(self.obs_dim) * self.grid_size,
            dtype=np.float64,
        )

        # We have 4 actions, corresponding to "right", "bottom", "left", "top"
        self.action_space = spaces.Discrete(4)
        self.action_to_direction = np.array(
            [
                [1, 0],  # right
                [0, 1],  # bottom
                [-1, 0],  # left
                [0, -1],  # top
            ]
        )

        # States: [0: Normal, 1: Fail, 2: Path, 3: Green_Path, 4: Border]
        self.state_to_color = np.array(
            [[255, 255, 255], [0, 0, 255], [255, 0, 0], [0, 255, 0], [0, 0, 0]]
        )
        self.state_to_masked_obs = np.array([0, 2, 2, 2, 4])

        self.grid = np.zeros(
            (n_envs, self.grid_size, self.grid_size), dtype=int
        )
        self.sample_grid = (
            np.ones(
                (n_envs, self.grid_size + 2, self.grid_size + 2), dtype=int
            )
            * 4
        )
        self.occupancy_grid = np.zeros(
            (n_envs, self.grid_size, self.grid_size), dtype=int
        )
        self.target_pos = np.zeros([n_envs, 2], dtype=int)
        self.start_pos = np.zeros([n_envs, 2], dtype=int)
        self.agent_position = np.zeros([n_envs, 2], dtype=int)
        self.last_position = np.zeros([n_envs, 2], dtype=int)
        self.last_obs = np.zeros([n_envs, self.obs_dim], dtype=float)

        self.log_data = {
            "episode_length": np.zeros([n_envs], dtype=int),
            "episode_reward": np.zeros([n_envs], dtype=float),
        }

        self.reset_envs(
            env_mask=np.ones([self.num_envs], dtype=bool), initialize_maze=True
        )

    def reset_envs(self, env_mask, initialize_maze=False):
        if initialize_maze:
            n_envs = np.sum(env_mask)
            maze = np.ones((n_envs, self.maze_size, self.maze_size))
            # Blue Paths
            maze = self.create_path(
                maze,
                state_id=2,
                mov_x_range=[1, 4],
                mov_y_range=[3, 5],
                path_x_range=[self.maze_size // 4, 3 * self.maze_size // 4],
            )
            self.grid[
                env_mask,
                self.margin_size : -self.margin_size,
                self.margin_size : -self.margin_size,
            ] = maze
            self.sample_grid[:, 1:-1, 1:-1] = self.grid.copy()

        mid_margin = self.margin_size // 2
        self.target_pos[env_mask, :] = np.array(
            [self.grid_size // 2, mid_margin]
        )[None, :]
        self.start_pos[env_mask, :] = np.array(
            [self.grid_size // 2, self.grid_size - 1 - mid_margin]
        )[None, :]
        self.agent_position[env_mask, :] = self.start_pos[env_mask, :]
        self.last_position[env_mask, :] = self.start_pos[env_mask, :]
        self.occupancy_grid[env_mask, :, :] = 0
        self.occupancy_grid[
            env_mask,
            self.agent_position[env_mask, 0],
            self.agent_position[env_mask, 1],
        ] = 1

    def create_path(
        self, maze, state_id, mov_x_range, mov_y_range, path_x_range
    ):
        n_envs = maze.shape[0]
        g_x = np.random.randint(
            path_x_range[0], path_x_range[1], size=(n_envs)
        )
        g_y = np.zeros([n_envs], dtype=int)
        finished_maze = np.zeros([n_envs], dtype=bool)
        while not finished_maze.all():
            mov_y = np.minimum(
                np.random.randint(mov_y_range[0], mov_y_range[1], size=n_envs),
                self.maze_size - g_y - 1,
            )
            mov_x = np.random.randint(
                mov_x_range[0], mov_x_range[1], size=n_envs
            ) * np.random.choice([-1, 1], size=n_envs)
            mov_y[finished_maze] = 0

            # Y-step
            max_mov_y = np.max(np.abs(mov_y))
            update_ids_y = np.rint(
                (np.arange(max_mov_y + 1)[None, :])
                / max_mov_y
                * mov_y[:, None]
                + g_y[:, None]
            ).astype(int)
            update_ids_x = np.tile(g_x[:, None], (1, max_mov_y + 1)).astype(
                int
            )
            env_ids = np.tile(
                np.arange(n_envs)[:, None], (1, max_mov_y + 1)
            ).astype(int)
            maze[env_ids, update_ids_x, update_ids_y] = state_id

            g_y += mov_y
            finished_maze = g_y >= self.maze_size - 1
            if finished_maze.all():
                break
            mov_x[finished_maze] = 0

            # X-step
            max_possible_mov_x = (
                np.maximum(g_x - path_x_range[0], path_x_range[1] - g_x) - 1
            )
            mov_x = np.minimum(np.abs(mov_x), max_possible_mov_x) * np.sign(
                mov_x
            )
            flip_direction_mask = np.logical_or(
                g_x + mov_x >= path_x_range[1], g_x + mov_x < path_x_range[0]
            )
            mov_x[flip_direction_mask] = -mov_x[flip_direction_mask]

            max_mov_x = np.max(np.abs(mov_x))

            update_ids_x = np.rint(
                (np.arange(max_mov_x)[None, :] + 1)
                / max_mov_x
                * mov_x[:, None]
                + g_x[:, None]
            ).astype(int)
            update_ids_y = np.tile(g_y[:, None], (1, max_mov_x)).astype(int)
            env_ids = np.tile(
                np.arange(n_envs)[:, None], (1, max_mov_x)
            ).astype(int)
            maze[env_ids, update_ids_x, update_ids_y] = state_id
            g_x += mov_x

        return maze

    def get_obs(self):
        # Observation: ['dist_to_target_x', 'dist_to_target_y', 'color_current_cell', 'color_bottom', 'color_right',
        #               'color_top', 'color_left', 'dist_last_cell_x', 'dist_last_cell_y']
        sample_agent_pos = self.agent_position + 1
        obs = np.zeros((self.num_envs, self.obs_dim))
        obs[:, :2] = self.target_pos - self.agent_position
        obs[:, 2] = self.sample_grid[
            np.arange(self.num_envs),
            sample_agent_pos[:, 0],
            sample_agent_pos[:, 1],
        ]
        obs[:, 3] = self.sample_grid[
            np.arange(self.num_envs),
            sample_agent_pos[:, 0],
            sample_agent_pos[:, 1] + 1,
        ]
        obs[:, 4] = self.sample_grid[
            np.arange(self.num_envs),
            sample_agent_pos[:, 0] + 1,
            sample_agent_pos[:, 1],
        ]
        obs[:, 5] = self.sample_grid[
            np.arange(self.num_envs),
            sample_agent_pos[:, 0],
            sample_agent_pos[:, 1] - 1,
        ]
        obs[:, 6] = self.sample_grid[
            np.arange(self.num_envs),
            sample_agent_pos[:, 0] - 1,
            sample_agent_pos[:, 1],
        ]
        obs[:, 7:] = self.agent_position - (self.last_position + 1)

        return obs

    def mask_student_obs(self, obs):
        student_obs = obs.copy()
        student_obs[:, 2:7] = self.state_to_masked_obs[
            (student_obs[:, 2:7]).astype(int)
        ]
        return student_obs

    def reset(self, seed=None, options=None):
        self.reset_envs(
            env_mask=np.ones([self.num_envs], dtype=bool),
            initialize_maze=False,
        )

        obs = self.get_obs()

        return obs

    def step(self, action):
        directions = self.action_to_direction[action, :]
        last_position = self.agent_position.copy()
        self.agent_position = np.clip(
            self.agent_position + directions, 0, self.grid_size - 1
        )

        success = (self.agent_position == self.target_pos).all(axis=1)
        failure = (
            self.grid[
                np.arange(self.num_envs),
                self.agent_position[:, 0],
                self.agent_position[:, 1],
            ]
            == 1
        )
        out_of_time = np.logical_and(
            self.log_data["episode_length"] >= self.grid_size**2,
            np.logical_not(np.logical_or(success, failure)),
        )

        # Reward
        current_path = (
            self.grid[
                np.arange(self.num_envs),
                self.agent_position[:, 0],
                self.agent_position[:, 1],
            ]
            == 2
        )
        first_time_cell = (
            self.occupancy_grid[
                np.arange(self.num_envs),
                self.agent_position[:, 0],
                self.agent_position[:, 1],
            ]
            == 0
        )
        path_reward = current_path * first_time_cell
        path_penalty = current_path * np.logical_not(first_time_cell)

        reward = (
            10 * success.astype(float)
            + 0.5 * path_reward.astype(float)
            - 0.5 * path_penalty.astype(float)
            - 0.1 * failure.astype(float)
        )

        # Other Updates
        self.occupancy_grid[
            np.arange(self.num_envs),
            self.agent_position[:, 0],
            self.agent_position[:, 1],
        ] += 1
        dones = np.logical_or(np.logical_or(success, failure), out_of_time)
        n_paths = (
            self.occupancy_grid[dones, :, :] * (self.grid[dones, :, :] == 2)
        ).sum(axis=(1, 2))

        self.reset_envs(env_mask=dones, initialize_maze=False)
        observation = self.get_obs()
        self.last_obs = observation.copy()
        self.last_position = last_position.copy()

        # Logging
        self.log_data["episode_reward"] += reward
        info = [{} for _ in range(self.num_envs)]
        done_ids = np.where(dones)[0]
        for i, done_id in enumerate(done_ids):
            if out_of_time[done_id]:
                info[done_id]["TimeLimit.truncated"] = True
            else:
                info[done_id]["TimeLimit.truncated"] = False
            info[done_id]["terminal_observation"] = self.last_obs[done_id]
            info[done_id]["episode"] = {
                "l": self.log_data["episode_length"][done_id],
                "r": self.log_data["episode_reward"][done_id],
                "n_paths": n_paths[i],
            }
            info[done_id]["is_success"] = success[done_id]

        self.log_data["episode_length"] += 1
        self.log_data["episode_length"][dones] = 0
        self.log_data["episode_reward"][dones] = 0

        return observation, reward, dones, info

    def render(self, env_id):
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.grid_size
        )  # The size of a single grid square in pixels

        # Draw the grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                color = self.state_to_color[int(self.grid[env_id, x, y])]
                pygame.draw.rect(
                    canvas,
                    color,
                    pygame.Rect(
                        pix_square_size * np.array([x, y]),
                        (pix_square_size, pix_square_size),
                    ),
                )

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                pix_square_size * self.target_pos[env_id, :],
                (pix_square_size, pix_square_size),
            ),
        )

        # Now we draw the start
        pygame.draw.circle(
            canvas,
            (170, 170, 170),
            (self.start_pos[env_id, :] + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 0),
            (self.agent_position[env_id, :] + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )

    def render_envs(self, env_ids):
        n_viz_env = len(env_ids)
        n_col = 5
        n_row = np.ceil(n_viz_env / n_col).astype(int)
        grid_size = 256
        margin = 5

        combined_grid = np.zeros(
            (
                n_row * grid_size + (n_row + 1) * margin,
                n_col * grid_size + (n_col + 1) * margin,
                3,
            )
        )
        for i, env_id in enumerate(env_ids):
            pos_col = i % n_col
            pos_row = i // n_col
            single_grid = self.render(env_id)
            single_grid = cv2.resize(single_grid, (grid_size, grid_size))
            combined_grid[
                pos_row * (grid_size + margin) : pos_row * (grid_size + margin)
                + grid_size,
                pos_col * (grid_size + margin) : pos_col * (grid_size + margin)
                + grid_size,
                :,
            ] = single_grid

        return combined_grid

    # Dummy Environment functions
    def step_async(self):
        raise RuntimeError("This method is not implemented")

    def step_wait(self):
        raise RuntimeError("This method is not implemented")

    def env_is_wrapped(
        self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [
            env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs
        ]

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs
    ) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [
            getattr(env_i, method_name)(*method_args, **method_kwargs)
            for env_i in target_envs
        ]

    def get_attr(self, attr_name, indices=None):
        """
        Return attribute from vectorized environment.
        :param attr_name: (str) The name of the attribute whose value to return
        :param indices: (list,int) Indices of envs to get attribute from
        :return: (list) List of values of 'attr_name' in all environments
        """
        raise RuntimeError("This method is not implemented")

    def set_attr(self, attr_name, value, indices=None):
        """
        Set attribute inside vectorized environments.
        :param attr_name: (str) The name of attribute to assign new value
        :param value: (obj) Value to assign to `attr_name`
        :param indices: (list,int) Indices of envs to assign value
        :return: (NoneType)
        """
        raise RuntimeError("This method is not implemented")

    def close(self):
        self.wrapper.close()

