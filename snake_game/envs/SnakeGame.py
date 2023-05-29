from typing import Any, Dict, Tuple
import gym
from gym import spaces
from gym.core import _ActionType, _OperationType
from gym.utils.seeding import np_random
import pygame
import numpy as np

class SnakeGame(gym.Env):
    metadata = {"render_modes":["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size=40, render_mode=None):
        super(SnakeGame, self).__init__()
        self.size = size
        self.window_size = 512
        # Observation are dictionaries with the snake's and the apple's location.
        self.observation_space = spaces.Dict(
            {
                "snake": spaces.Box(0, size-1, shape=(2,), dtype=int),
                "apple": spaces.Box(0, size-1, shape=(2,), dtype=int),
            }
        )
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),        #moves right
            1: np.array([0, 1]),        #moves up
            2: np.array([-1, 0]),       #moves left
            3: np.array([0, -1]),       #moves down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
    
    def _get_obs(self):
        return {"snake": self._snake_position, "apple": self._apple_position}
    
    def _get_info(self):
        return {"distance": np.linalg.norm(self._snake)}
    
    def reset(self):
        self._snake_position = np.array([self.size // 2, self.size // 2])
        self._apple_position = self._snake_position
        while np.array_equal(self._snake_position, self._apple_position):
            self._apple_position = np.array([np.random.randint(0, self.size), np.random.randint(0, self.size)])
        
        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        
        return obs, info
    
    def step(self, action):
        direction = self._action_to_direction[action]
        self._snake_position = np.clip(
            self._snake_position + direction, 0, self.size-1
        )
        done = np.array_equal(self._snake_position, self._apple_position)
        reward = 1 if done else 0
        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, done, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )
        # For the apple
        pygame.draw.rect(
            canvas,
            (255,0,0),
            pygame.Rect(
            pix_square_size * self._apple_position,
            (pix_square_size, pix_square_size),
            ),
        )
        # For the snake
        pygame.draw.circle(
            canvas,
            (0, 255, 0),
            (self._snake_position+0.5)*pix_square_size,
            pix_square_size/3,
        )
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axex=(1, 0, 2)
            )
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()