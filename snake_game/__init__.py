from gym.envs.registration import register

register(
    id="SnakeGame-v0",
    entry_point="snake_game.envs.SnakeGame",
)