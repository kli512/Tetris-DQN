import gym
from baselines import deepq

env = gym.make('gym_tetris:tetris-v0')

actor = deepq.learn(env, network='tetris_cnn', total_timesteps=0, load_path="tetris_model.pkl")

observation = env.reset()
while True:
    env.render()
    # action = env.action_space.sample()  # agent.act(observation)
    action = actor.step(observation)[0][0]
    print(f'Steps: {env.game.time_passed}\n Action: "{env._actions[action]}""')
    print()
    observation, reward, done, info = env.step(action)
    if done:
        print("Episode finished")
        break

env.close()
