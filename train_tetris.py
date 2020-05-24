import gym
import signal

import threading

from baselines import deepq

end = False

def wait_for_enter():
    global end
    input()
    print('Ending...')
    end = True

def callback(lcl, _glb):
    global end
    if end:
        print('YEET')
    return end

def main():
    threading.Thread(target=wait_for_enter).start()
    env = gym.make('gym_tetris:tetris-v0')
    act = deepq.learn(
        env,
        network='tetris_cnn',
        lr=1e-3,
        total_timesteps=50000,
        buffer_size=50000,
        exploration_fraction=0.3,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback,
        load_path='tetris_model.pkl'
    )
    print("Saving model to tetris_model.pkl")
    act.save("tetris_model.pkl")

if __name__ == '__main__':
    main()
