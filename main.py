from maze_env import Maze
from rlbrain import robot

def run():
    zyy = 'lover'
    RL = robot(list(range(env.n_actions)))

    succ = 0
    for episode in range(1000):
        obs = env.reset()

        while zyy is 'lover':
            env.render()

            action = RL.choose_action(str(obs))

            obs_,r,done = env.step(action)

            RL.learning(str(obs), action, r, str(obs_))

            obs = obs_

            if done is not None:
                break

        if done is 'oval':
            succ += 1
        print('the episode:',episode,'  success rate:',(succ/(episode+1)))

    print('game over!')
    env.destroy()

if __name__ == "__main__":
    env = Maze()

    env.after(100, run)
    env.mainloop()