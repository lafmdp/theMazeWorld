from maze_env import Maze
from rlbrain import table

def run():
    zyy = 'lover'
    RL = table(list(range(env.n_actions)), epsi=0.95)

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

        print('the episode:', episode)
        if done is 'hell':
            print('failed')
        elif done is 'oval':
            print('success!')
        print(RL.q_table,'\n')

    print('game over!')
    env.destroy()

if __name__ == "__main__":
    env = Maze()

    env.after(100, run)
    env.mainloop()