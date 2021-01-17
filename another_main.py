from DQN_EX.RL_brain import DeepQNetwork
from DQN_EX.another_maze import Mz

import sys

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


def run_mz():
    step = 0
    for episode in range(300):

        observation = env.reset()

        while True:
            env.render()

            action = RL.choose_action(observation)

            num = int(observation[0])

            observation_, reward, done = env.step(num, action)

            if num == 2:
                env.doEscape()

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_
            if done:
                print("total step: "+str(step))
                break
            step += 1
    print("Game Over")
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Mz()
    env.sgg = tk.Label(
        text="learning_rate:0.1  reward_deacy=0.8  e_greedy=0.8")
    env.sgg.pack(side=tk.TOP)
    env.sgg2 = tk.Label(text="replace_target_iter=200 memory_size=2000")
    env.sgg2.pack(side=tk.TOP)
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.1,
                      reward_decay=0.8,
                      e_greedy=0.8,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    # env.after(100, run_mz)
    env.com = tk.Button(text="Go!", command=run_mz, width=20)
    env.com.pack(side=tk.BOTTOM)
    env.mainloop()
    RL.plot_cost()
