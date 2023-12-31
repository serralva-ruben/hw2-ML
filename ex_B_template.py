# -*- coding: utf-8 -*-
# Revised version 02/11/2023

import gymnasium as gym
import matplotlib.pyplot as plt
import time
import random
import numpy as np

RENDER_MODE = "graphic"  # choose between "graphic" or text;  graphic mode needs the pygame package to be installed
RENDER_FREQUENCY = 0.01  # output the game state at most every X seconds

KNOW_OPTIMAL_REWARD = -13 #we know that the optimal reward is -13 because each step without falling has a reward of -1, and since the shortest path takes 13 steps, that means that best possible reward if -13

ALPHA = 0.95
EPSILON = 0
GAMMA = 0.9

env = gym.make('CliffWalking-v0', render_mode = "rgb_array" if RENDER_MODE == "graphic" else "ansi") # environment setup

#agent that uses Equiprobable Random Policy
class ERPAgent:
    def select_action(self, _):
        #Since our actions are represented as integers from 0 to 3, we generate a random integer between 0 and 3
        return np.random.choice(4)

    def update(self, old_state, action, reward, new_state):
        # The ERP Agent doesn't learn, so there's no need for an update
        pass
#Agent that uses QLearning
class QLearningAgent:
    def __init__(self, state_space_size, action_space_size, alpha, epsilon, gamma):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        #2D array to store actions and reward
        self.value_table = np.zeros((state_space_size, action_space_size))

    # decide what action to take in the provided state by applying a certain policy
    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.value_table.shape[1])  # Exploration
        else:
            return self.policy(state)  # Exploitation
    
    def policy(self, state):
        best_actions = np.argwhere(self.value_table[state] == np.amax(self.value_table[state])).flatten()
        return np.random.choice(best_actions)  # Randomly choose among best actions
    
    # do the learning (e.g. update the Q-values, or collect data to do off-policy learning)
    def update(self, old_state, action, reward, new_state):
        best_next_action = np.argmax(self.value_table[new_state])
        td_target = reward + self.gamma * self.value_table[new_state][best_next_action]
        td_error = td_target - self.value_table[old_state][action]
        self.value_table[old_state][action] += self.alpha * td_error

def to_coord(state_id):
    # we return the (x, y) coorinates, in the description page the use [y, x] to describe the locations
    return (state_id % 12, state_id // 12)

actions = {
    0: "up",
    1: "right",
    2: "down",
    3: "left"
}

#game rendering related code
def display_game(env, wait_time_s = RENDER_FREQUENCY):
    if RENDER_MODE == "graphic":
        img = env.render()  # show the current state of the game (the environment)
        plt.ion()
        if display_game.plt_im is None:
            display_game.plt_im = plt.imshow(img)
        else:
            display_game.plt_im.set_data(img)
        plt.draw()
        plt.show()
        plt.pause(0.001)
    else:
        img_ansi = env.render()  # show the current state of the game (the environment)
        print(img_ansi)
    time.sleep(wait_time_s)
display_game.plt_im = None

## Returns the number of steps taken and the ending reason (-1 if fallen off, 0 if survived but out of steps, 1 if reached goal)
## runs a single episode
def run_episode(agent, max_steps = 1000, muted = False):
    observation, info = env.reset() # restart the game
    total_reward = 0
    if not muted:
        print(f"Starting in position: {to_coord(observation)}")

    for k in range(max_steps):
        # HINT: you may want to disable displaying of the game to run the experiments
        if not muted:
            display_game(env)

        action = agent.select_action(observation)  # select an action based on the current state
        new_observation, reward, terminated, truncated, info = env.step(action)  # perform the action and observe what happens 
        total_reward+=reward
        fallen_off_cliff = (reward == -100)  # Beware! we cannot check for cliff state because the environment automatically returns us to the starting position when falling off a cliff
        goal_reached = terminated  # if we reach the goal state, the environment returns terminated = True 

        agent.update(observation, action, reward, new_observation)  # perform some learning if the agent is capable of it

        if not muted:
            print(f"Action determined by agent: {actions[action]}")
            print(f"Reward for action: {actions[action]} in state: {observation} is: {reward}")
            print(f"New state is: {new_observation}")

        if goal_reached:
            if not muted:
                print(f"Goal reached after terminated after {k+1} steps\n\n")
            return k+1, 1, total_reward  # we reached the goal
        elif fallen_off_cliff:
            if not muted:
                print(f"Fell off the cliff after {k+1} steps\n\n")
            
            return k+1, -1, total_reward  # we fell off the cliff

        observation = new_observation

    if not muted:
        print(f"Survived for {k+1} steps but goal not reached\n\n")
    return k+1, 0, total_reward  # we survived but did not reach the goal

#run multiple experiments
def run_experiments(agent, experiments = 100, episodes = 500):
    averages, bests, win_counts, optimal_reward_counts = [], [], 0, 0

    for _ in range(experiments):
        avg_reward, best_reward, wins, _, optimal_reward_count = run_experiment(agent, episodes)
        averages.append(avg_reward)
        bests.append(best_reward)
        win_counts += wins
        optimal_reward_counts += optimal_reward_count
    return averages, bests, win_counts/experiments, optimal_reward_counts/experiments
#run single experiment
def run_experiment(agent, episodes = 500):
    win_count, mute_output, rewards, optimal_reward_count = 0, True, [], 0
    
    for _ in range(episodes):
        _, reason, reward = run_episode(agent, muted=mute_output)
        if reason == 1:
            if reward == KNOW_OPTIMAL_REWARD:
                optimal_reward_count += 1
            win_count += 1
        rewards.append(reward)
    average_reward = sum(rewards) / episodes
    best_reward = max(rewards)
    return average_reward, best_reward, win_count, rewards, optimal_reward_count
#perform grid search to find the best alpha and epsilon
def grid_search(env, episodes = 500):
    best_metric = -float('inf')
    best_alpha = 0
    best_epsilon = 0
    #we test the agent for every alpha and epsilon combination in [0,1] with steps of 0.05 and we compare the performance of the average of 10 runs with every epsilon alpha combination to find the best
    for alpha in np.arange(0, 1.05, 0.05):
        for epsilon in np.arange(0, 1.05, 0.05):
            print(f"Testing alpha: {alpha}, epsilon: {epsilon}")
            agent = QLearningAgent(env.observation_space.n, env.action_space.n, alpha, epsilon, GAMMA)
            metric_sum = 0

            #We average out for 10 runs so we hget more stable results
            for _ in range(10):
                avg_reward, _, _, _, _ = run_experiment(agent, episodes)
                metric_sum += avg_reward  # Using the reward as the metric used to evaluate the performance
            #after adding the 10 rewards obtained we divide by 10 to get an average
            average_metric = metric_sum / 10
            print(best_metric)
            #if the average metric we got for this combination is better than the best combination we found before, we replace the alpha and epsilon
            if average_metric > best_metric:
                best_metric = average_metric
                best_alpha = alpha
                best_epsilon = epsilon

    return best_alpha, best_epsilon, best_metric

if __name__ == "__main__":

    EXPERIMENTS = 100
    EPISODES = 500
    run_grid_search = False  # Set to False to use alpha and epsilon defined in the beginning otherwise it will perform the grid search
    use_q_learning = True   # Set false to use ERP
    if run_grid_search:
        best_alpha, best_epsilon, best_metric = grid_search(env)
        print(f"Best alpha: {best_alpha}, Best epsilon: {best_epsilon}, Best metric: {best_metric}")

    else:
        agent = QLearningAgent(env.observation_space.n, env.action_space.n, ALPHA, EPSILON, GAMMA) if use_q_learning else ERPAgent()
        run_multiple_experiments = True
        
        if run_multiple_experiments:
            averages, bests, win_rate, optimal_reward_rate = run_experiments(agent, EXPERIMENTS, EPISODES)

            print(f"Average Reward: {np.mean(averages)}")
            print(f"Best Reward: {max(bests)}")
            print(f"Average Times Goal Reached per Experiment: {win_rate}")
            print(f"Average Times Optimal Reward Achieved per Experiment: {optimal_reward_rate}")

            plt.scatter(range(EXPERIMENTS), averages, label='Average Reward')
            plt.scatter(range(EXPERIMENTS), bests, label='Best Reward')
            plt.xlabel('Experiment')
            plt.ylabel('Reward')
            plt.title('Rewards per Experiment')
            plt.legend()
            plt.show()

        else:
            avg_reward, best_reward, total_wins, rewards = run_experiment(agent, EPISODES)

            print(f"Average Reward: {avg_reward}")
            print(f"Best Reward: {best_reward}")
            print(f"Times Goal Reached: {total_wins}")

            plt.scatter(range(EPISODES), rewards, label='Best Reward')
            plt.xlabel('Episodes')
            plt.ylabel('Reward')
            plt.title('Rewards per Episode')
            plt.legend()
            plt.show()
        
env.close() # end the game