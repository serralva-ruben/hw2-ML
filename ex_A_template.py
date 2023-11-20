# -*- coding: utf-8 -*-
# Revised version 02/11/2023

import numpy as np
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns

# compute a reward based on a probability
def get_reward(prob, n=10):
    reward = 0
    for i in range(n):
        if random.random() < prob:
            reward += 1
    return reward

# find the best machine based on past rewards
def get_best_arm(pastRewards, actions):
    bestArm = 0
    bestAvg = 0
    avg = 0
    for action in actions:
        # TODO: complete this function
        # Step 1: find in the history the index of all experiments where "action" was performed
        indexes = np.where(pastRewards[:, 0] == action)
        # Step 2: compute the mean reward over these experiments
        if len(indexes[0]) > 0:
            avg = np.mean(pastRewards[indexes, 1])
            if avg > bestAvg:
                bestAvg = avg
                bestArm = action
    return bestArm

# New function to run a single experiment
def run_experiment(nb_machines, nb_trials, arms):
    np.random.seed()  # Reset the random seed for each experiment
    pastRewards = np.zeros((nb_trials, 2))
    for i in range(nb_trials):
        choice = np.random.choice(nb_machines)
        reward = get_reward(arms[choice])
        pastRewards[i] = (choice, reward)
    return pastRewards

# Constants
nb_machines = 10
nb_trials = 1000
np.random.seed(0)
arms = np.random.rand(nb_machines)
print(arms)

# Running multiple experiments
num_experiments = 15  
average_winnings = []

for _ in range(num_experiments):
    pastRewards = run_experiment(nb_machines, nb_trials, arms)
    avg_win = np.mean(pastRewards[:, 1])
    average_winnings.append(avg_win)

# Plotting the average winnings of each experiment
plt.figure(figsize=(10, 5))
plt.bar(range(num_experiments), average_winnings)
plt.xlabel("Experiment Number")
plt.ylabel("Average Winnings")
plt.title("Average Winnings in Multiple Experiments")
plt.show()    

def run_experiment_with_average(nb_machines, nb_trials, arms):
    np.random.seed()  # Reset the random seed for each experiment
    pastRewards = np.zeros((nb_trials, 2))
    runningMeans = np.zeros(nb_trials)  # To store the running mean after each trial

    for i in range(nb_trials):
        choice = np.random.choice(nb_machines)
        reward = get_reward(arms[choice])
        pastRewards[i] = (choice, reward)
        runningMeans[i] = np.mean(pastRewards[:i+1, 1])  # Calculate running mean

    return pastRewards, runningMeans

# Running a single experiment and plotting the average winnings
pastRewards, runningMeans = run_experiment_with_average(nb_machines, nb_trials, arms)

plt.figure(figsize=(12, 6))
plt.xlabel("Trials")
plt.ylabel("Average Winnings")
plt.title("Evolution of Average Winnings Over Trials")
plt.plot(runningMeans)
plt.show()

def run_experiment_and_calculate_ratio(nb_machines, nb_trials, arms):
    np.random.seed()  
    pastRewards = np.zeros((nb_trials, 2))
    for i in range(nb_trials):
        choice = np.random.choice(nb_machines)
        reward = get_reward(arms[choice])
        pastRewards[i] = (choice, reward)
    avg_winning = np.mean(pastRewards[:, 1])
    return avg_winning

# Running multiple experiments and calculating the ratio
ratios = []
num_experiments = 50
best_possible_winning = np.max(arms) * 10  # Theoretical best winning per trial

for _ in range(num_experiments):
    avg_winning = run_experiment_and_calculate_ratio(nb_machines, nb_trials, arms)
    ratio = avg_winning / best_possible_winning
    ratios.append(ratio)

# Plotting the ratios
plt.figure(figsize=(10, 5))
plt.plot(ratios, marker='o')
plt.xlabel("Experiment Number")
plt.ylabel("Average Winnings to Best Possible Winnings Ratio")
plt.title("Winnings Ratio Over Multiple Experiments")
plt.show()

def run_experiment_for_each_machine(nb_machines, nb_trials, arms):
    np.random.seed()  
    machine_winnings = np.zeros((nb_machines, nb_trials))
    for i in range(nb_trials):
        for machine in range(nb_machines):
            reward = get_reward(arms[machine])
            machine_winnings[machine, i] = reward
    return machine_winnings

# Running the experiment for each machine and plotting
num_experiments = 100   
all_machine_winnings = []

for _ in range(num_experiments):
    machine_winnings = run_experiment_for_each_machine(nb_machines, nb_trials, arms)
    all_machine_winnings.append(machine_winnings)

# Reshaping the data for plotting
all_machine_winnings = np.array(all_machine_winnings).reshape(nb_machines, -1)

# Plotting using violin plots
plt.figure(figsize=(15, 8))
sns.violinplot(data=all_machine_winnings.T)
plt.xlabel("Machine Number")
plt.ylabel("Winnings")
plt.title("Distribution of Winnings per Machine")
plt.show()

def run_experiment_with_epsilon(nb_machines, nb_trials, arms, epsilon):
    np.random.seed()  
    pastRewards = np.zeros((nb_trials, 2))
    runningMeans = np.zeros(nb_trials)  # To store the running mean after each trial

    for i in range(nb_trials):
        # Epsilon-greedy strategy
        if random.random() < epsilon:
            # Exploration: choose a random machine
            choice = np.random.choice(nb_machines)
        else:
            # Exploitation: choose the best machine based on past rewards
            choice = get_best_arm(pastRewards[:i], list(range(nb_machines)))
        
        # Get the reward from the chosen machine
        reward = get_reward(arms[choice])
        pastRewards[i] = [choice, reward]
        runningMeans[i] = np.mean(pastRewards[:i+1, 1])  # Update running mean

    return pastRewards, runningMeans

# Parameters for the experiment
epsilon = 0.1  # 10% exploration and 90% exploatation

# Running the experiment with epsilon-greedy strategy
pastRewards, runningMeans = run_experiment_with_epsilon(nb_machines, nb_trials, arms, epsilon)

# Plotting the evolution of average winnings over trials
plt.figure(figsize=(12, 6))
plt.plot(runningMeans, label='Epsilon = {}'.format(epsilon))
plt.xlabel("Trials")
plt.ylabel("Average Reward")
plt.title("Evolution of Average Reward Over Trials with Epsilon-Greedy Strategy")
plt.axhline(y=np.max(arms) * 10, color='r', label='Best Possible Average Reward')
plt.ylim(ymin=0)
plt.legend()
plt.show()

def calculate_winnings_ratio(pastRewards, arms):
    best_possible_winning = np.max(arms) * 10  # Theoretical best per trial
    avg_winning = np.mean(pastRewards[:, 1])
    return avg_winning / best_possible_winning

# Parameters for the experiments
num_experiments = 40  # Number of experiments to run
ratios = []

# Running multiple experiments
for _ in range(num_experiments):
    pastRewards, _ = run_experiment_with_epsilon(nb_machines, nb_trials, arms, epsilon)
    ratio = calculate_winnings_ratio(pastRewards, arms)
    ratios.append(ratio)

# Plotting the average winnings to best possible winnings ratio over multiple experiments
plt.figure(figsize=(12, 6))
plt.plot(ratios, marker='o', label='Epsilon-Greedy Strategy')
plt.axhline(y=1, color='r', linestyle='--', label='Best Possible Winnings Ratio')
plt.xlabel("Experiment Number")
plt.ylabel("Average Winnings to Best Possible Winnings Ratio")
plt.title("Average Winnings Ratio with Epsilon-Greedy Strategy Over Multiple Experiments")
plt.legend()
plt.show()
