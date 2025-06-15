import numpy as np
import math
import matplotlib.pyplot as plt
import time

# Constants
min_position = -1.5
max_position = 1.8
goal_position_hill = math.pi / 6
force = 0.001
gravity = 0.0025
max_speed = 0.07

# Q-learning Parameters
n_bins = 30
alpha = 0.1
gamma = 0.99
epsilon = 0.1
epsilon_decay = 0.94
epsilon_min = 0.05
action_space = [-1, 1]


# Discretization bins
gradient_bins = np.linspace(-1, 1, n_bins)
action_space = [-1, 1]

# Q-table: shape (n_bins, 2 actions)
Q = np.zeros((n_bins, len(action_space)))

# Discretize gradient
def discretize_gradient(gradient):
    state =  np.digitize(gradient, gradient_bins) - 1
    return state


    
# Hill environment
def hill(x):
    return np.sin(3 * x) * 0.5


# Q-learning Training Loop
def train_q_learning(runs=5, episodes=10):
    all_runs_data = []

    for run in range(runs):
        episode_data = []
        global epsilon, Q, position
        epsilon = 0.1
        Q = np.zeros((n_bins, len(action_space)))
        print(f"\n=== Run {run+1}/{runs} ===")
        
        for episode in range(episodes):
            position = -math.pi / 6  # start position
            velocity = 0.0
            steps = 0
            done = False

            while not done:
                gradient = np.cos(3 * position)
                state = discretize_gradient(gradient)

                # Epsilon-greedy policy
                if np.random.rand() < epsilon:
                    action = np.random.choice(action_space)
                else:
                    action = action_space[np.argmax(Q[state])]

                applied_force = action * force
                gradient = np.cos(3 * position)
                velocity += applied_force - gravity * gradient
                velocity = np.clip(velocity, -max_speed, max_speed)

                position += velocity
                position = np.clip(position, min_position, max_position)

                reward = -0.01
                done = False

                if position <= min_position or position >= goal_position_hill + 0.005:
                    reward = -1
                    done = True
                    outcome = "Fail"
                if goal_position_hill - 0.005 < position < goal_position_hill + 0.005:
                    reward = 1
                    done = True
                    outcome = "Success"

                gradient = np.cos(3*position)
                next_state = discretize_gradient(gradient)
                best_next_q = np.max(Q[next_state])
                Q[state, action_space.index(action)] += alpha * (reward + gamma * best_next_q - Q[state, action_space.index(action)])

                steps += 1
                if steps > 50000:
                    done = True
                    outcome = "Timeout"

            print(f"Episode {episode+1} - Steps: {steps} - Outcome: {outcome} - Epsilon: {epsilon:.3f}")
            episode_data.append(steps)

            # Decay epsilon
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
        all_runs_data.append(episode_data)

    return all_runs_data

# Run training and collect results
performance_data = train_q_learning(runs=10, episodes=20)

# Plot performance
plt.figure(figsize=(10, 5))
for i, run_data in enumerate(performance_data):
    plt.plot(run_data, marker='o', label=f'Run {i+1}')
plt.xlabel('Episode')
plt.ylabel('Steps to Goal')
plt.title('Q-learning Performance Over Runs')
plt.grid(True)
plt.legend()
plt.show()
