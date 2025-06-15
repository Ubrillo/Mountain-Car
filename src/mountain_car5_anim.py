import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# Constants
min_position = -1.5
max_position = 2.0 # Change this to see more or less of the right hill
goal_position_hill = math.pi/6
goal_position_valley = 3*math.pi/6 # Task 3 onwards
force = 0.001
gravity = 0.0025
max_speed = 0.07

# Initialize car
start_position = -1*math.pi/6 # Start position
position = start_position
velocity = 0.0  # Start velocity


# Hill function (for visualization)
def hill(x):
    return np.sin(3 * x) * 0.5


# Simulation parameters
num_steps = 1000  # change if needed
positions = []
runs = 1
episode_steps = 0
n_episodes = 1
done = False

# Data
state_action_pair = ([], [])
episode_data = []


# Initialize figure
fig, ax = plt.subplots()
ax.grid(True)
x = np.linspace(min_position, max_position, 100)
y = hill(x)
ax.plot(x, y, 'k')  # Draws the hill
car, = ax.plot([], [], 'ro', markersize=10)  # Car marker

# Q-learning Parameters
n_bins = 60
alpha = 0.3
gamma = 0.99
epsilon = 0.1
epsilon_decay = 0.94
epsilon_min = 0.05

#action space
action_space = [-1, 1]

# Discretization bins
margin_red = 0.5 #[0.5 , max_position]
margin_blue= -0.5 # [-0.5, 0.5)
margin_green = -1.5 #[-1.5, 0.5)

# Discretization bins
margin_red = 0.5 #[0.5 , max_position]
margin_blue= -0.5 # [-0.5, 0.5)
margin_green = -1.5 #[-1.5, 0.5)

# Q-table: shape (n_bins, 2 actions)
Q = np.zeros((n_bins, len(action_space)))

#green and red bin
gradient_bins_gr = np.linspace(0, -1, n_bins//3)
#blue bin
gradient_bins_blue = np.linspace(0, 1, n_bins//3)


# Discretize gradient
def discretize_gradient(gradient):
    state =  np.digitize(gradient, gradient_bins_gr) -1

    #if position falls in red bin
    if position >= margin_red:
        state += int(n_bins//1.5)
    
    #position falls in blue bin
    elif position >= margin_blue:
        state =  np.digitize(gradient, gradient_bins_blue) -1
        state += n_bins//3
    
    return state

num_exploit = 1
num_explore = 1 


# Animation update function
def update(frame):
    global position, velocity, outcome, action_space, episode_steps, Q, epsilon, runs, done, num_explore, num_exploit, n_episodes, episode_data
    if n_episodes <= 20:
        if done:
            print(f'Runs: {runs} outcome: {outcome} - Steps: {episode_steps} - explore: {num_explore} - exploit: {num_exploit}, n_episodes: {n_episodes} E:{epsilon}')
            print(Q)
            done = False
            num_explore = 0
            num_exploit = 0
            episode_data.append(episode_steps)
            episode_steps = 0
            n_episodes += 1
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            time.sleep(1)
            position = start_position

        #descritize current gradient
        gradient = np.cos(3 * position)
        state = discretize_gradient(gradient)

        action_space = [-1, 1]
        if position >= goal_position_hill+0.15:
            action_space = [-1.9, -1.8]

        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.choice(action_space)  # Exploration
            num_explore += 1
        else:
            action = action_space[np.argmax(Q[state])]  # Exploitation
            num_exploit += 1

        # Edit this block for later data/state changes and perhaps make this an actual good data structure
        state_action_pair[0].append(position)
        state_action_pair[1].append(action)     
        applied_force = action * force

        # Update velocity
        velocity += applied_force - (gravity * gradient)
        velocity = np.clip(velocity, -max_speed, max_speed)

        # Update position
        position += velocity
        position = np.clip(position, min_position, max_position)    

        if position >= margin_red:
            reward = -0.01
        
        elif position >= margin_blue:
            reward = -0.05
        else:
            reward = -0.1


        if position <= min_position:
            reward = -5.0
            done = True
            outcome = "Fail"

        if position >= goal_position_valley + 0.005:
            reward = -0.5
            done = True
            outcome = "Fail"
        
        if goal_position_valley - 0.005 < position < goal_position_valley + 0.005 and abs(velocity) < 0.005:
            reward = 5.0
            done = True
            outcome = "Success"
    
        """Q Learning"""
        gradient = np.cos(3 * position)
        next_state = discretize_gradient(gradient) #new state after based on the action taken

        # Q-Learning update
        best_next_q = np.max(Q[next_state])
        Q[state, action_space.index(action)] += alpha * (reward + gamma * best_next_q - Q[state, action_space.index(action)])

        if done:
            print(f'state: {state} Q: {Q[state]} reward: {reward}')

        positions.append(position)  # Save the current position for later analysis (if wanted)
        car.set_data(position, hill(position))  # Update car position

        episode_steps += 1
        if episode_steps > 50000:
            done = True
            outcome="Failure"

    elif runs <= 10:
        plotQ_once(episode_data)
        plotQ(episode_data)
        episode_data = []
        n_episodes = 1
        runs += 1
        Q = np.zeros((n_bins, len(action_space)))
        epsilon = 0.1
    
    if runs > 10:
        ani.event_source.stop()

    return car,

# Set plot limits
ax.set_xlim(min_position, max_position)
ax.set_ylim(-0.6, 0.6)
ax.set_xlabel("Position")
ax.set_ylabel("Height")
ax.set_title("Mountain Car Simulation")


#Q - performance - Graph
fig2, ax2 = plt.subplots()
ax2.set_title("Q performance Combined")
ax2.grid(True)

#plot the combined Q graph
def plotQ(data):
    global ax2
    ax2.plot(data, marker="*", label=f'Run-{runs}')
    ax2.set_xlim(0, 10)
    ax2.set_xlabel("Episodes")
    ax2.set_ylabel("Steps")
    ax2.grid(True)
    ax2.legend()
    fig2.show()

#plot graph once per episode
def plotQ_once(data):
    fig3, ax3 = plt.subplots()
    ax3.plot(data, marker='*')
    ax3.set_xlabel("Episodes")
    ax3.set_ylabel("Steps")
    ax3.set_xlim(0, 10)
    ax3.set_title(f"Q performance: Runs: {runs}")
    ax3.grid(True)
    fig3.show()

# Create animation
ani = animation.FuncAnimation(fig, update, frames=num_steps+episode_steps, interval=1, blit=True)
plt.show()
