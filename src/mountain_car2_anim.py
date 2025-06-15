import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# Constants
min_position = -1.5
max_position = 1.8 # Change this to see more or less of the right hill
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

# Data
state_action_pair = ([], [])
episode_data = [] #hold episode data over runs



# Initialize figure
fig, ax = plt.subplots()
ax.grid(True)
x = np.linspace(min_position, max_position, 100)
y = hill(x)
ax.plot(x, y, 'k')  # Draws the hill
car, = ax.plot([], [], 'ro', markersize=10)  # Car marker


# Q-Learning Parameters
n_bins = 30      #no of bins
alpha = 0.3     # Learning rate
gamma = 0.99     # Discount factor
epsilon = 0.1    # Exploration rate
epsilon_decay = 0.94    #decay rate
epsilon_min = 0.05      # minum decay 
num_explore = 0     #holds the exploration steps
num_exploit = 0     #stores exploitation steps
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

# Simulation parameters
num_steps = 1000  # change if needed
positions = []
episode_steps = 0
n_episodes = 1
done = False
max_episodes = 20
runs = 1
max_runs = 10
max_steps = 50000


# Animation update function
def update(frame):
    global position, velocity, outcome, action, episode_steps, Q, epsilon, runs, done, num_explore, num_exploit, n_episodes, episode_data, max_steps

    if n_episodes <= max_episodes:  
        #checks if an episode is completed
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
        
        #descritze current position
        gradient = np.cos(3 * position)
        state = discretize_gradient(gradient)

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
        gradient = np.cos(3 * position)

        # Update velocity
        velocity += applied_force - (gravity * gradient)
        velocity = np.clip(velocity, -max_speed, max_speed)

        # Update position
        position += velocity
        position = np.clip(position, min_position, max_position)

        #REWARD
        # reward = get_reward(position)
        reward = -0.01
        # Reset if going over the wrong hill
        if position <= -1.5 or position >= goal_position_hill + 0.005:
            outcome = "Failure"
            done = True
            reward = -1.0
            
        # Stops once on the hill with Success
        if goal_position_hill - 0.005 < position < goal_position_hill + 0.005:
            # ani.event_source.stop()
            outcome = "Success"
            done = True
            reward = 1.0
            
    
        """Q Learning"""
        #new state after based on the action taken
        gradient = np.cos(3*position)
        next_state = discretize_gradient(gradient)
        best_next_q = np.max(Q[next_state])

        #Q learning update
        Q[state, action_space.index(action)] += alpha * (reward + gamma * best_next_q - Q[state, action_space.index(action)])

        episode_steps += 1

        if episode_steps > max_steps:
            done = True
            outcome = ""
        
        
        # Garage case
        # This is needed for task3 onwards
        #if goal_position_valley - 0.005 < position < goal_position_valley + 0.005 and abs(velocity) < 0.005:
        #    ani.event_source.stop()
        #    outcome = "Success"


        positions.append(position)  # Save the current position for later analysis (if wanted)
        car.set_data(position, hill(position))  # Update car position
    
    #plot graph at the end of every episode
    #stops animation if runs exceeds 10 times
    elif runs <= max_runs:
        plotQ_once(episode_data)
        plotQ(episode_data)
        episode_data = []
        n_episodes = 1
        runs += 1
        epsilon = 0.1
        Q = np.zeros((n_bins, len(action_space)))
    
    if runs > max_runs:
        ani.event_source.stop()

    return car,

# Set plot limits
ax.set_xlim(min_position, max_position)
ax.set_ylim(-0.6, 0.6)
ax.set_xlabel("Position")
ax.set_ylabel("Height")
ax.set_title("Mountain Car Simulation")

#plot the combined Q graph
#Q - performance - Graph
fig2, ax2 = plt.subplots()
ax2.set_title("Q performance Combined")
ax2.grid(True)
def plotQ(data):
    global ax2
    ax2.plot(data, marker="*", label=f'Run-{runs}')
    # ax2.set_ylim(0, 2000)
    # ax2.set_xlim(0, 10)
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
    # ax3.set_xlim(0, 10)
    ax3.set_title(f"Q performance: Runs: {runs}")
    ax3.grid(True)
    fig3.show()

# Create animation
ani = animation.FuncAnimation(fig, update, frames=num_steps+episode_steps, interval=1, blit=True)
plt.show()


