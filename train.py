import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import pygame
import matplotlib.pyplot as plt
from collections import deque
from game import JumpGame
import os

# Define the Deep Q-Network (DQN) model
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # Define the layers of the neural network
        self.fc1 = nn.Linear(input_size, 24)  # First fully connected layer
        self.fc2 = nn.Linear(24, 24)          # Second fully connected layer
        self.fc3 = nn.Linear(24, output_size) # Output layer
    
    def forward(self, x):
        # Define the forward pass of the network
        x = torch.relu(self.fc1(x))  # Apply ReLU activation to first layer
        x = torch.relu(self.fc2(x))  # Apply ReLU activation to second layer
        return self.fc3(x)           # Return the output

# Main training function
def train(episodes=800, batch_size=32, gamma=0.99, epsilon_start=0.5, epsilon_end=0.01, epsilon_decay=0.995, render=False, resume=False, checkpoint_interval=100):
    # Initialize the game environment
    env = JumpGame(render_mode=render)
    
    # Initialize the main model and target model
    model = DQN(input_size=5, output_size=2)
    target_model = DQN(input_size=5, output_size=2)
    
    # Set up the optimizer and loss function
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    # Resume training from a checkpoint if specified
    if resume:
        checkpoint = load_latest_checkpoint()
        if checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_episode = checkpoint['episode'] + 1
            epsilon = checkpoint['epsilon']
            scores = checkpoint['scores']
            print(f"Resumed training from episode {start_episode}")
        else:
            print("No checkpoint found. Starting from scratch.")
            start_episode = 0
            epsilon = epsilon_start
            scores = []
    else:
        start_episode = 0
        epsilon = epsilon_start
        scores = []

    # Initialize target model with the same weights as the main model
    target_model.load_state_dict(model.state_dict())

    # Set up rendering and replay buffer
    render_every = 10
    replay_buffer = deque(maxlen=10000)
    render_steps = 500
    
    clock = pygame.time.Clock()
 
    # Main training loop
    for episode in range(start_episode, episodes):
        state = env.reset()
        state = torch.FloatTensor(state)
        done = False
        score = 0
        step = 0
        
        should_render = (episode % render_every == 0)

        while not done:
            # Epsilon-greedy action selection
            if random.random() <= epsilon:
                action = random.randint(0, 1)  # Explore: random action
            else:
                with torch.no_grad():
                    action = torch.argmax(model(state)).item()  # Exploit: best action

            # Take action and observe next state and reward
            next_state, reward, done = env.step(action)
            next_state = torch.FloatTensor(next_state)
            score += reward

            # Render the game if needed
            if render:
                if should_render and step < render_steps:
                   env.render()
                   env.update_display()
                   clock.tick(60)  # 60 FPS
                   
                   for event in pygame.event.get():
                       if event.type == pygame.QUIT:
                           pygame.quit()
                           return
                
            # Store experience in replay buffer
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state

            # Training step
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                # Convert to torch tensors
                states = torch.stack(states)
                next_states = torch.stack(next_states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                dones = torch.FloatTensor(dones)

                # Compute Q values
                current_q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = target_model(next_states).max(1)[0]
                target_q_values = rewards + gamma * next_q_values * (1 - dones)

                # Compute loss and update model
                loss = criterion(current_q_values, target_q_values.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            step += 1

        # Update epsilon and scores
        scores.append(score)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Update target model periodically
        if episode % 10 == 0:
            target_model.load_state_dict(model.state_dict())

        print(f"Episode {episode+1}/{episodes} - Score: {score:.2f} - Epsilon: {epsilon:.2f}")

        # Save checkpoint and plot progress
        if (episode + 1) % checkpoint_interval == 0:
            save_checkpoint(episode, model, optimizer, epsilon, scores)
            print(f"Saved checkpoint at episode {episode+1}")
            plt.plot(scores)
            plt.title('Training Progress')
            plt.xlabel('Episode')
            plt.ylabel('Score')
            plt.savefig(f'models/training_progress_{episode+1}.png')

    env.close()
    plt.show()
    save_checkpoint(episodes - 1, model, optimizer, epsilon, scores)

# Function to save checkpoints
def save_checkpoint(episode, model, optimizer, epsilon, scores):
    checkpoint = {
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epsilon': epsilon,
        'scores': scores
    }
    torch.save(checkpoint, f'models/checkpoint_episode_{episode+1}.pth')

# Function to load the latest checkpoint
def load_latest_checkpoint():
    checkpoints = [f for f in os.listdir('models') if f.startswith('checkpoint_episode_')]
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return torch.load(f'models/{latest_checkpoint}')

if __name__ == "__main__":
    # Set training parameters
    render = False  # Set to True to visualize training
    resume = False  # Set to True to resume from a checkpoint
    checkpoint_interval = 200  # Save a checkpoint every 200 episodes

    # Start training
    train(render=render, resume=resume, checkpoint_interval=checkpoint_interval)