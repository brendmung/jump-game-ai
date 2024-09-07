import pygame
import torch
from game import JumpGame
from train import DQN
import time
import os

# Function to load the latest trained model
def load_latest_model():
    # Get all checkpoint files in the 'models' directory
    checkpoints = [f for f in os.listdir('models') if f.startswith('checkpoint_episode_')]
    if not checkpoints:
        print("No checkpoints found. Please train the model first.")
        return None
    # Find the latest checkpoint based on episode number
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # Load the checkpoint
    checkpoint = torch.load(f'models/{latest_checkpoint}')
    # Return only the model's state dictionary
    return checkpoint['model_state_dict']

# Main evaluation function
def evaluate(num_games=5, render=True):
    # Initialize the game environment
    env = JumpGame(render_mode=render)
    # Create a new DQN model
    model = DQN(input_size=5, output_size=2)
    
    # Load the latest model state
    model_state_dict = load_latest_model()
    if model_state_dict is None:
        return
    
    # Load the state into the model and set it to evaluation mode
    model.load_state_dict(model_state_dict)
    model.eval()

    # Run the evaluation for the specified number of games
    for game in range(num_games):
        state = env.reset()
        done = False
        total_reward = 0
        clock = pygame.time.Clock()

        # Play a single game
        while not done:
            # Convert the state to a PyTorch tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            # Use the model to choose an action (no gradient computation needed)
            with torch.no_grad():
                action = torch.argmax(model(state_tensor)).item()
            # Take the action in the environment
            state, reward, done = env.step(action)
            total_reward += reward

            # Render the game if specified
            if render:
                env.render()
                env.update_display()
                
                # Handle Pygame events (e.g., window close)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                clock.tick(60)  # Limit to 60 FPS

        # Print the results of the game
        print(f"Game {game+1} - Total Reward: {total_reward:.2f}, Score: {env.score}")
        
        # Pause between games if rendering
        if render:
            time.sleep(1)

    # Quit Pygame if we were rendering
    if render:
        pygame.quit()

if __name__ == "__main__":
    # Configuration variables
    render = True  # Set to False if you don't want to visualize
    num_games = 5  # Number of games to evaluate

    try:
        # Run the evaluation
        evaluate(num_games=num_games, render=render)
    except KeyboardInterrupt:
        print("Evaluation interrupted by user")
    finally:
        # Ensure Pygame quits properly
        pygame.quit()