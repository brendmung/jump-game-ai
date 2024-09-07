# Jump Game AI: A Learning Platform for Reinforcement Learning

This project provides a simple yet engaging platform for learning and experimenting with reinforcement learning techniques, specifically Deep Q-Networks (DQN).

## Overview

This repository contains a Pygame-based Jump Game along with a DQN AI that learns to play the game. It's designed as an educational tool for those interested in AI and reinforcement learning, allowing to visualize the training process, experiment with parameters, and understand how AI agents improve over time.

## Key Features

1. **Interactive Learning Environment**: The Jump Game provides a simple yet challenging environment for reinforcement learning.

2. **Visualization with Render Function**: Watch the AI learn in real-time or play the game yourself.

3. **Pre-trained Model**: Start with a pre-trained model to see advanced gameplay or use it as a baseline for further training.

4. **Customizable Parameters**: Easily adjust learning parameters to experiment with different training strategies.

5. **Comprehensive Documentation**: Detailed explanations and comments to help learners understand the code and concepts.

## Getting Started

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/jump-game-ai.git
   cd jump-game-ai
   ```

2. Set up a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Playing the Game

Run the game to play it yourself:

```
python game.py
```

### Training the AI

Start training the AI:

```
python train.py
```

#### Render Function in Training

The `render` parameter in `train.py` controls visualization during training:

- `render=True`: Periodically displays the game, showing how the AI improves over time.
- `render=False`: Trains without visualization for faster processing.

Example:
```python
train(render=True, episodes=1000)
```

### Evaluating the AI

Evaluate the trained AI:

```
python src/eval.py
```

#### Render Function in Evaluation

Control game visualization during evaluation with the `render` parameter:

- `render=True`: Watch the AI play the game.
- `render=False`: Run evaluation without visual output for quick performance testing.

Example:
```python
evaluate(num_games=5, render=True)
```

## Learning and Experimentation

1. **Start with the Pre-trained Model**: 
   - Load the pre-trained model to see example gameplay (1200 epochs).
   - Use it as a starting point for transfer learning or fine-tuning.

2. **Experiment with Hyperparameters**:
   - Adjust learning rate, discount factor, epsilon values, etc.
   - Observe how different parameters affect learning speed and performance.

3. **Modify the Reward Structure**:
   - Experiment with different reward schemes in `game.py` to see how it impacts the AI's behavior.

4. **Visualize the Learning Process**:
   - Use the render function to watch the AI's progression from random movements to skilled gameplay.

5. **Analyze Performance Metrics**:
   - Study the training curves generated during the training process.
   - Compare different runs with varied parameters.

## License

This project is open source and available under the [MIT License](LICENSE).

Happy learning and experimenting with Jump Game AI!
