import pygame
import random
import numpy as np
import os

# Game settings
SCREEN_WIDTH = 480
SCREEN_HEIGHT = 300
GROUND_HEIGHT = SCREEN_HEIGHT - 50
BOX_SIZE = 40
OBSTACLE_WIDTH = 20
OBSTACLE_HEIGHT = 50
FLOATING_OBSTACLE_HEIGHT = 30
FLOATING_OBSTACLE_CHANCE = 0.2
FPS = 60

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
BROWN = (139, 69, 19)  # Color for the ground

class JumpGame:
    def __init__(self, render_mode=False):
        # Initialize the game, optionally with rendering
        self.render_mode = render_mode
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Jump Game")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
        self.high_score = self.load_high_score()
        self.reset()

    def reset(self):
        # Reset the game state
        self.box_x = 40
        self.box_y = GROUND_HEIGHT - BOX_SIZE
        self.box_vel_y = 0
        self.is_jumping = False
        self.obstacles = []
        self.score = 0
        self.game_speed = 5
        self.time_since_last_spawn = 0
        self.spawn_obstacle()  # Spawn the first obstacle
        return self._get_state()

    def load_high_score(self):
        # Load the high score from a file
        if os.path.exists('high_score.txt'):
            with open('high_score.txt', 'r') as f:
                return int(f.read())
        return 0

    def save_high_score(self):
        # Save the high score to a file
        with open('high_score.txt', 'w') as f:
            f.write(str(self.high_score))

    def spawn_obstacle(self):
        # Spawn a new obstacle (either floating or ground-based)
        if random.random() < FLOATING_OBSTACLE_CHANCE:
            obstacle_type = 'floating'
            obstacle_height = FLOATING_OBSTACLE_HEIGHT
            y = random.randint(GROUND_HEIGHT - BOX_SIZE - 100, GROUND_HEIGHT - BOX_SIZE - 50)
        else:
            obstacle_type = 'ground'
            obstacle_height = random.randint(OBSTACLE_HEIGHT - 20, OBSTACLE_HEIGHT + 20)
            y = GROUND_HEIGHT - obstacle_height
        
        self.obstacles.append([SCREEN_WIDTH, y, obstacle_height, {'passed': False, 'type': obstacle_type}])

    def update_obstacles(self):
        # Move obstacles and remove off-screen ones
        for obstacle in self.obstacles:
            obstacle[0] -= self.game_speed
        
        self.obstacles = [obs for obs in self.obstacles if obs[0] > -OBSTACLE_WIDTH]
        
        # Spawn new obstacles and increase game speed
        if not self.obstacles or self.obstacles[-1][0] < SCREEN_WIDTH - random.randint(200, 900):
            self.spawn_obstacle()
            self.game_speed = min(25, self.game_speed + 0.07)

    def check_passed_obstacle(self):
        # Check if the player has passed an obstacle and update score
        for obstacle in self.obstacles:
            if self.box_x > obstacle[0] + OBSTACLE_WIDTH and not obstacle[3]['passed']:
                obstacle[3]['passed'] = True
                self.score += 1
                if self.score > self.high_score:
                    self.high_score = self.score
                    self.save_high_score()

    def check_collision(self):
        # Check if the player has collided with an obstacle
        for obstacle in self.obstacles:
            if obstacle[3]['type'] == 'ground':
                if (self.box_x + BOX_SIZE > obstacle[0] and
                    self.box_x < obstacle[0] + OBSTACLE_WIDTH and
                    self.box_y + BOX_SIZE > obstacle[1]):
                    return True
            else:  # floating obstacle
                if (self.box_x + BOX_SIZE > obstacle[0] and
                    self.box_x < obstacle[0] + OBSTACLE_WIDTH and
                    self.box_y < obstacle[1] + obstacle[2]):
                    return True
        return False

    def jump(self):
        # Make the player jump
        if not self.is_jumping:
            self.is_jumping = True
            self.box_vel_y = -13.5

    def update_box(self):
        # Update the player's position and velocity
        if self.is_jumping:
            self.box_vel_y += 0.8
            self.box_y += self.box_vel_y
            if self.box_y >= GROUND_HEIGHT - BOX_SIZE:
                self.box_y = GROUND_HEIGHT - BOX_SIZE
                self.is_jumping = False
                self.box_vel_y = 0

    def _get_state(self):
        # Get the current state of the game (used for RL)
        if not self.obstacles:
            return np.array([self.box_y, SCREEN_WIDTH, 0, self.box_vel_y])
        return np.array([
            self.box_y,
            self.obstacles[0][0] - self.box_x,
            self.obstacles[0][2],
            self.box_vel_y
        ])

    def step(self, action):
        # Perform one step of the game (used for RL)
        if action == 1:
            self.jump()
        
        self.update_box()
        self.update_obstacles()
        self.check_passed_obstacle()
        collision = self.check_collision()
        
        reward = 0.1
        if collision:
            reward = -10
            done = True
        else:
            done = False
        
        # Add reward for passing under floating obstacles
        for obstacle in self.obstacles:
            if (obstacle[3]['type'] == 'floating' and 
                self.box_x > obstacle[0] + OBSTACLE_WIDTH and 
                not obstacle[3]['passed']):
                reward += 0.5
                obstacle[3]['passed'] = True
        
        if self.render_mode:
            self.render()
        
        return self._get_state(), reward, done

    def render(self):
        # Render the game state (if render_mode is True)
        if not self.render_mode:
            return

        self.screen.fill(WHITE)
        
        # Draw the ground
        pygame.draw.rect(self.screen, BLACK, (0, GROUND_HEIGHT, SCREEN_WIDTH, SCREEN_HEIGHT - GROUND_HEIGHT))
        
        pygame.draw.rect(self.screen, GREEN, (self.box_x, self.box_y, BOX_SIZE, BOX_SIZE))
        for obstacle in self.obstacles:
            if obstacle[3]['type'] == 'ground':
                color = BLACK
            else:
                color = BLACK
            pygame.draw.rect(self.screen, color, (obstacle[0], obstacle[1], OBSTACLE_WIDTH, obstacle[2]))
        
        score_text = self.font.render(f"Score: {self.score}", True, BLACK)
        high_score_text = self.font.render(f"High Score: {self.high_score}", True, BLACK)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(high_score_text, (10, 50))
        pygame.display.flip()

    def update_display(self):
        # Update the display (if render_mode is True)
        if self.render_mode:
            pygame.display.flip()

    def close(self):
        # Close the game
        if self.render_mode:
            pygame.quit()

    def run(self):
        # Run the game loop (for human play)
        if not self.render_mode:
            print("Error: Cannot run the game without render mode. Initialize with render_mode=True.")
            return

        running = True
        while running:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.jump()
            
            _, _, done = self.step(0)
            if done:
                running = False
                print(f"Game Over! Score: {self.score}, High Score: {self.high_score}")

        self.close()

if __name__ == "__main__":
    game = JumpGame(render_mode=True)
    game.run()