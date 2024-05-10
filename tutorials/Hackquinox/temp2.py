import pygame
import math

# Constants
WIDTH, HEIGHT = 800, 600
AMPLITUDE = 50
WAVELENGTH = 200
PERIOD = 1.0  # Time period of wave cycle in seconds
NUM_POINTS = 20
PARTICLE_RADIUS = 2
PARTICLE_COLOR = (255, 255, 255)
BG_COLOR = (0, 0, 0)

# Function to calculate wave height at a given x-coordinate and time
def wave_height(x, t):
    return AMPLITUDE * math.sin((2 * math.pi / WAVELENGTH) * x - (2 * math.pi / PERIOD) * t)

# Function to draw the wave
def draw_wave(screen, t):
    for x in range(0, WIDTH, 10):
        y = HEIGHT // 2 + wave_height(x, t)
        pygame.draw.circle(screen, PARTICLE_COLOR, (x, y), PARTICLE_RADIUS)

# Function to animate the particles
def animate_particles(screen, t):
    for i in range(NUM_POINTS):
        x = i * WIDTH // (NUM_POINTS - 1)
        y = HEIGHT // 2 + wave_height(x, t)
        pygame.draw.circle(screen, PARTICLE_COLOR, (x, y), PARTICLE_RADIUS)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Gerstner Wave Animation")
    clock = pygame.time.Clock()

    running = True
    t = 0  # Initial time

    while running:
        screen.fill(BG_COLOR)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Draw the wave
        draw_wave(screen, t)

        # Animate the particles
        animate_particles(screen, t)

        pygame.display.flip()
        clock.tick(60)  # Limit frame rate to 60 FPS
        t += 0.1  # Increment time

    pygame.quit()

if __name__ == "__main__":
    main()
