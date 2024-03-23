import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import math

# Constants
WIDTH, HEIGHT = 800, 600
AMPLITUDE = 5
WAVELENGTH = 1200
PERIOD = 5.0  # Time period of wave cycle in seconds
NUM_POINTS = 80
PARTICLE_RADIUS = 2
PARTICLE_COLOR = (1.0, 1.0, 1.0, 1.0)  # RGBA
BG_COLOR = (0, 0, 0, 1.0)  # RGBA

# Function to calculate wave height at a given x-coordinate and time
def wave_height(x, t):
    return AMPLITUDE * math.sin((2 * math.pi / WAVELENGTH) * x - (2 * math.pi / PERIOD) * t)

# Function to draw the wave
def draw_wave(t):
    glBegin(GL_LINE_STRIP)
    glColor4f(1.0, 1.0, 1.0, 1.0)  # White color for the wave
    for x in range(0, WIDTH, 10):
        y = HEIGHT // 2 + wave_height(x, t)
        glVertex3f(x - WIDTH / 2, y - HEIGHT / 2, 0)
    glEnd()

# Function to animate the particles
def animate_particles(t):
    glColor4f(*PARTICLE_COLOR)
    for i in range(NUM_POINTS):
        x = i * WIDTH // (NUM_POINTS - 1)
        y = HEIGHT // 2 + wave_height(x, t)
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(x - WIDTH / 2, y - HEIGHT / 2, 0)
        for j in range(360):
            angle = math.radians(j)
            glVertex3f(x - WIDTH / 2 + math.cos(angle) * PARTICLE_RADIUS,
                       y - HEIGHT / 2 + math.sin(angle) * PARTICLE_RADIUS,
                       -PARTICLE_RADIUS / 2 * math.sin(angle))  # Introduce slight z-offset for 3D effect
        glEnd()

def main():
    pygame.init()
    display = (WIDTH, HEIGHT)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -25)

    running = True
    t = 0  # Initial time

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                running = False

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        draw_wave(t)
        animate_particles(t)

        pygame.display.flip()
        pygame.time.wait(10)  # Limit frame rate

        t += 0.1  # Increment time

if __name__ == "__main__":
    main()
