import pygame
import threading
import math
import time
import random

class HUDState:
    """Shared state between the audio pipeline and the HUD render loop."""
    def __init__(self):
        self.mode = 'idle'       # 'idle' | 'listening' | 'speaking'
        self.amplitude = 0.0     # 0.0 to 1.0, updated from audio chunks
        self.lock = threading.Lock()

    def set_mode(self, mode):
        with self.lock:
            self.mode = mode
            if mode == 'idle':
                self.amplitude = 0.0

    def set_amplitude(self, value):
        with self.lock:
            self.amplitude = max(0.0, min(1.0, value))

    def snapshot(self):
        with self.lock:
            return self.mode, self.amplitude


hud_state = HUDState()


def _draw_frame(screen, cx, cy, r_inner, n_lines, t, mode, amplitude, line_angles, phases):
    screen.fill((10, 10, 12))

    for i, angle in enumerate(line_angles):
        if mode == 'idle':
            amp = 6 + math.sin(t * 0.8 + phases[i]) * 3
        elif mode == 'listening':
            amp = 10 + math.sin(t * 2 + angle * 3) * 6 + amplitude * 25
        else:  # speaking
            amp = 14 + amplitude * 60 + random.uniform(0, 6)

        r_outer = r_inner + float(amp)
        x1 = cx + math.cos(angle) * r_inner
        y1 = cy + math.sin(angle) * r_inner
        x2 = cx + math.cos(angle) * r_outer
        y2 = cy + math.sin(angle) * r_outer
        pygame.draw.line(screen, (90, 160, 255), (float(x1), float(y1)), (float(x2), float(y2)), 2)

    pygame.draw.circle(screen, (25, 25, 28), (cx, cy), r_inner - 6)
    pygame.draw.circle(screen, (60, 60, 66), (cx, cy), r_inner - 6, 1)

    font = pygame.font.SysFont('Arial', 28, bold=True)
    label = font.render('ANTON', True, (220, 220, 225))
    screen.blit(label, label.get_rect(center=(cx, cy)))


def run_hud():
    pygame.init()
    size = 800
    screen = pygame.display.set_mode((size, size), pygame.NOFRAME)
    pygame.display.set_caption('Anton')
    clock = pygame.time.Clock()

    cx, cy = size // 2, size // 2
    r_inner = 110
    n_lines = 48
    line_angles = [(i / n_lines) * 2 * math.pi for i in range(n_lines)]
    phases = [random.uniform(0, 2 * math.pi) for _ in range(n_lines)]

    t = 0.0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        mode, amplitude = hud_state.snapshot()
        _draw_frame(screen, cx, cy, r_inner, n_lines, t, mode, amplitude, line_angles, phases)
        pygame.display.flip()

        t += 0.05
        clock.tick(60)

    pygame.quit()


def start_hud_thread():
    thread = threading.Thread(target=run_hud, daemon=True)
    thread.start()
    return thread