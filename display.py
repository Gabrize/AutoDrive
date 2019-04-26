import pygame
import race.Components_without_keras as Components
import math
import random
import time
import sys
import matplotlib.pyplot as plt
import numpy as np

def launch():
    display_sensor = True
    display_crashed_cars = True
    display_network = True
    display_selection = True
    display_score_map = True
    pygame.display.set_caption('Evolutive Race')
    (width, height) = (640, 400)
    (network_width, network_height) = (150, 150)

    screen = pygame.display.set_mode((width, height))
    path = "race/circuit_rond_trou.png"
    start = (74, 300)
    race_map = Components.Map(path, start)
    if display_score_map :
        plt.imshow(race_map.score_map)
        plt.ion()
        plt.show()
    background = pygame.image.load(race_map.image_path).convert()
    running = True
    screen.blit(background, (0, 0))

    env = Components.Environment(width, height, race_map, start)
    gen = Components.Generation(50, env)
    gen.initial_angle = 0
    env.colour = (0, 0, 0)
    env.addFunctions(['move', 'detect_exit', 'neural_command'])


    def draw_network_figure(screen, car: Components.Car, l: int, h: int, mh: int, ml: int):
        max_thickness = 5
        negative_color = (255, 0, 0)
        positive_color = (0, 255, 0)
        top = height - network_height
        left = 0  # width - network_width
        shapes = car.neural_network.shapes
        weights = car.neural_network.weights
        n = len(shapes)
        t_max = max(shapes)
        dh = (h - 2 * mh) // (t_max - 1)
        dl = (l - 2 * ml) // (n - 1)
        radius = min(dh, dl) // 3
        w_max = max(map(lambda x: x.max(), weights))

        # pygame.draw.rect(screen, Components.WHITE, (left, top, network_width, network_height))
        for i in range(n - 1):
            column = left + ml + i * dl
            weights = car.neural_network.weights
            for j in range(weights[i].shape[0]):
                for k in range(weights[i].shape[1]):
                    line_2 = top + h // 2 - int(dh * (shapes[i + 1] - 1) / 2) + j * dh
                    line_1 = top + h // 2 - int(dh * (shapes[i] - 1) / 2) + k * dh
                    w = weights[i][j, k]
                    color = negative_color if w < 0 else positive_color
                    pygame.draw.lines(screen, color, False, [(column, line_1), (column + dl, line_2)],
                                      int(abs(w) / w_max * max_thickness))
            for k in range(weights[i].shape[1]):
                line = top + h // 2 - int(dh * (shapes[i] - 1) / 2) + k * dh
                pygame.draw.circle(screen, Components.GREY, (column, line), radius)
            if i == n - 2:
                for k in range(shapes[-1]):
                    line = top + h // 2 - int(dh * (shapes[-1] - 1) / 2) + k * dh
                    pygame.draw.circle(screen, Components.GREY, (column + dl, line), radius)


    for i in range(env.nb_epochs):
        print(i + 1, "/", env.nb_epochs)
        print(sys.getsizeof(env.cars), sys.getsizeof(gen))
        start_time = time.time()
        compt = 0
        while running and env.number_of_active_car > 0 and compt < env.max_compt:
            env.update()
            compt += 1
            screen.blit(background, (0, 0))
            for c in env.cars:
                if display_sensor:
                    if not c.out:
                        for l in c.measures:
                            pygame.draw.line(screen, (255, 200, 200), (c.y, c.x), (l[1], l[0]), 1)

                if display_crashed_cars or not c.out:
                    vertices = c.get_corners()
                    pygame.draw.polygon(screen, c.colour, vertices, 0)

            if display_network:
                draw_network_figure(screen, max(env.cars, key=lambda c: c.y), network_width, network_height, 10, 20)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            pygame.display.flip()
        env.give_scores()
        gen.next()

        if display_selection:
            screen.blit(background, (0, 0))
            if display_network:
                draw_network_figure(screen, max(env.cars, key=lambda c: c.y), network_width, network_height, 10, 20)
            for car in gen.selected_parents:
                pygame.draw.polygon(screen, car[1], car[0], 0)
            pygame.display.flip()
            time.sleep(1)

        print(compt)
        if not running:
            break
    # print(Components.A)
    plt.plot(gen.average)
    plt.show()
    plt.plot(gen.best)
