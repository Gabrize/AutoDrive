from matplotlib.image import imread
import numpy as np
import itertools
import matplotlib.pyplot as plt

image_path = "circuit.png"
in_color = np.array([1., 1., 1.], dtype='float32')

def calculate_score(image_path, in_color, start):
    data = imread(image_path)
    score = np.zeros((data.shape[0], data.shape[1]))
    print((data.shape[0], data.shape[1]))


    x, y = start
    move = list(itertools.product([-1, 0, 1], [-1, 0, 1]))
    frontier = [(x,y)]
    score[x, y] = 1
    while frontier :
        point = frontier.pop(0)
        x, y = point
        for dx, dy in move :
            if not(score[x+dx, y+dy]) and all(data[int(x+dx), int(y+dy)] == in_color) :
                    score[x + dx, y + dy] = score[x, y]+1
                    frontier.append((x+dx, y+dy))
    d_max = score.max()
    score = score/score.max()
    plt.imshow(score)
    return score, d_max


