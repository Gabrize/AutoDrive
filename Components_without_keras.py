import math
import random
import numpy as np
import race.process_map as process_map
import matplotlib.pyplot as plt

WHITE = (255, 255, 255)
GREY = (240, 240, 240)
ANGLES = []


# np.random.seed(0)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return (np.maximum(0, x))


def addVectors(v1, v2):
    angle1, length1 = v1
    angle2, length2 = v2
    x = math.sin(angle1) * length1 + math.sin(angle2) * length2
    y = math.cos(angle1) * length1 + math.cos(angle2) * length2
    length = math.hypot(x, y)
    angle = 0.5 * math.pi - math.atan2(y, x)
    return (angle, length)


def is_grey(color):
    return color[0] == color[1] == color[2]


def generate_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


class NeuralNetwork:
    def __init__(self, shapes, activations):
        self.nb_hidden_layers = len(shapes) - 2
        self.weights = []
        self.bias = []
        self.activations = activations
        self.shapes = shapes
        for i in range(len(shapes) - 2):
            self.weights.append(np.random.uniform(-1, 1, (shapes[i + 1], shapes[i])))
            self.bias.append(np.random.uniform(-1, 1, (shapes[i + 1], 1)))
        self.weights.append(np.random.uniform(-1, 1, (shapes[-1], shapes[-2])))

    def predict(self, input):
        c = np.copy([input]).T
        for i in range(self.nb_hidden_layers):
            c = self.activations[i](np.dot(self.weights[i], c) + self.bias[i])
        c = self.activations[-1](np.dot(self.weights[-1], c))
        return c.T

    def set_weights(self, weights, bias):
        self.weights = weights
        self.bias = bias


class Car:
    def __init__(self, centre, length, width):
        self.x, self.y = centre
        self.prev_x, self.prev_y = centre
        self.length = length
        self.width = width
        self.colour = (0, 0, 255)
        self.thickness = 1
        self.drag = 1
        self.speed = 0.1
        self.angular_speed = 0
        self.angle = 0
        self.acceleration = 0
        self.angular_acceleration = 0
        self.inertia_rate = 0.9
        self.score = 0
        self.out = False
        self.neural_network = NeuralNetwork([5, 3, 2], [sigmoid, sigmoid, tanh])
        self.measures = []

    def move(self):
        self.angle += self.angular_speed
        self.angle = self.angle % (2 * math.pi)
        prev_x, prev_y = self.x, self.y
        inertia_x = (self.x - self.prev_x)
        inertia_y = (self.y - self.prev_y)
        self.prev_x, self.prev_y = prev_x, prev_y
        inertia_speed = math.sqrt(inertia_x ** 2 + inertia_y ** 2)
        if inertia_speed > 0:
            inertia_x /= inertia_speed
            inertia_y /= inertia_speed
            self.x += ((1 - self.inertia_rate) * math.sin(self.angle) + inertia_x * self.inertia_rate) * self.speed
            self.y += ((1 - self.inertia_rate) * -math.cos(self.angle) + inertia_y * self.inertia_rate) * self.speed
        else:
            self.x += (math.sin(self.angle)) * self.speed
            self.y -= (math.cos(self.angle)) * self.speed
        self.speed *= self.drag

    def accelerate(self, vector):
        (self.angle, self.speed) = addVectors((self.angle, self.speed), vector)

    def radial_accelerate(self, pulse):
        self.angular_speed += self.angular_acceleration

    def get_corners(self):
        vertices = []
        vertices.append((int(self.y + self.width / 2 * math.sin(self.angle) - self.length / 2 * math.cos(self.angle)),
                         int(self.x + self.width / 2 * math.cos(self.angle) + self.length / 2 * math.sin(self.angle))))

        vertices.append((int(self.y + self.width / 2 * math.sin(self.angle) + self.length / 2 * math.cos(self.angle)),
                         int(self.x + self.width / 2 * math.cos(self.angle) - self.length / 2 * math.sin(self.angle))))

        vertices.append((int(self.y - self.width / 2 * math.sin(self.angle) + self.length / 2 * math.cos(self.angle)),
                         int(self.x - self.width / 2 * math.cos(self.angle) - self.length / 2 * math.sin(self.angle))))

        vertices.append((int(self.y - self.width / 2 * math.sin(self.angle) - self.length / 2 * math.cos(self.angle)),
                         int(self.x - self.width / 2 * math.cos(self.angle) + self.length / 2 * math.sin(self.angle))))
        return vertices

    def extract_weights(self):
        return np.copy(self.neural_network.weights), np.copy(self.neural_network.bias)

    def give_color(self, score, score_max, score_min):
        alpha = (score - score_min) / (score_max - score_min)
        colour = (255, int((1 - alpha) * 200), 0)
        self.colour = colour


class Environment:
    def __init__(self, width, height, map, start):
        self.width = width
        self.height = height
        self.start = start
        self.cars = []
        self.map = map
        self.car_functions = []
        self.function_dict = {
            'move': lambda c: c.move(),
            'detect_exit': lambda c: self.detect_exit(c),
            'neural_command': self.command
        }
        self.number_of_active_car = 0
        self.max_time = 5
        self.max_compt = 500
        self.nb_epochs = 200
        self.v_max = 6

    def addFunctions(self, function_list):
        for func in function_list:
            self.car_functions.append(self.function_dict[func])

    def addCar(self, n=1, **kargs):
        for i in range(n):
            depart = kargs.get('depart', self.start)
            length = 30
            width = 15
            c = Car(depart, length, width)
            c.speed = kargs.get('speed', random.random() * 2)
            c.angular_speed = kargs.get('angular_speed', random.random() * 0.1)
            c.angle = kargs.get('angle', 0)  # random.uniform(0, math.pi * 2))
            c.colour = kargs.get('colour', (0, 0, 255))
            self.cars.append(c)
            self.number_of_active_car += 1

    def update(self):
        for c in self.cars:
            if not (c.out):
                for f in self.car_functions:
                    f(c)

        self.max_compt = max(100,
                             max(map(lambda c: self.map.d_max * self.map.score_map[int(c.x), int(c.y)], self.cars)))

    def detect_exit(self, car):
        corners = car.get_corners()
        for vertex in corners:
            x, y = vertex
            if not (all(self.map.data[y, x] == self.map.in_color)):
                car.out = True
                self.number_of_active_car -= 1
                car.score = self.map.score(car)
                return None

    def command(self, car: Car):
        datas = self.map.get_sensors(car)
        instruction = car.neural_network.predict(datas)
        speed, angle = instruction[0]
        car.speed += (self.v_max * (speed + 1) - car.speed) * 0.1
        car.angle += angle * math.pi * 0.02
        ANGLES.append(car.angle)

    def give_scores(self):
        for c in self.cars:
            if c.score == 0:
                c.score = self.map.score(c)


class Map:
    def __init__(self, path, start):
        from matplotlib.image import imread
        self.image_path = path
        self.in_color = np.array([1., 1., 1.], dtype='float32')
        self.data = imread(self.image_path)
        self.score_map, self.d_max = process_map.calculate_score(self.image_path, self.in_color, start)

    def score(self, car):
        # goal = car.y >= 546
        # return (max(0, car.y - 32) + 1 + (100 if goal else 0)) ** 2
        return self.score_map[int(car.x), int(car.y)]

    def raycast(self, extremal, angle, car):
        x, y = extremal
        dx = 2 * math.sin(angle)
        dy = -2 * math.cos(angle)
        compt = 0
        while all(self.data[int(x), int(y)] == self.in_color) or is_grey(self.data[int(x), int(y)]):
            compt += 1
            x += dx
            y += dy
        car.measures.append([int(x), int(y)])
        return compt, x, y

    def get_sensors(self, car):
        datas = []
        angles = [-math.pi / 2, -math.pi / 4, 0, math.pi / 4, math.pi / 2]
        car.measures = []
        for da in angles:
            compt, x, y = self.raycast((car.x, car.y), car.angle + da, car)
            datas.append(compt)
        return np.array(datas)


class Generation:
    def __init__(self, n, environment):
        self.size = n
        self.environment = environment
        self.individuals = []
        self.init_population()
        self.initial_angle = 0
        self.noise_amplitude = 2
        self.decrease = 1
        self.mutating_probability = 0.5
        self.average = []
        self.best = []
        self.allow_crossing = True
        self.selected_parents = []
        self.selected_parents_index = []
        self.elite_size = 0
        self.elitism = 8
        self.crossing_rate = 0.5

    def addCar(self, n=1, **kargs):
        for i in range(n):
            start = kargs.get('start', self.environment.start)
            length = 30
            width = 15
            c = Car(start, length, width)
            c.speed = kargs.get('speed', random.random() * 2)
            c.angular_speed = kargs.get('angular_speed', random.random() * 0.1)
            c.angle = 0  # self.initial_angle  # random.uniform(0, math.pi * 2))
            c.colour = kargs.get('colour', (0, 0, 255))
            self.individuals.append(c)
            self.environment.number_of_active_car += 1

    def init_population(self, current=True):
        self.addCar(self.size, start=self.environment.start)
        if current:
            self.environment.cars = self.individuals

    def generate_population(self):
        population = []
        for i in range(self.size):
            start = self.environment.start
            length = 30
            width = 15
            c = Car(start, length, width)
            c.speed = 0  # random.random()*2
            c.angular_speed = 0  # random.random()*0.1
            c.angle = self.initial_angle  # random.uniform(0, math.pi * 2)
            c.colour = (0, 0, 255)
            c.neural_network = NeuralNetwork([5, 3, 2], [sigmoid, tanh])

            population.append(c)
        return population

    def next(self):
        self.individuals.sort(key=lambda c: c.score, reverse=True)
        self.best.append(self.individuals[0].score)
        self.average.append(np.average([c.score for c in self.individuals]))
        self.cross_and_mutate()
        self.environment.number_of_active_car = self.size
        return self.best[-1]

    def random_pick_biased(self, scores):
        def f(x):
            return x ** self.elitism

        np.vectorize(f)
        probas = f(np.array(scores))
        probas = probas / sum(probas)
        # print(probas)
        return np.random.choice(np.arange(len(scores)), p=probas)
        # rnd = random.random() * sum(f(np.array(scores))) + sum(np.exp(-(i + 1)) for i in range(len(scores)))
        # for i, w in enumerate(scores):
        #     rnd -= f(w) + np.exp(-(i + 1))
        #     if rnd < 0:
        #         return i

    def noise_array(self, shape):
        amplitude = self.noise_amplitude * self.decrease
        mutated_weights = np.random.uniform(0, 1, shape) < self.mutating_probability
        mutation = np.random.uniform(-amplitude / 2, amplitude / 2, shape) * mutated_weights
        return mutation

    def noise(self, p):
        noise = np.array([self.noise_array(layer.shape) for layer in p])
        return noise

    def cross_slices(self, w1, w2, alpha):
        genotyp1 = np.concatenate(np.array([layer.flatten() for layer in w1])).ravel()
        genotyp2 = np.concatenate(np.array([layer.flatten() for layer in w2])).ravel()
        n = len(genotyp1)
        slice = int(n * alpha)
        new_genotype = np.concatenate((genotyp1[:slice], genotyp2[slice:]))
        shaped_genom = [None] * w1.shape[0]
        start = 0
        for i in range(w1.shape[0]):
            layer = w1[i]
            weights = new_genotype[start:start + layer.size]
            shaped_genom[i] = weights.reshape(layer.shape)
            start += layer.size
        return np.array(shaped_genom)

    def cross(self, w1, w2):
        n = len(w1)
        genom = []
        for i in range(n):
            selection_array = np.random.randint(0, 1, w1[i].shape)
            genom.append(selection_array * w1[i] + (1 - selection_array) * w2[i])
        return np.array(genom)

    def cross_and_mutate(self):
        self.selected_parents = []
        self.selected_parents_index = []
        # print("taux de mutation :", self.noise_amplitude * self.decrease)
        scores = [c.score for c in self.individuals]
        plt.plot(scores)
        res = [np.exp(-(i + 1) / 2) + score for i, score in enumerate(scores)]
        plt.plot([r / sum(res) for r in res])
        # print(scores)
        score_min = scores[-1]
        score_max = scores[0]
        weights = [c.extract_weights()[0] for c in self.individuals]
        # print(weights[0])
        bias = [c.extract_weights()[1] for c in self.individuals]
        # print(weights[0])
        new_populations = self.generate_population()
        indice = 0
        # crossing
        n_parents = ((self.size - self.elite_size) * self.crossing_rate)
        for k in range(0, self.size - 1, 2):
            if k < self.elite_size:
                i, j = k, k + 1
                # print('e')
            else:
                i, j = self.random_pick_biased(scores), self.random_pick_biased(scores)
            # print(i, j, end=" ")
            if i not in self.selected_parents_index:
                self.selected_parents_index.append(i)
                # print(i, len(self.individuals), self.size)
                self.selected_parents.append((self.individuals[i].get_corners(), self.individuals[i].colour))
            if j not in self.selected_parents_index:
                self.selected_parents_index.append(j)
                self.selected_parents.append((self.individuals[j].get_corners(), self.individuals[j].colour))
            avg_score = (scores[i] + scores[j]) / 2
            wi, wj = weights[i], weights[j]
            bi, bj = bias[i], bias[j]

            if self.elite_size < k < self.elite_size + n_parents:
                # print('p')
                w1, w2 = self.cross(wi, wj), self.cross(wi, wj)
                b1, b2 = self.cross(bi, bj), self.cross(bi, bj)
            else:
                # print('r')
                w1, w2 = wi.copy(), wj.copy()
                b1, b2 = bi.copy(), bj.copy()
            # alpha*wi + (1-alpha)*wj, (1-alpha)*wi + alpha*wj

            # mutating
            if k > self.elite_size:
                # print('m', k, self.elite_size)
                w1 = w1 + self.noise(w1)
                b1 = b1 + self.noise(b1)
                b2 = b2 + self.noise(b2)
                w2 = w2 + self.noise(w2)

            new_populations[indice].neural_network.set_weights(w1, b1)
            # new_populations[indice].colour = generate_random_color()
            new_populations[indice].give_color(avg_score, score_min, score_max)
            indice += 1
            new_populations[indice].neural_network.set_weights(w2, b2)
            new_populations[indice].give_color(avg_score, score_min, score_max)
            # new_populations[indice].colour = generate_random_color()
            indice += 1
        new_populations.reverse()
        # print(new_populations)

        self.noise_amplitude *= self.decrease
        self.individuals = new_populations.copy()
        self.environment.cars = self.individuals
        # print()
