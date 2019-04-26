import math
import random
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
import keras
WHITE = (255, 255, 255)
ANGLES = []
def relu(x):
    return(max(0,x))
np.vectorize(relu)

def init_neural_network():
    a = Input(shape = (3,))
    b = Dense(3, activation = 'sigmoid')(a)
    c = Dense(2, activation = 'tanh')(b)
    model = Model(inputs=a, outputs=c)
    return model


def addVectors(v1, v2):
    angle1, length1 = v1
    angle2, length2 = v2
    x = math.sin(angle1) * length1 + math.sin(angle2) * length2
    y = math.cos(angle1) * length1 + math.cos(angle2) * length2
    length = math.hypot(x, y)
    angle = 0.5 * math.pi - math.atan2(y, x)
    return (angle, length)

def generate_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

class NeuralNetwork:
    def __init__(self, shapes, activations):
        self.nb_hidden_layers = len(shapes)-2
        self.weights = []
        self.bias = []
        self.activations = []
        self.shapes = shapes
        for i in range(len(shapes)-2):
            self.weights.append(np.random.uniform(0, 1, (shapes[i], shapes[i+1])))
            self.bias.append(np.zeros(shapes[i+1]))
            self.activations = activations[i]
        self.weights.append(np.random.uniform(0, 1, (shapes[-2], shapes[-1])))

    def predict(self, input):
        c = np.copy(input)
        for i in range(self.nb_hidden_layers):
            c = self.activations[i](np.dot(self.weights[i], c) + self.bias[i])
        c = np.dot(self.weights[-1], c)
        return c

    def set_weights(self, weights, bias):
        self.weights = weights
        self.bias = bias


class Environment:
    def __init__(self,width, height, map, start):
        self.width = width
        self.height = height
        self.start = start
        self.cars = []
        self.map = map
        self.car_functions = []
        self.function_dict = {
            'move': lambda c: c.move(),
            'detect_exit' : lambda c: self.detect_exit(c),
            'neural_command' : self.command
            }
        self.number_of_active_car = 0
        self.max_time = 15
        self.nb_epochs = 10

    def addFunctions(self, function_list):
        for func in function_list:
            self.car_functions.append(self.function_dict[func])

    def addCar(self, n=1, **kargs):
        for i in range(n):
            depart = kargs.get('depart', self.start)
            length = 30
            width = 15
            c = Car(depart, length, width)
            c.speed = kargs.get('speed', random.random()*2)
            c.angular_speed = kargs.get('angular_speed', random.random()*0.1)
            c.angle = kargs.get('angle', 0)#random.uniform(0, math.pi * 2))
            c.colour = kargs.get('colour', (0, 0, 255))
            self.cars.append(c)
            self.number_of_active_car += 1

    def update(self):
        for c in self.cars :
            if not(c.out):
                for f in self.car_functions :
                    f(c)


    def detect_exit(self, car):
        corners = car.get_corners()
        for vertex in corners :
            x, y = vertex
            if not(all(self.map.data[y,x]==self.map.in_color)) :
                car.out = True
                self.number_of_active_car -= 1
                car.score = self.map.score(car)
                return None

    def command(self, car):
        datas = self.map.get_sensors(car)
        instruction = car.neural_network.predict(np.array([datas]))
        speed, angle = instruction[0]
        car.speed = 5*speed#+= (3*speed-car.speed)*0.1
        car.angle += ((angle+1)*math.pi - car.angle)*0.1
        ANGLES.append(angle)
        #print(speed, angle)

    def give_scores(self):
        for c in self.cars :
            if c.score == 0:
                c.score = self.map.score(c)


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Car:
    def __init__(self, centre, length, width):
        self.x = centre.x
        self.y = centre.y
        self.length = length
        self.width = width
        self.colour = (0, 0, 255)
        self.thickness = 1
        self.drag = 0.999
        self.speed = 0.1
        self.angular_speed = 0
        self.angle = 0
        self.acceleration = 0
        self.angular_acceleration = 0
        self.score = 0
        self.out =  False
        self.neural_network = init_neural_network()
        self.measures = []
        self.nb_iterations = 0

    def move(self):
        self.angle += self.angular_speed
        self.angle = self.angle % (2*math.pi)
        self.x += math.sin(self.angle) * self.speed
        self.y -= math.cos(self.angle) * self.speed
        self.speed *= self.drag
        self.nb_iterations += 1

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
        return np.array(self.neural_network.get_weights())


class Map:
    def __init__(self):
        from matplotlib.image import imread
        self.image_path = "circuit.png"
        self.in_color = np.array([1., 1., 1.], dtype='float32')
        self.data = imread(self.image_path)

    def score(self, car):
        goal = car.y >= 546
        time_bonus = 1000/car.nb_iterations
        return (max(0, car.y - 32)+1+((100+time_bonus) if goal else 0))**2

    def raycast(self, extremal, angle,car):
        x, y = extremal
        dx = math.sin(angle)
        dy = -math.cos(angle)
        compt = 0
        while all(self.data[int(x),int(y)] == self.in_color):
            compt+=1
            x += dx
            y += dy
        car.measures.append([int(x), int(y)])
        return compt,x,y

    def get_sensors(self, car):
        datas = []
        angles = [-math.pi/4, 0, math.pi/4]
        car.measures = []
        for da in angles :
            compt, x, y = self.raycast((car.x, car.y), car.angle + da, car)
            datas.append(compt)
        return np.array(datas)


class Generation:
    def __init__(self, n, environment):
        self.size = n
        self.environment = environment
        self.individuals = []
        self.init_population()
        self.noise_amplitude = 2
        self.decrease = 1
        self.mutating_probability = 0.05
        self.average = []
        self.best = []
        self.nb_iteration = 0


    def addCar(self, n=1, **kargs):
        for i in range(n):
            start = kargs.get('start', self.environment.start)
            length = 30
            width = 15
            c = Car(start, length, width)
            c.speed = kargs.get('speed', random.random()*2)
            c.angular_speed = kargs.get('angular_speed', random.random()*0.1)
            c.angle = kargs.get('angle', -math.pi)#random.uniform(0, math.pi * 2))
            c.colour = kargs.get('colour', (0, 0, 255))
            self.individuals.append(c)
            self.environment.number_of_active_car += 1

    def init_population(self, current = True):
        self.addCar(self.size, start = self.environment.start)
        if current :
            self.environment.cars = self.individuals


    def generate_population(self):
        population = [None]*self.size
        for i in range(self.size):
            start = self.environment.start
            length = 30
            width = 15
            c = Car(start, length, width)
            c.speed = 0#random.random()*2
            c.angular_speed = 0#random.random()*0.1
            c.angle = -math.pi#random.uniform(0, math.pi * 2)
            c.colour = (0, 0, 255)
            c.neural_network = init_neural_network()

            population[i]=c
        return population

    def next(self):
        self.individuals.sort(key=lambda c: c.score, reverse=True)
        self.best.append(self.individuals[0].score)
        self.average.append(np.average([c.score for c in self.individuals]))
        self.cross_and_mutate()
        self.environment.number_of_active_car = self.size
        self.nb_iteration += 1


    def random_pick_biased(self, scores):
        rnd = random.random() * sum(scores)
        for i, w in enumerate(scores):
            rnd -= w
            if rnd < 0:
                return i

    def noise_array(self, shape):
        amplitude = self.noise_amplitude * self.decrease
        mutated_weights = np.random.uniform(0, 1, shape) < self.mutating_probability
        mutation = np.random.uniform(-amplitude/2, amplitude/2, shape) * mutated_weights
        return mutation

    def noise(self, p):
        noise = np.array([self.noise_array(layer.shape) for layer in p])
        return noise

    def cross(self, w1, w2, alpha):
        genotyp1 = np.concatenate(np.array([layer.flatten() for layer in w1])).ravel()
        genotyp2 = np.concatenate(np.array([layer.flatten() for layer in w2])).ravel()
        n = len(genotyp1)
        slice = int(n*alpha)
        new_genotype = np.concatenate((genotyp1[:slice], genotyp2[slice:]))
        shaped_genom = [None]*w1.shape[0]
        start = 0
        for i in range(w1.shape[0]):
            layer = w1[i]
            weights = new_genotype[start:start + layer.size]
            shaped_genom[i] = weights.reshape(layer.shape)
            start += layer.size
        return np.array(shaped_genom)

    def cross_and_mutate(self):
        print("taux de mutation :", self.noise_amplitude*self.decrease)
        scores = [c.score for c in self.individuals]
        print(scores)
        weights = [c.extract_weights() for c in self.individuals]
        print(weights[0])
        #print(weights[0])
        keras.backend.clear_session()
        new_populations = self.generate_population()
        indice = 0
        #crossing
        for k in range(self.size//2):
            if k == 0:
                i,j=0,0
            else :
                i, j = self.random_pick_biased(scores), self.random_pick_biased(scores)
            print(i, j, end=" ")
            pi, pj = weights[i], weights[j]
            alpha = random.uniform(0, 1)
            alpha *= random.choice([-1, 1])
            p1, p2 = self.cross(pi, pj, alpha), self.cross(pi, pj, 1-alpha)
                #alpha*pi + (1-alpha)*pj, (1-alpha)*pi + alpha*pj

            #mutating
            if k :
                p1 = p1 + self.noise(p1)
            p2 = p2 + self.noise(p2)
            #

            new_populations[indice].neural_network.set_weights(p1)
            new_populations[indice].colour = generate_random_color()
            indice+=1
            new_populations[indice].neural_network.set_weights(p2)
            new_populations[indice].colour = generate_random_color()
            indice+=1

        self.noise_amplitude *= self.decrease
        self.individuals = new_populations.copy()
        self.environment.cars = self.individuals
        del(new_populations)
        print()

