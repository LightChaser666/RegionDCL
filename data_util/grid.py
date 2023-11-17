"""
@file: grid.py
@time: 2022/09/12
"""
import math
import random

import numpy as np
import rasterio
import rasterio.features
from matplotlib import pyplot as plt
from scipy.spatial import KDTree

"""
    The Grid class is a utility class which perform poisson disk sampling
"""


class Grid(object):
    def __init__(self, boundary, radius, buildings, pois):
        self.boundary = boundary
        self.bounds = boundary.bounds
        self.radius = radius
        self.min_x = self.bounds[0]
        self.max_x = self.bounds[2]
        self.min_y = self.bounds[1]
        self.max_y = self.bounds[3]
        self.grid_size = radius * np.sqrt(2) / 2
        self.num_grid_x = int((self.max_x - self.min_x) / self.grid_size) + 1
        self.num_grid_y = int((self.max_y - self.min_y) / self.grid_size) + 1
        self.buildings = [[building['shape'].centroid.x, building['shape'].centroid.y] for building in buildings]
        self.pois = [[poi['x'], poi['y']] for poi in pois]
        self.point_tree = KDTree(self.buildings + self.pois)
        print('Grid:', self.num_grid_x, 'x', self.num_grid_y, ', grid size:', self.grid_size)
        self.grid = [[[] for _ in range(self.num_grid_y)] for _ in range(self.num_grid_x)]
        transform = rasterio.transform.from_bounds(self.min_x, self.max_y, self.max_x, self.min_y, self.num_grid_x, self.num_grid_y)
        self.valid_grid = rasterio.features.rasterize(self.boundary.geoms, out_shape=(self.num_grid_y, self.num_grid_x), transform=transform)
        # plt.imshow(self.valid_grid)
        # plt.show()
        self.random_points = []

    def poisson_disk_sampling(self):
        # pick a starting point
        start_x = np.random.rand() * (self.max_x - self.min_x) + self.min_x
        start_y = np.random.rand() * (self.max_y - self.min_y) + self.min_y
        # put in grid
        grid_x = int((start_x - self.min_x) / self.grid_size)
        grid_y = int((start_y - self.min_y) / self.grid_size)
        self.grid[grid_x][grid_y].append([start_x, start_y])
        active_list = [[start_x, start_y, grid_x, grid_y]]
        # perform poisson disk sampling
        print('Performing poisson disk sampling...')
        count = 1
        radius = self.radius
        radius2 = 2 * radius
        radiusSq = radius * radius
        while len(active_list) > 0:
            # randomly pick an active point
            index = random.randint(0, len(active_list)-1)
            point = active_list[index]
            found = False
            for i in range(30):
                new_x, new_y = self._random_point_in_ring(radius, radius2)
                new_x += point[0]
                new_y += point[1]
                # check if the new point is in the boundary
                if new_x < self.min_x or new_x > self.max_x or new_y < self.min_y or new_y > self.max_y:
                    continue
                # get the new point's grid
                new_grid_x = int((new_x - self.min_x) / self.grid_size)
                new_grid_y = int((new_y - self.min_y) / self.grid_size)
                new_grid = self.grid[new_grid_x][new_grid_y]
                if len(new_grid) > 0:
                    continue
                # for all grids nearby, check if any inside points are within the radius
                valid = True
                min_grid_x = int ((new_x - self.min_x - self.radius) / self.grid_size)
                max_grid_x = int ((new_x - self.min_x + self.radius) / self.grid_size)
                min_grid_y = int ((new_y - self.min_y - self.radius) / self.grid_size)
                max_grid_y = int ((new_y - self.min_y + self.radius) / self.grid_size)
                for i in range(min_grid_x, max_grid_x + 1):
                    if i < 0 or i >= self.num_grid_x:
                        continue
                    for j in range(min_grid_y, max_grid_y + 1):
                        if j < 0 or j >= self.num_grid_y:
                            continue
                        if i == new_grid_x and j == new_grid_y:
                            continue
                        if len(self.grid[i][j]) > 0:
                            point = self.grid[i][j][0]
                            delta_x = new_x - point[0]
                            delta_y = new_y - point[1]
                            if delta_x * delta_x + delta_y * delta_y < radiusSq:
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    active_list.append([new_x, new_y, new_grid_x, new_grid_y])
                    new_grid.append([new_x, new_y])
                    count += 1
                    found = True
                    break
            if not found:
                active_list.pop(index)
        # output the random points
        print('Total Random points:', count)
        self.pick_valid_points()
        return self.random_points

    def pick_valid_points(self):
        count = 0
        for i in range(self.num_grid_x):
            for j in range(self.num_grid_y):
                if len(self.grid[i][j]) > 0 and self.valid_grid[j][i] > 0 and len(self.point_tree.query_ball_point([self.grid[i][j][0][0], self.grid[i][j][0][1]], self.radius)) == 0:
                    self.random_points.append(self.grid[i][j][0])
                    count += 1
        print('Valid random points:', count)
        # self.plot_random_points()

    def plot_random_points(self):
        plt.figure(figsize=(150, 150))
        for building in self.buildings:
            plt.plot(building[0], building[1], 'bo')
        for poi in self.pois:
            plt.plot(poi[0], poi[1], 'bo')
        for point in self.random_points:
            plt.plot(point[0], point[1], 'ro')
        plt.savefig('random_points.png')


    @staticmethod
    def _random_point_in_ring(r1, r2):
        """
        在圆环内随机取点, r1<=r2
        :param r1: 内径
        :param r2: 外径
        :return:
        """
        a = 1 / (r2 * r2 - r1 * r1)
        random_r = math.sqrt(random.uniform(0, 1) / a + r1 * r1)
        random_theta = random.uniform(0, 2 * math.pi)
        return random_r * math.cos(random_theta), random_r * math.sin(random_theta)


