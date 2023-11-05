from typing import List 
from Manipulator2DMap.Manipulator2D import Manipulator_2d_supervisor, PI
from Manipulator2DMap.obstacle import Obstacle
from Manipulator2DMap.inverse_kinematics import inverse_kinematics
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
import random
import heapq
from matplotlib.widgets import Slider
from ipywidgets import interact



class GridMap2D:
    eps = 0.3

    def __init__(self, manipulator: Manipulator_2d_supervisor,
                       start_angles,
                       goal_position,
                       heuristic = None,
                       obstacles: List[Obstacle] = []):
        '''
        Constructor of map
        manipulator: Manipulator_2d_supervisor to work with manipulator
        start_position: start_position
        goal_position = ((x, y), angle) x,y are cords of finish
                        angle is finish angle
        '''
        self._manipulator = manipulator
        r = manipulator.radius
        self._map = np.zeros((r * 2, r))
        self._start_angles = start_angles
        self._goal_position = np.array(goal_position[0])
        print(f"print goal position {self._goal_position}")
        self._goal_angle = goal_position[1]
        self._heuristic_function = heuristic
        self._obstacles = obstacles
        # print(self._obstacles)
        if not self.valid(self._start_angles):
            self._start_angles = None
        self.cnt = 0
        self.inverse_angles = inverse_kinematics(self._goal_position, manipulator)
    
    def sample_point_on_map(self):
        r = self._map.shape[0]
        x = random.uniform(-r, r)
        y = random.uniform(0, r)
        point = np.array([x,y])
        while (not self.check_point_correct(point)):
            x = random.uniform(-r, r)
            y = random.uniform(0, r)
            point = np.array([x,y])
        return point
    
    def check_point_correct(self, point):
        for obs in self._obstacles:
            if obs.in_sphere(point):
                return False
        return True
    
    def set_start(self, angles):
        self._start_angles = angles
    
    def get_start(self):
        return self._start_angles

    def valid(self, angles):
        dots = self._manipulator.calculate_dots(angles)

        for obs in self._obstacles:
            if obs.intersect(angles, self._manipulator):
                return False
        return True
    
    def random_init_start(self):
        state = self._manipulator.generate_random_state()
        while not self.valid(state):
            state = self._manipulator.generate_random_state()
        self._start_angles = state
    
    def angle_heuristic(self, angles):
        cnt = 0
        if (cnt + 1) % 100 == 0:
            self.inverse_angles = inverse_kinematics(self._goal_position, manipulator)
        return self._manipulator.calculate_distance_between_states(self.inverse_angles, angles)
    
    
    def dist_to_finish(self, angles):
        return np.linalg.norm(self._manipulator.calculate_end(angles) - self._goal_position)
    
    def heuristic(self, angles):
        if self._heuristic_function is not None:
            dots = self._manipulator.calculate_dots(angles)
            return self._heuristic_function(dots, self.goal_position)
        euclid = self.dist_to_finish(angles)
        angle_dist = self.angle_to_finish(angles)
        weight = 1 / (10 + euclid + angle_dist)
        return euclid + weight * angle_dist
    
    def get_successors(self, angles):
        successors = []
        for succ, move_cost in self._manipulator.get_successors(angles):
            if not self._obstacles or self.valid(succ):
                successors.append((succ, move_cost))
        return successors

    def angle_to_finish(self, angles):
        last_angle = (np.sum(angles) + PI / 2) % (2 * PI)
        # if (last_angle > 0):
        #     last_angle -= PI
        # else:
        #     last_angle += PI
        return abs(last_angle - self._goal_angle)

    def is_goal(self, angles):
        return self.dist_to_finish(angles) < self.eps and self.angle_to_finish(angles) < 5 * self.eps