import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
from functools import lru_cache
import heapq
from matplotlib.widgets import Slider
from ipywidgets import interact

PI = np.pi
TWO_PI = 2 * PI

class Manipulator_2d_supervisor():
    def __init__(self, 
                 num_joints: int, 
                 lengths: np.ndarray, 
                 angle_discretization: int, 
                 angles_constraints: np.ndarray,
                 ground_level: int = 0.3,
                 distanse_between_edges: int = 0.2):
        '''
        initializing Manipulator supervisor
        
        num_joint: number of manipulator joints
        length: np.ndarray of manipulator's arms length
        angle_discretization: int value of how many bins of angles we will have. Number of bins is  TWO_PI / angle_discretization
        angles_constraints: np.ndarray with shape num_joints with left and right_bounds. \
                            Absolute value of differrence between to near angles couln't be more than PI(1 - angles_constraints)
        ground_level: float minimum level of joints positions.
        distanse_between_edges: float minimum distance between edges
        '''
        self.num_joints = num_joints
        self.lengths = np.array(lengths)
        self.radius = sum(self.lengths)
        self.angle_discretization = angle_discretization
        self.deltas = TWO_PI / angle_discretization
        self.angles_constraints = angles_constraints
        self.possible_moves = self.deltas * np.concatenate([np.eye(num_joints), -np.eye(num_joints)], 0)
        self.move_cost = self.deltas * (np.argmax(np.abs(self.possible_moves), -1) + 1)
        self.ground_level = ground_level
        self.distanse_between_edges = distanse_between_edges
        
    def calculate_dots(self, angles):
        '''
        By angles of joints calculates coordinates of each joint.
        
        angles: angles of joints (local angles)
        return: dots - np.ndarray(num_joints, 2) with coordinates x,y of each joint
        '''
        global_angles = np.cumsum(angles)
        dots = np.zeros((self.num_joints + 1, 2))
        dots[1:, 0] = np.cumsum(np.cos(global_angles) * self.lengths)
        dots[1:, 1] = np.cumsum(np.sin(global_angles) * self.lengths)
        return dots
    
    def calculate_end(self, angles):
        '''
        By angles of joints calculates coordinates of last point of manipulator.
        
        angles: angles of joints
        return: (x, y) coordinates x,y of last point
        '''
        global_angles = np.cumsum(angles)
        x = np.sum(np.cos(global_angles) * self.lengths)
        y = np.sum(np.sin(global_angles) * self.lengths)
        return (x, y)
    
    def calculate_distance_between_states(self, angles_1, angles_2):
        '''
        Calculates distanst between states like a difference between angles
        distanse = sum(
                        min(
                            abs(angles_1[i] - angles_1[i]),
                            TWO_PI - abs(angles_1[i] - angles_1[i])
                            )
                        )
        '''
        diffs = (np.abs(angles_1 - angles_2)) % TWO_PI
        angle_diffs = np.minimum(diffs, (TWO_PI - diffs))
        return angle_diffs.sum()
    
    def generate_random_state(self):
        random_delta_numbers = np.random.randint(0, self.angle_discretization, self.num_joints)
        random_state = random_delta_numbers * self.deltas - PI
        random_state[0] += PI / 2
        # print(random_state)
        while (not self.check_angles_correctness(random_state)) or (not self.position_correctness(random_state)):
            random_delta_numbers = np.random.randint(0, self.angle_discretization, self.num_joints)
            random_state = random_delta_numbers * self.deltas - PI
            random_state[0] += PI / 2
            # print(random_state)
        return random_state
    
    
    def dot_on_segment(self, x1, y1, x2, y2, x3, y3): 
        '''
        checks if (x2, y2) lies in segment ((x1, y1), (x3, y3))
        '''
        if ((x2 <= max(x1, x3)) and (x2 >= min(x1, x3)) and 
            (y2 <= max(y1, y3)) and (y2 >= min(y1, y3))): 
            return True
        return False

    def orientation(self, x1, y1, x2, y2, x3, y3): 
        '''
        returns orientation of 3 given dots
        '''
        val = (y2 - y1) * (x3 - x2) - (x2 - x1) * (y3 - y2)
        if val > 0: 
            return 'clockwize'
        elif val < 0: 
            return 'counterclockwise'
        else: 
            return 'collinear'

    def do_intersect(self, segment1, segment2): 
        (x1, y1), (x2, y2) = segment1
        (x3, y3), (x4, y4) = segment2
        o1 = self.orientation(x1, y1, x2, y2, x3, y3) 
        o2 = self.orientation(x1, y1, x2, y2, x4, y4) 
        o3 = self.orientation(x3, y3, x4, y4, x1, y1) 
        o4 = self.orientation(x3, y3, x4, y4, x2, y2) 
        if (o1 != o2) and (o3 != o4): 
            return True
        if (o1 == 0) and self.dot_on_segment(x1, y1, x3, y3, x2, y2): 
            return True
        if (o2 == 0) and self.dot_on_segment(x1, y1, x4, y4, x2, y2): 
            return True
        if (o3 == 0) and self.dot_on_segment(x3, y3, x1, y1, x4, y4): 
            return True
        if (o4 == 0) and self.dot_on_segment(x3, y3, x2, y2, x4, y4): 
            return True
        return False
    
#     def orientation(self, A, B, C):
#         '''
#         A, B, C: points of triangle
#         return orientation of triangle ABC as +1, -1
#                 if A is to close to BC segment return 0
#         '''
#         left = (C[1] - A[1]) * (B[0] - A[0])
#         right = (B[1] - A[1]) * (C[0] - A[0])
#         if right == 0 or abs(left - right) / right < self.distanse_between_edges:
#             return 0
#         if left > right: return 1
#         return -1
    
#     def do_intersect(self, segment1, segment2): 
#         '''
#         Check intersection of two segments
        
#         segment1: coordinates of start and finish poits of segment1
#         segment2: coordinates of start and finish poits of segment2
        
#         return True if has intersect and False otherwise
#         '''
#         A, B = segment1
#         C, D = segment2
#         o1 = self.orientation(A, C, D)
#         o2 = self.orientation(B, C, D)
#         o3 = self.orientation(A, B, C)
#         o4 = self.orientation(A, B, D)
#         return o1 * o2 * o3 * o4 == 0 or (o1 != o2 and o3 != o4)  

    def position_intersection(self, dots: np.ndarray) -> bool:
        '''
        Returns True if NO self-intersection, else False
        '''
        for segment_1_index in range(self.num_joints - 1):
            for segment_2_index in range(segment_1_index + 2, self.num_joints):
                segment_1_dots = dots[segment_1_index: segment_1_index + 2]  
                segment_2_dots = dots[segment_2_index: segment_2_index + 2]
                if self.do_intersect(segment_1_dots, segment_2_dots):
                    return False
        return True

    def position_correctness(self, angles):
        '''
        Checking if possition is correct 
        input: angles of joints
        return: True if position is correct and False otherwise
        '''
        dots = self.calculate_dots(angles)
        return self.position_intersection(dots) and \
                (dots[1:, 1] > self.ground_level).all()

    def check_angles_correctness(self, angles):
        return (np.abs(angles[1:]) < (PI - self.angles_constraints[1:])).all()
    

    def get_successors(self, angles):
        '''
        Returns massive, which elements are [<elements of moves massive>, <x, y of end>]
        '''
        possible_neighbours = self.possible_moves + angles
        successors = []
        for successor_state, move_cost in zip(possible_neighbours, self.move_cost):
            if not self.check_angles_correctness(successor_state):
                continue
            if self.position_correctness(successor_state):
                successors.append((successor_state, move_cost))
        return successors

    def are_states_directly_connected(self, begin_state, end_state):
        connect_path = []
        current_state = begin_state
        while not self.calculate_distance_between_states(current_state, end_state) < 0.01:
            print(f' current = {current_state}')
            best_neighbour = None
            for successor_state, move_cost, x, y in self.get_successors(current_state):
                diffs = (np.abs(successor_state - end_state)) % TWO_PI
                angle_diffs = np.minimum(diffs, (TWO_PI - diffs))

                print(f'successor = {successor_state}, dist = {self.calculate_distance_between_states(successor_state, end_state)}, diff = {angle_diffs}')
                if self.calculate_distance_between_states(successor_state, end_state) < self.calculate_distance_between_states(current_state, end_state):
                    neighbour = Manipulator_2d_node(successor_state, parent=current_state) 
                    best_neighbour = neighbour
                    connect_path.append(best_neighbour)
                    current_state = successor_state     
            diffs = (np.abs(successor_state - end_state)) % TWO_PI
            angle_diffs = np.minimum(diffs, (2 * TWO_PI - diffs))
            print(f'chosen = {best_neighbour.get_angles()}, dist = {self.calculate_distance_between_states(best_neighbour.get_angles(), end_state)}, diff = {angle_diffs}')
            if best_neighbour is None:
                return False, None
        return True, connect_path
    
    def visualize_state(self, angles):
        dots = self.calculate_dots(angles)
        plt.figure(figsize=(10,6))
        plt.axis([-self.radius, self.radius, 0, self.radius])
        plt.plot(dots[:, 0], dots[:, 1])
        plt.scatter(dots[:, 0], dots[:, 1])