import numpy as np
from Manipulator2DMap.Manipulator2D import Manipulator_2d_supervisor, PI

def inverse_kinematics(position, man: Manipulator_2d_supervisor, alpha = 1e-2, num_of_starts=10, max_step=1000):
    states = np.array([man.generate_random_state() for _ in range(num_of_starts)])
    length = man.lengths
    end_effectors = np.array([man.calculate_end(state) for state in states])
    losss = np.linalg.norm(position - end_effectors, axis=1)
    step = 0
    while losss.min() > 1e-2:
        lsins = np.sin(states) * length[np.newaxis, :]
        lcoss = np.cos(states) * length[np.newaxis, :]
        grads = 2 * ((end_effectors[:, 0] - position[0])[:, np.newaxis] * lcoss - 
                    (end_effectors[:, 1] - position[1])[:, np.newaxis] * lsins)
        states -= alpha * grads
        end_effectors = np.array([man.calculate_end(state) for state in states])
        losss = np.array([np.linalg.norm(position - end_effector) for end_effector in end_effectors])
        step += 1
        if step > max_step:
            break
    return states[np.argmin(losss)]