from Manipulator2DMap.Map import GridMap2D
from Manipulator2DMap.Manipulator2D import Manipulator_2d_supervisor, PI
from Manipulator2DMap.obstacle import SphereObstacle
from Manipulator2DMap.inverse_kinematics import inverse_kinematics
from AStarDefaults.SearchTreeNode import SearchTreeNode
from AStarDefaults.AStar import make_path
from tqdm import tqdm


def create_rrt(space_map: GridMap2D, num_steps, alpha=1e-3):
    added_nodes = [SearchTreeNode(space_map.get_start())]
    
    num_nodes = 1
    iterations = 0
    
    for step in tqdm(range(num_steps)):
        iterations += 1
        
        new_point = space_map.sample_point_on_map()
        target_angles = inverse_kinematics(new_point, space_map._manipulator)
        
        nearest_node = added_nodes[0]
        best_dist = float('+inf')
        for selected_node in added_nodes:
            current_dist = space_map._manipulator.calculate_distance_between_states(selected_node.get_state(), target_angles)
            if current_dist < best_dist:
                nearest_node = selected_node
                best_dist = current_dist
           
        diffs = target_angles - nearest_node.get_state()
        
        new_angles = nearest_node.get_state() + alpha * diffs
        
        if space_map.valid(new_angles):
            new_node = SearchTreeNode(new_angles, parent=nearest_node)
            added_nodes.append(new_node)
            num_nodes += 1
            
    return added_nodes, iterations