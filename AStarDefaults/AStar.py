from AStarDefaults.SearchTree import  SearchTree
from AStarDefaults.SearchTreeNode import SearchTreeNode

def make_path(goal):
    '''
    Creates a path by tracing parent pointers from the goal node to the start node
    It also returns path's length.
    '''
    length = goal.g
    current = goal
    path = []
    while current.parent:
        path.append(current.state)
        current = current.parent
    path.append(current.state)
    return path[::-1], length


def AStar(space_map, heuristic_func=None, search_tree=SearchTree, max_steps=1_000_000):
    ast = search_tree()
    steps = 0
    nodes_created = 0
    CLOSED = None
    
    start_state = space_map.get_start()
    start = SearchTreeNode(start_state, 0, space_map.heuristic(start_state))
    ast.add_to_open(start)

    current_state = start
    while (not ast.open_is_empty()) and (space_map.dist_to_finish(current_state.state) > .1):
        current_state = ast.get_best_node_from_open()
        if (current_state is None):
            break
        for successor_state, move_cost in space_map.get_successors(current_state.state):
            neighbour = SearchTreeNode(successor_state,
                                       current_state.g + move_cost,
                                       space_map.heuristic(successor_state),
                                       parent=current_state)
            if space_map.is_goal(neighbour.state):
                return True, neighbour, steps, nodes_created, ast.OPEN, ast.CLOSED
            nodes_created += 1
            if not ast.was_expanded(neighbour):
                ast.add_to_open(neighbour) 
        ast.add_to_closed(current_state)
        steps += 1
        if steps > max_steps:
            print("Stop by max_steps")
            break
        if (steps + 1) % 50000 == 0:
            print(f"step = {steps + 1} g = {current_state.g} heuristic = {current_state.h} dist = {space_map.dist_to_finish(current_state.state)}")
    if space_map.dist_to_finish(current_state.state) > .5:    
        print("OPEN is empty")
    OPEN = ast.OPEN 
    CLOSED = ast.CLOSED
    return False, current_state, steps, nodes_created, OPEN, CLOSED