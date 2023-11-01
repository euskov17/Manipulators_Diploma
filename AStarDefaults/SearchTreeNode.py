class SearchTreeNode():
    def __init__(self, state, g = 0, h = 0, f = None, parent = None, path_from_parent=None):
        '''
        Node class represents a search node
        
        - state: some representation of state (like angles in Manipulator2D)
        - g: g-value of the node
        - h: h-value of the node
        - F: f-value of the node
        - parent: pointer to the parent-node

        '''
        self.state = state
        self.g = g
        self.h = h
        if f is None:
            self.f = self.g + h
        else:
            self.f = f        
        self.parent = parent

    def get_state(self):
        '''
        return self.state
        '''
        return self.state
    
    def __eq__(self, other):
        return (self.state == other.state).all()
    
    def __hash__(self):
        return hash(tuple(self.state))

    def __lt__(self, other): 
        return self.f < other.f