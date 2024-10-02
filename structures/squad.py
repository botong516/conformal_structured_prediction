import cvxpy as cp

class SQuADDAG:
    def __init__(self, label, children=[], parents=[]):
        self.label = label
        self.prob = cp.Parameter(nonneg=True)
        self.children = children
        self.alpha = cp.Variable(integer=True)
        self.beta = cp.Variable(integer=True)
        self.parents = parents
    
    def build_constraint_problem(self, mParam, tauParam):
        visited_nodes = set()
        constraints = self._build_constraints_helper(visited_nodes)
        # beta => alpha
        constraints.append(self.beta <= self.alpha)
        # \sum_{L \in leaf nodes} beta_L * prob_L >= tau
        leaves = self.get_leaves()
        constraints.append(tauParam <= sum([leaf.beta * leaf.prob for leaf in leaves]))
        # \sum_{N \in nodes} alpha_N <= m
        nodes = self.get_nodes()
        constraints.append(sum([node.alpha for node in nodes]) <= mParam)
        # objective \sum_{N \in noes} beta_N
        objective = cp.Minimize(sum([leaf.beta for leaf in leaves]))
        return cp.Problem(objective, constraints)

    def _build_constraints_helper(self, visited_nodes):
        if self in visited_nodes:
            return []
        visited_nodes.add(self)

        constraints = []
        for child in self.children:
            constraints += child._build_constraints_helper(visited_nodes)
        # binary
        constraints += [self.alpha >= 0.0, self.beta >= 0.0, self.alpha <= 1.0, self.beta <= 1.0]
        # alpha => beta
        constraints.append(self.alpha <= self.beta)
        # beta => beta_child
        for child in self.children:
            constraints.append(self.beta <= child.beta)
        # beta_child => alpha_child \/ beta
        for child in self.children:
            parent_betas = 0
            for parent in child.parents:
                parent_betas += parent.beta
            constraints.append(child.beta <= parent_betas + child.alpha)
        return constraints

    def get_nodes(self):
        nodes = []
        visited = set()
        self._get_nodes_helper(visited, nodes)
        return nodes

    def _get_nodes_helper(self, visited, nodes):
        if self in visited:
            return
        visited.add(self)
        nodes.append(self)
        for child in self.children:
            child._get_nodes_helper(visited, nodes)
    
    def get_leaves(self):
        leaves = []
        visited = set()
        self._get_leaves_helper(visited, leaves)
        return leaves

    def _get_leaves_helper(self, visited, leaves):
        if self in visited:
            return
        visited.add(self)
        if len(self.children) == 0:
            leaves.append(self)
        else:
            for child in self.children:
                child._get_leaves_helper(visited, leaves)



def construct_year_dag(start_year, end_year):
    print(f"Constructing DAG for years {start_year} to {end_year} ...")
    leaves = [SQuADDAG((year, year), [], []) for year in range(start_year, end_year + 1)]
    current_layer = leaves

    while len(current_layer) > 1:
        next_layer = []
        for i in range(0, len(current_layer) - 1):
            interval_start = current_layer[i].label[0]
            interval_end = current_layer[i + 1].label[1]
            parent_node = SQuADDAG((interval_start, interval_end), [], [])

            # Add children to parent
            parent_node.children.append(current_layer[i])
            parent_node.children.append(current_layer[i + 1])

            # Add parent to children
            current_layer[i].parents.append(parent_node)
            current_layer[i + 1].parents.append(parent_node)

            next_layer.append(parent_node)

        current_layer = next_layer

    root = current_layer[0]
    print(f"Done.")
    return root