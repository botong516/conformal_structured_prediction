import cvxpy as cp

class ImageNetTree:
    def __init__(self, label, children={}):
        self.label = label
        self.prob = cp.Parameter(nonneg=True)
        self.children = children
        self.alpha = cp.Variable(integer=True)
        self.beta = cp.Variable(integer=True)

    def build_constraint_problem(self, mParam, tauParam):
        constraints = self._build_constraints_helper()
        # beta => alpha
        constraints.append(self.beta <= self.alpha)
        # \sum_{L \in leaf nodes} beta_L * prob_L >= tau
        leaves = self.get_leaves()
        constraints.append(tauParam <= sum([leaf.beta * leaf.prob for leaf in leaves]))
        # \sum_{N \in nodes} alpha_N <= m
        nodes = self.get_nodes()
        constraints.append(sum([node.alpha for node in nodes]) <= mParam)
        # objective \sum_{N \in nodes} beta_N
        objective = cp.Minimize(sum([leaf.beta for leaf in leaves]))
        return cp.Problem(objective, constraints)

    def _build_constraints_helper(self):
        constraints = []
        for child in self.children.values():
            constraints += child._build_constraints_helper()
        # binary
        constraints += [self.alpha >= 0.0, self.beta >= 0.0, self.alpha <= 1.0, self.beta <= 1.0]
        # alpha -> beta
        constraints.append(self.alpha <= self.beta)
        # beta -> beta_child
        for child in self.children.values():
            constraints.append(self.beta <= child.beta)
        # beta_child -> alpha_child \/ beta
        for child in self.children.values():
            constraints.append(child.beta <= self.beta + child.alpha)
        return constraints

    def get_nodes(self):
        nodes = [self]
        for child in self.children.values():
            nodes += child.get_nodes()
        return nodes

    def get_leaves(self):
        if len(self.children) == 0:
            return [self]
        else:
            leaves = []
            for child in self.children.values():
                leaves += child.get_leaves()
            return leaves



def construct_imagenet_tree(hierarchy):
    root = ImageNetTree("root")
    for _, row in hierarchy.iterrows():
        current_node = root
        leaf_node = row[0]
        for node in row:
            if node == leaf_node or node is None:
                continue
            if node not in current_node.children:
                current_node.children[node] = ImageNetTree(node, {})
            current_node = current_node.children[node]
        current_node.children[leaf_node] = ImageNetTree(leaf_node, {})

    return next(iter(root.children.values()))