import numpy as np
import cvxpy as cp

class LabelTree:
    def __init__(self, label, prob, children):
        self.label = label
        self.prob = prob
        self.children = children
        self.alpha = cp.Variable(integer=True)
        self.beta = cp.Variable(integer=True)

    def build_constraint_problem(self, m, tau):
        constraints = self._build_constraints_helper()
        # beta => alpha
        constraints.append(self.beta <= self.alpha)
        # \sum_{L \in leaf nodes} beta_L * prob_L >= tau
        leaves = self.get_leaves()
        constraints.append(tau <= sum([leaf.beta * leaf.prob for leaf in leaves]))
        # \sum_{N \in nodes} alpha_N <= m
        nodes = self.get_nodes()
        constraints.append(sum([node.alpha for node in nodes]) <= m)
        # objective \sum_{N \in noes} beta_N
        objective = cp.Minimize(sum([leaf.beta for leaf in leaves]))
        return cp.Problem(objective, constraints)

    def _build_constraints_helper(self):
        constraints = []
        for child in self.children:
            constraints += child._build_constraints_helper()
        # binary
        constraints += [self.alpha >= 0.0, self.beta >= 0.0, self.alpha <= 1.0, self.beta <= 1.0]
        # alpha => beta
        constraints.append(self.alpha <= self.beta)
        # beta => beta_child
        for child in self.children:
            constraints.append(self.beta <= child.beta)
        # beta_child => alpha_child \/ beta
        for child in self.children:
            constraints.append(child.beta <= self.beta + child.alpha)
        return constraints

    def get_nodes(self):
        nodes = [self]
        for child in self.children:
            nodes += child.get_nodes()
        return nodes

    def get_leaves(self):
        if len(self.children) == 0:
            return [self]
        else:
            leaves = []
            for child in self.children:
                leaves += child.get_leaves()
            return leaves



class MNISTDigit:
    def __init__(self, label, probs):
        self.label = label
        self.probs = np.array(probs)



class MNISTList:
    def __init__(self, digits):
        self.digits = digits



def construct_mnist_tree(digit_list):
    return _construct_mnist_tree_helper(digit_list, -1, 1.0, "", None)

def _construct_mnist_tree_helper(digit_list, i, prob, prefix, value):
    if i == len(digit_list) - 1:
        child_trees = []
    else:
        child_trees = [_construct_mnist_tree_helper(digit_list, i+1, prob * digit_list[i+1].probs[new_value], prefix + str(new_value), new_value) for new_value in range(10)]
    return LabelTree(prefix + ''.join(['X' for _ in range(len(digit_list) - i - 1)]), prob, child_trees)

def parse_mnist():
    f = open('../collected/mnist/mnist.csv')
    digits = []
    for line in f:
        toks = line.strip().split(',')
        label = int(toks[0])
        probs = [float(tok) for tok in toks[1:]]
        digits.append(MNISTDigit(label, probs))
    f.close()
    return digits

def construct_mnist_lists(k):
    digits = parse_mnist()
    return [digits[i:i+k] for i in range(len(digits) - k + 1)]