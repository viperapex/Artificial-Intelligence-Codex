from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import tensorflow
import numpy as np


# Operation
class Operation():

    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes
        self.output_nodes = []

        for node in input_nodes:
            node.output_nodes.append(self)

        _default_graph.operations.append(self)

    def compute(self):
        pass


class add(Operation):

    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var + y_var


class multiply(Operation):

    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var * y_var


class matmul(Operation):

    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var.dot(y_var)


class Placeholder():

    def __init__(self):

        self.output_nodes = []
        _default_graph.placeholders.append(self)


class Variable():

    def __init__(self, initial_value=None):

        self.value = initial_value
        self.output_nodes = []

        _default_graph.variables.append(self)


class Graph():

    def __init__(self):

        self.operations = []
        self.placeholders = []
        self.variables = []

    def set_as_default(self):
        # allows us to access the graph in other classes like placeholder and variables
        global _default_graph
        _default_graph = self


"""
z = Ax + b

A = 10
b = 1

z = 10x + 1
"""


g = Graph()

g.set_as_default()
A = Variable(10)
b = Variable(1)

x = Placeholder()

y = multiply(A, x)

z = add(y, b)


def traverse_postorder(operation):
    """
    PostOrder Traversal of Nodes. Basically makes sure 
    computations are done in the correct order (Ax first, ten Ax + b).
    """
    nodes_postorder = []

    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)

    recurse(operation)
    return nodes_postorder


class Session():

    def run(self, operation, feed_dict={}):

        nodes_postorder = traverse_postorder(operation)

        for node in nodes_postorder:

            if type(node) == Placeholder:
                node.output = feed_dict[node]

            elif type(node) == Variable:

                node.output = node.value

            else:
                # Operation
                node.inputs = [
                    input_node.output for input_node in node.input_nodes]

                node.output = node.compute(*node.inputs)

            if type(node.output) == list:
                node.output = np.array(node.output)

        return operation.output


sess = Session()

result = sess.run(operation=z, feed_dict={x: 10})
print(result)


g = Graph()

g.set_as_default()

A = Variable([[10, 20], [30, 40]])
b = Variable([1, 1])

x = Placeholder()

y = matmul(A, x)

z = add(y, b)

sess = Session()

result = sess.run(operation=z, feed_dict={x: 10})
print(result)


# Classification

# Activation Function


def sigmoid(z):
    return 1/(1 + np.exp(-z))


sample_z = np.linspace(-10, 10, 100)
sample_a = sigmoid(sample_z)

# sigmid function: all values are converted to values between 0  and 1
plt.plot(sample_z, sample_a)
plt.show()


class Sigmoid(Operation):
    def __init__(self, z):

        super().__init__([z])

    def compute(self, z_val):
        return 1 / (1 + np.exp(-z_val))


data = make_blobs(n_samples=50, n_features=2, centers=2, random_state=75)

print(data)


features = data[0]
labels = data[1]

plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='coolwarm')
plt.show()


x = np.linspace(0, 11, 10)
y = -x + 5
plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='coolwarm')
plt.plot(x, y)
plt.show()


# (1,1) * f - 5 = 0
np.array([1, 1]).dot(np.array([[8], [10]])) - 5

np.array([1, 1]).dot(np.array([[2], [-10]])) - 5

g = Graph()

g.set_as_default()

x = Placeholder()

w = Variable([1, 1])

b = Variable(-5)

z = add(matmul(w, x), b)

a = Sigmoid(z)

sess = Session()

r = sess.run(operation=a, feed_dict={x: [8, 10]})
print(r)
r1 = r = sess.run(operation=a, feed_dict={x: [2, -10]})
print(r1)
