import tensorflow as tf

n1 = tf.constant(1)
n2 = tf.constant(2)
n3 = n1 + n2

print(n3.numpy())  # Shows just the value: 3

print(n3)          # Shows tensor with shape/type info


print(n3)


# Get the default graph
graph_one = tf.compat.v1.get_default_graph()
print("Default graph:", graph_one)

# Create a new graph
graph_two = tf.Graph()
print("New graph:", graph_two)

# Set the new graph as default temporarily
with graph_two.as_default():
    print("Is graph_two the default?",
          graph_two is tf.compat.v1.get_default_graph())

# Back to original default graph
print("Back to default?", graph_one is tf.compat.v1.get_default_graph())
