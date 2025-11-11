import tensorflow as tf

# InteractiveSession is removed in TF 2.x - operations run immediately
# Random tensor generation
my_tensor = tf.random.uniform((4, 4), 0, 1)
print("Random tensor:", my_tensor)

# Variable creation and initialization
my_var = tf.Variable(initial_value=my_tensor)
print("Variable:", my_var)

# Variables are immediately usable, no init needed
print("Variable value:", my_var.numpy())

# Placeholders are removed - use function inputs or create tensors directly
# For similar functionality, create a tensor with flexible shape
flexible_tensor = tf.Variable(tf.zeros((0, 5)), shape=(None, 5))
print("Flexible tensor shape:", flexible_tensor.shape)
