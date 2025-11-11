**TensorFlow is a powerful, open-source library developed by the Google Brain team for numerical computation and large-scale machine learning. Its core strength lies in using dataflow graphs, where nodes represent mathematical operations, and edges represent the multidimensional data arrays (tensors) flowing between them. This flexible architecture allows developers to deploy machine learning models seamlessly across a variety of platforms, from mobile devices (TensorFlow Lite) and embedded systems to large-scale distributed server farms with GPUs and TPUs. By providing a comprehensive ecosystem of tools, libraries, and community resources, TensorFlow drastically simplifies the entire ML workflow—from building and training complex deep neural networks to deploying them in production environments, making it a foundational tool for both research and industry applications.**

## TF Syntax Basics

```python
import tensorflow as tf

print(tf.__version__)

hello = tf.constant("Hello ")
world = tf.constant("World")

result = hello + world
print(result)


print(result.numpy().decode())
```
The `.numpy()` method converts the TensorFlow tensor to a standard NumPy array or string, and `.decode()` is used here to convert the byte string to a regular string



```python
a = tf.constant(10)

b = tf.constant(20)

print(a + b)

const = tf.constant(10)
fill_mat = tf.fill((4, 4), 10)
myzeros = tf.zeros((4, 4))
myones = tf.ones((4, 4))
myrandn = tf.random.normal((4, 4), mean=0, stddev=1.0)
myrandu = tf.random.uniform((4, 4), minval=0, maxval=1)

my_ops = [const, fill_mat, myzeros, myones, myrandn, myrandu]

for op in my_ops:
    print(op)
```
note: the `print(op)` in TensorFlow 2.x already shows the value, but in a format that includes the tensor structure. Using `.numpy()` gives a plain NumPy array representation.

However, there's a catch: the `print(op)` in TensorFlow 2.x will print the entire tensor if it's small, but for large tensors it will print a summary. Using `.numpy()` and then printing will print the entire array (which might be too big for large tensors).