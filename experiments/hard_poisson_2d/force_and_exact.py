# File contains the exact solution, rhs and boundary conditions for the poisson problem
import numpy as np
import tensorflow as tf


# Setup the hard constraints
@tf.function
def apply_hard_boundary_constraints(inputs, x):
    """This method applies hard boundary constraints to the model.
    :param inputs: Input tensor
    :type inputs: tf.Tensor
    :param x: Output tensor from the model
    :type x: tf.Tensor
    :return: Output tensor with hard boundary constraints
    :rtype: tf.Tensor
    """
    x_in = inputs[:, 0:1]
    y_in = inputs[:, 1:2]

    ansatz = 16 * x_in * (1.0 - x_in) * y_in * (1.0 - y_in)
    boundary_extension = (2.0 * x_in - 1.0) * np.tanh(1.0) * tf.sin(4 * np.pi * y_in)
    tf.cast(ansatz, tf.float64)
    tf.cast(boundary_extension, tf.float64)

    return ansatz * x + boundary_extension

def rhs(x, y):
    """
    This function will return the value of the rhs at a given point.
    """
    
    f_temp = -8.0 * (2 * np.pi**2 + 1.0/(np.cosh(1 - 2*x))**2) * np.sin(4 * np.pi * y) * np.tanh(1 - 2 * x)

    return f_temp


def exact_solution(x, y):
    """
    This function will return the exact solution at a given point
    f = tanh(2x - 1)sin(4*pi*y)
    """

    val = np.tanh(2.0 * x - 1.0) * np.sin(4 * np.pi * y)

    return val


def get_bilinear_params_dict():
    """
    This function will return a dictionary of bilinear parameters
    """
    eps = 1.0

    return {"eps": eps}
