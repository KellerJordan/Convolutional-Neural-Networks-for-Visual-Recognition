import numpy as np

"""
This file implements various first-order update rules that are commonly used
for training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning
    rate, momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not
perform well; however the default values of the other hyperparameters should
work well for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the momentum update formula. Store the updated value in #
    # the next_w variable. You should also use and update the velocity v.     #
    ###########################################################################
    v = config['momentum'] * v - config['learning_rate'] * dw
    next_w = w + v
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    config['velocity'] = v

    return next_w, config



def rmsprop(x, dx, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(x))

    next_x = None
    ###########################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of x #
    # in the next_x variable. Don't forget to update cache value stored in    #
    # config['cache'].                                                        #
    ###########################################################################
    d, lr, eps = config['decay_rate'], config['learning_rate'], config['epsilon']
    config['cache'] = d * config['cache'] + (1 - d) * dx**2
    next_x = x - lr * dx / (np.sqrt(config['cache']) + eps)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_x, config


def adam(x, dx, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))
    config.setdefault('t', 1)

    next_x = None
    ###########################################################################
    # TODO: Implement the Adam update formula, storing the next value of x in #
    # the next_x variable. Don't forget to update the m, v, and t variables   #
    # stored in config.                                                       #
    ###########################################################################
    beta1, beta2 = config['beta1'], config['beta2']
    eps, lr = config['epsilon'], config['learning_rate']
    config['m'] = beta1 * config['m'] + (1 - beta1) * dx
    config['v'] = beta2 * config['v'] + (1 - beta2) * dx**2
    config['t'] += 1
    mt = config['m'] / (1 - beta1**config['t'])
    vt = config['v'] / (1 - beta2**config['t'])
    next_x = x - lr * mt / (np.sqrt(vt) + eps)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_x, config

## tried to use SGD to update the learning rate of SGD itself.. didn't work better than vanilla SGD
# def meta_sgd(x, dx, config=None):
#     if config is None: config = {}
# #     config.setdefault('v', np.zeros_like(x))
#     config.setdefault('lr', 1e-2)
#     config.setdefault('mu', 1e-5)
#     config.setdefault('epsilon', 1e-8)
#     config.setdefault('t', 1)
#     config.setdefault('dx_prev', np.zeros_like(x))
    
#     next_x = None
#     config['lr'] = .99*config['lr'] + config['mu']*np.sum(config['dx_prev']*dx)
#     next_x = x - config['lr']*dx
# #     config['v'] = config['v'] + config['mu']*config['dx_prev']*dx
# #     next_x = x - config['v']*dx
#     config['dx_prev'] = dx
    
#     if config['t'] % 200 == 0:
#         print(config['lr'])
# #         print('mean', np.mean(config['v']))
# #         print('max', np.max(config['v']))
# #         print('min', np.min(config['v']))
#     config['t'] += 1
#     return next_x, config
