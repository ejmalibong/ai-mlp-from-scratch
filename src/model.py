import numpy as np

class MLP:
  def __init__(self,input_dim,hidden_dims,output_dim,lr,batch_size,activation,activation_grad,loss_fn,loss_grad):
    self.dims = [input_dim] + hidden_dims + [output_dim]
    self.lr = lr
    self.batch_size = batch_size
    self.activation = activation
    self.activation_grad = activation_grad
    self.loss_fn = loss_fn
    self.loss_grad = loss_grad

    # params
    self.W = []
    self.b = []

    for i in range(len(self.dims)-1):
      w = np.random.randn(self.dims[i], self.dims[i+1]) * np.sqrt(2.0 / self.dims[i])
      b = np.zeros((1, self.dims[i+1]))
      self.W.append(w)
      self.b.append(b)

  def forward(self, X):
    self.z = []
    self.a = []

    a = X
    self.a.append(a)   # a0 = input

    for i in range(len(self.W)):
      z = a @ self.W[i] + self.b[i]
      self.z.append(z)

      # activation on hidden layers only
      if i < len(self.W) - 1:
          a = self.activation(z)
      else:
          a = z   # linear output for regression

      self.a.append(a)

    return a  # y_hat

  def backward(self, X, y):
    grads_W = [None] * len(self.W)
    grads_b = [None] * len(self.b)

    N = y.shape[0]

    # ---- output layer gradient ----
    y_hat = self.a[-1]
    dz = self.loss_grad(y_hat, y)   # dL/dy_hat

    for i in reversed(range(len(self.W))):

      a_prev = self.a[i]   # a_{i}
      grads_W[i] = a_prev.T @ dz
      grads_b[i] = np.sum(dz, axis=0, keepdims=True)

      if i > 0:
          da = dz @ self.W[i].T
          dz = da * self.activation_grad(self.z[i-1])

    return grads_W, grads_b

  def update(self, grads_W, grads_b):
    for i in range(len(self.W)):
      self.W[i] -= self.lr * grads_W[i]
      self.b[i] -= self.lr * grads_b[i]

