import numpy as np
from model import MLP
import matplotlib.pyplot as plt
from data_pipeline import load_data, preprocess, split

df = load_data("../data/student.csv")
X, y = preprocess(df, "Exam_Score")
X_train, X_test, y_train, y_test = split(X, y)
# ---- config ----
epochs = 500
batch_size = 64
lr = 1e-3

# dims
input_dim = X_train.shape[1]
hidden_dims = [64, 32]
output_dim = 1

# ---- functions ----
def mse(y_hat, y):
    return np.mean((y_hat - y)**2)

def mse_grad(y_hat, y):
    return 2*(y_hat - y)/y.shape[0]

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(np.float32)

# ---- model ----
model = MLP(
    input_dim=input_dim,
    hidden_dims=hidden_dims,
    output_dim=output_dim,
    lr=lr,
    batch_size=batch_size,
    activation=relu,
    activation_grad=relu_grad,
    loss_fn=mse,
    loss_grad=mse_grad
)

# ---- training loop ----
N = X_train.shape[0]
train_losses = []
val_losses = []

for epoch in range(epochs):

    idx = np.random.permutation(N)
    Xs = X_train[idx]
    ys = y_train[idx]

    for i in range(0, N, batch_size):
        Xb = Xs[i:i+batch_size]
        yb = ys[i:i+batch_size]

        y_hat = model.forward(Xb)
        gW, gB = model.backward(Xb, yb)
        model.update(gW, gB)

    # ---- metrics ----
    train_pred = model.forward(X_train)
    val_pred = model.forward(X_test)

    train_loss = mse(train_pred, y_train)
    val_loss = mse(val_pred, y_test)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if epoch % 50 == 0:
        print(epoch, train_loss, val_loss)

plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("MSE")
plt.title("Training curve")
plt.show()
print(train_losses[-1], val_losses[-1])
