import numpy as np


def sigmoid(z):
  z = np.clip(z, -500, 500)
  return 1 / (1 + np.exp(-z))


def compute_loss(w, X, y):
  z = X @ w
  predictions = sigmoid(z)
  predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
  loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
  return loss


def compute_gradient(w, X, y):
  z = X @ w
  predictions = sigmoid(z)
  errors = predictions - y
  gradient = X.T @ errors / len(y)
  return gradient


def validation_accuracy(w, X_val, y_val):
  probabilities = sigmoid(X_val @ w)
  predictions = (probabilities >= 0.5).astype(int)
  accuracy = np.mean(predictions == y_val)
  return accuracy


def gradient_descent_logistic(
    X_train,
    y_train,
    X_val,
    y_val,
    learning_rate=0.1,
    n_steps=1000,
    tolerance=1e-6,
):
  w = np.zeros(X_train.shape[1])
  loss_history = [compute_loss(w, X_train, y_train)]
  val_accuracy_history = [validation_accuracy(w, X_val, y_val)]
  weights_history = [w.copy()]

  for step in range(1, n_steps + 1):
    grad = compute_gradient(w, X_train, y_train)
    w -= learning_rate * grad

    loss = compute_loss(w, X_train, y_train)
    loss_history.append(loss)

    acc = validation_accuracy(w, X_val, y_val)
    val_accuracy_history.append(acc)

    if step % 10 == 0:
      weights_history.append(w.copy())

    if (
        len(loss_history) > 1
        and np.abs(loss_history[-2] - loss_history[-1]) < tolerance
    ):
      print(f'Converged at step {step}')
      break

    if step % 200 == 0:
      print(f'Step {step}: Loss = {loss:.4f}, Validation Accuracy = {acc:.4f}')

  return w, loss_history, val_accuracy_history, weights_history