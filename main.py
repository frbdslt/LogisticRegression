import matplotlib.pyplot as plt
import numpy as np
from src.logistic import (
    gradient_descent_logistic,
    sigmoid,
)
from src.preprocessing import (
    LoadAndCleanData,
    StandardizeFeatures,
    TrainTestSplit,
)


def ConfusionMatrix(y_true, y_pred):
  tn = np.sum((y_true == 0) & (y_pred == 0))
  fp = np.sum((y_true == 0) & (y_pred == 1))
  fn = np.sum((y_true == 1) & (y_pred == 0))
  tp = np.sum((y_true == 1) & (y_pred == 1))
  return np.array([[tn, fp], [fn, tp]])

def RecallScore(y_true, y_pred):
    cm = ConfusionMatrix(y_true, y_pred)

    tp = cm[1, 1]
    fn = cm[1, 0]

    denominator = tp + fn

    if denominator == 0:
        return 0.0

    return tp / denominator


def PrecisionScore(y_true, y_pred):
    cm = ConfusionMatrix(y_true, y_pred)

    tp = cm[1, 1]
    fp = cm[0, 1]

    denominator = tp + fp

    if denominator == 0:
        return 0.0

    return tp / denominator


def F1Score(y_true, y_pred):
    precision = PrecisionScore(y_true, y_pred)
    recall = RecallScore(y_true, y_pred)

    denominator = precision + recall

    if denominator == 0:
        return 0.0

    return 2 * precision * recall / denominator

def main():
  print('Loading and cleaning dataset...')
  X, y = LoadAndCleanData('data/heart.csv')

  X_train, y_train, X_val, y_val, X_test, y_test = TrainTestSplit(
      X, y, test_ratio=0.2, val_ratio=0.15
  )

  X_train, X_val, X_test = StandardizeFeatures(X_train, X_val, X_test)

  print(f'Training samples: {len(X_train)}')
  print(f'Validation samples: {len(X_val)}')
  print(f'Test samples: {len(X_test)}')

  print('\nTraining Logistic Regression model...')
  w, loss_history, val_acc_history, _ = gradient_descent_logistic(
      X_train, y_train, X_val, y_val, learning_rate=0.1, n_steps=1000
  )

  test_probs = sigmoid(X_test @ w)
  test_preds = (test_probs >= 0.5).astype(int)

  test_acc = np.mean(test_preds == y_test)
  cm = ConfusionMatrix(y_test, test_preds)
  test_probabilities = sigmoid(X_test @ w)
  test_predictions = (test_probabilities >= 0.5).astype(int)

  test_accuracy = np.mean(test_predictions == y_test)
  test_precision = PrecisionScore(y_test, test_predictions)
  test_recall = RecallScore(y_test, test_predictions)
  test_f1 = F1Score(y_test, test_predictions)

  print('\n--- Final Test Results ---')
  print(f'Test Accuracy: {test_acc:.4f}')
  print('Confusion Matrix [[TN, FP], [FN, TP]]:\n', cm)
  print("\n--- Final Test Results ---")
  print(f"Accuracy  : {test_accuracy:.4f}")
  print(f"Precision : {test_precision:.4f}")
  print(f"Recall    : {test_recall:.4f}")
  print(f"F1-Score  : {test_f1:.4f}")


  plt.figure(figsize=(12, 5))

  plt.subplot(1, 2, 1)
  plt.plot(loss_history, color='red', lw=2)
  plt.title('Training Loss (Binary Cross-Entropy)')
  plt.xlabel('Step')
  plt.ylabel('Loss')
  plt.grid(True)

  plt.subplot(1, 2, 2)
  plt.plot(val_acc_history, color='blue', lw=2)
  plt.title('Validation Accuracy per Step')
  plt.xlabel('Step')
  plt.ylabel('Accuracy')
  plt.grid(True)

  plt.tight_layout()
  plt.show()


if __name__ == '__main__':
  main()