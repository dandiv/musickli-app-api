import tensorflow as tf
from sklearn.metrics import accuracy_score

def custom_accuracy(y_true, y_pred, threshold=0.6):
    y_pred_np = tf.cast(y_pred > threshold, dtype=tf.int32)
    y_true_np = tf.cast(y_true, dtype=tf.int32)
    accuracy = tf.numpy_function(accuracy_score, [y_true_np, y_pred_np], tf.float64)
    return accuracy