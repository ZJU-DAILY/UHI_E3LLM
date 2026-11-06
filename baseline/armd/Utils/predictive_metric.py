import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf1
tf.compat.v1.disable_eager_execution()
import numpy as np
from sklearn.metrics import mean_absolute_error
from Utils.metric_utils import extract_time


def predictive_score_metrics(ori_data, generated_data):
  tf1.reset_default_graph()

  no, seq_len, dim = ori_data.shape

  ori_time, ori_max_seq_len = extract_time(ori_data)
  generated_time, generated_max_seq_len = extract_time(ori_data)
  max_seq_len = max([ori_max_seq_len, generated_max_seq_len])

  hidden_dim = int(dim/2)
  iterations = 5000
  batch_size = 128

  X = tf1.placeholder(tf.float32, [None, max_seq_len-1, dim-1], name = "myinput_x")
  T = tf1.placeholder(tf.int32, [None], name = "myinput_t")
  Y = tf1.placeholder(tf.float32, [None, max_seq_len-1, 1], name = "myinput_y")

  def predictor (x, t):
    with tf1.variable_scope("predictor", reuse = tf1.AUTO_REUSE) as vs:
      p_cell = tf1.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name = 'p_cell')
      p_outputs, p_last_states = tf1.nn.dynamic_rnn(p_cell, x, dtype=tf.float32, sequence_length = t)
      y_hat_logit = tf1.layers.dense(p_outputs, 1, activation=None)
      y_hat = tf.nn.sigmoid(y_hat_logit)
      p_vars = [v for v in tf1.all_variables() if v.name.startswith(vs.name)]

    return y_hat, p_vars

  y_pred, p_vars = predictor(X, T)
  p_loss = tf1.losses.absolute_difference(Y, y_pred)
  p_solver = tf1.train.AdamOptimizer().minimize(p_loss, var_list = p_vars)

  sess = tf1.Session()
  sess.run(tf1.global_variables_initializer())

  from tqdm.auto import tqdm

  for itt in tqdm(range(iterations), desc='training', total=iterations):

    idx = np.random.permutation(len(generated_data))
    train_idx = idx[:batch_size]

    X_mb = list(generated_data[i][:-1,:(dim-1)] for i in train_idx)
    T_mb = list(generated_time[i]-1 for i in train_idx)
    Y_mb = list(np.reshape(generated_data[i][1:,(dim-1)],[len(generated_data[i][1:,(dim-1)]),1]) for i in train_idx)

    _, step_p_loss = sess.run([p_solver, p_loss], feed_dict={X: X_mb, T: T_mb, Y: Y_mb})

  idx = np.random.permutation(len(ori_data))
  train_idx = idx[:no]

  X_mb = list(ori_data[i][:-1,:(dim-1)] for i in train_idx)
  T_mb = list(ori_time[i]-1 for i in train_idx)
  Y_mb = list(np.reshape(ori_data[i][1:,(dim-1)], [len(ori_data[i][1:,(dim-1)]),1]) for i in train_idx)

  pred_Y_curr = sess.run(y_pred, feed_dict={X: X_mb, T: T_mb})

  MAE_temp = 0
  for i in range(no):
    MAE_temp = MAE_temp + mean_absolute_error(Y_mb[i], pred_Y_curr[i,:,:])

  predictive_score = MAE_temp / no

  return predictive_score
