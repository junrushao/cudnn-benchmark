import numpy as np


def relu(x):
  return x * (x > 0)


def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))


def linear(x, w, b):
  assert x.ndim == 1
  if w is not None:
    assert w.ndim == 2
    x = w @ x
  if b is not None:
    assert b.ndim == 1
    x = x + b
  return x


def relu_cell(w, b, x, h, c):
  x = linear(x, w[0], b[0])
  h = linear(h, w[1], b[1])
  return relu(x + h), c


def tanh_cell(w, b, x, h, c):
  x = linear(x, w[0], b[0])
  h = linear(h, w[1], b[1])
  return np.tanh(x + h), c


def gru_cell(w, b, x, h, c):
  r = sigmoid(linear(x, w[0], b[0]) + linear(h, w[3], b[3]))
  i = sigmoid(linear(x, w[1], b[1]) + linear(h, w[4], b[4]))
  hp = np.tanh(linear(x, w[2], b[2]) + r * linear(h, w[5], b[5]))
  return (1.0 - i) * hp + i * h, c


def lstm_cell(w, b, x, h, c):
  i = sigmoid(linear(x, w[0], b[0]) + linear(h, w[4], b[4]))
  f = sigmoid(linear(x, w[1], b[1]) + linear(h, w[5], b[5]))
  cp = np.tanh(linear(x, w[2], b[2]) + linear(h, w[6], b[6]))
  o = sigmoid(linear(x, w[3], b[3]) + linear(h, w[7], b[7]))
  c = f * c + i * cp
  h = o * np.tanh(c)
  return h, c


def run(cell, w, b, seq, h, c):
  seq_y = []
  for x in seq:
    h, c = cell(w, b, x, h, c)
    seq_y.append(h.copy())
  return seq_y, h, c


def verify_relu_cell():
  # =========================================================
  w = [None] * 2
  b = [None] * 2
  w[0] = None
  w[1] = np.array([2, 2, -3, 3], dtype='float32').reshape([2, 2])
  b[0] = np.array([2, 4], dtype='float32').reshape([2])
  b[1] = np.array([-1, 1], dtype='float32').reshape([2])
  # =========================================================
  hx = np.array([0, 0], dtype='float32').reshape([2])
  cx = np.array([0, 0], dtype='float32').reshape([2])
  seq_x = [None] * 10
  seq_x[0] = np.array([-2, 3], dtype='float32').reshape([2])
  seq_x[1] = np.array([3, -5], dtype='float32').reshape([2])
  seq_x[2] = np.array([3, 4], dtype='float32').reshape([2])
  seq_x[3] = np.array([3, -3], dtype='float32').reshape([2])
  seq_x[4] = np.array([-1, -5], dtype='float32').reshape([2])
  seq_x[5] = np.array([0, -3], dtype='float32').reshape([2])
  seq_x[6] = np.array([-1, 0], dtype='float32').reshape([2])
  seq_x[7] = np.array([4, 1], dtype='float32').reshape([2])
  seq_x[8] = np.array([-5, 4], dtype='float32').reshape([2])
  seq_x[9] = np.array([-4, 4], dtype='float32').reshape([2])
  # =========================================================
  hy = np.array([14745, 0], dtype='float32').reshape([2])
  cy = np.array([0, 0], dtype='float32').reshape([2])
  seq_y = [None] * 10
  seq_y[0] = np.array([0, 8], dtype='float32').reshape([2])
  seq_y[1] = np.array([20, 24], dtype='float32').reshape([2])
  seq_y[2] = np.array([92, 21], dtype='float32').reshape([2])
  seq_y[3] = np.array([230, 0], dtype='float32').reshape([2])
  seq_y[4] = np.array([460, 0], dtype='float32').reshape([2])
  seq_y[5] = np.array([921, 0], dtype='float32').reshape([2])
  seq_y[6] = np.array([1842, 0], dtype='float32').reshape([2])
  seq_y[7] = np.array([3689, 0], dtype='float32').reshape([2])
  seq_y[8] = np.array([7374, 0], dtype='float32').reshape([2])
  seq_y[9] = np.array([14745, 0], dtype='float32').reshape([2])
  # =========================================================
  np_seq_y, np_hy, np_cy = run(relu_cell, w, b, seq_x, hx, cx)
  for np_sy, sy in zip(np_seq_y, seq_y):
    np.testing.assert_allclose(np_sy, sy, rtol=1e-6, atol=1e-6)
  np.testing.assert_allclose(np_hy, hy, rtol=1e-6, atol=1e-6)
  np.testing.assert_allclose(np_cy, cy, rtol=1e-6, atol=1e-6)
  print("Verify ReLU cell done!")


def verify_tanh_cell():
  # =========================================================
  w = [None] * 2
  b = [None] * 2
  w[0] = None
  w[1] = np.array([2, 2, -3, 3], dtype='float32').reshape([2, 2])
  b[0] = np.array([2, 4], dtype='float32').reshape([2])
  b[1] = np.array([-1, 1], dtype='float32').reshape([2])
  # =========================================================
  hx = np.array([0, 0], dtype='float32').reshape([2])
  cx = np.array([0, 0], dtype='float32').reshape([2])
  seq_x = [None] * 10
  seq_x[0] = np.array([-2, 3], dtype='float32').reshape([2])
  seq_x[1] = np.array([3, -5], dtype='float32').reshape([2])
  seq_x[2] = np.array([3, 4], dtype='float32').reshape([2])
  seq_x[3] = np.array([3, -3], dtype='float32').reshape([2])
  seq_x[4] = np.array([-1, -5], dtype='float32').reshape([2])
  seq_x[5] = np.array([0, -3], dtype='float32').reshape([2])
  seq_x[6] = np.array([-1, 0], dtype='float32').reshape([2])
  seq_x[7] = np.array([4, 1], dtype='float32').reshape([2])
  seq_x[8] = np.array([-5, 4], dtype='float32').reshape([2])
  seq_x[9] = np.array([-4, 4], dtype='float32').reshape([2])
  # =========================================================
  hy = np.array([-0.764096, 1], dtype='float32').reshape([2])
  cy = np.array([0, 0], dtype='float32').reshape([2])
  seq_y = [None] * 10
  seq_y[0] = np.array([-0.761594, 1], dtype='float32').reshape([2])
  seq_y[1] = np.array([0.999741, 0.999949], dtype='float32').reshape([2])
  seq_y[2] = np.array([1, 1], dtype='float32').reshape([2])
  seq_y[3] = np.array([1, 0.964028], dtype='float32').reshape([2])
  seq_y[4] = np.array([0.999226, -0.107499], dtype='float32').reshape([2])
  seq_y[5] = np.array([0.992385, -0.866827], dtype='float32').reshape([2])
  seq_y[6] = np.array([0.245966, -0.520945], dtype='float32').reshape([2])
  seq_y[7] = np.array([0.999727, 0.998776], dtype='float32').reshape([2])
  seq_y[8] = np.array([-0.00299262, 1], dtype='float32').reshape([2])
  seq_y[9] = np.array([-0.764096, 1], dtype='float32').reshape([2])
  # =========================================================
  np_seq_y, np_hy, np_cy = run(tanh_cell, w, b, seq_x, hx, cx)
  for np_sy, sy in zip(np_seq_y, seq_y):
    np.testing.assert_allclose(np_sy, sy, rtol=1e-6, atol=1e-6)
  np.testing.assert_allclose(np_hy, hy, rtol=1e-6, atol=1e-6)
  np.testing.assert_allclose(np_cy, cy, rtol=1e-6, atol=1e-6)
  print("Verify tanh cell done!")


def verify_gru_cell():
  # =========================================================
  w = [None] * 6
  b = [None] * 6
  w[0] = None
  w[1] = None
  w[2] = None
  w[3] = np.array([2, 2, -3, 3], dtype='float32').reshape([2, 2])
  w[4] = np.array([2, 4, -1, 1], dtype='float32').reshape([2, 2])
  w[5] = np.array([-2, 3, 3, -5], dtype='float32').reshape([2, 2])
  b[0] = np.array([3, 4], dtype='float32').reshape([2])
  b[1] = np.array([3, -3], dtype='float32').reshape([2])
  b[2] = np.array([-1, -5], dtype='float32').reshape([2])
  b[3] = np.array([0, -3], dtype='float32').reshape([2])
  b[4] = np.array([-1, 0], dtype='float32').reshape([2])
  b[5] = np.array([4, 1], dtype='float32').reshape([2])
  # =========================================================
  hx = np.array([0, 0], dtype='float32').reshape([2])
  cx = np.array([0, 0], dtype='float32').reshape([2])
  seq_x = [None] * 10
  seq_x[0] = np.array([-5, 4], dtype='float32').reshape([2])
  seq_x[1] = np.array([-4, 4], dtype='float32').reshape([2])
  seq_x[2] = np.array([1, 2], dtype='float32').reshape([2])
  seq_x[3] = np.array([-2, -1], dtype='float32').reshape([2])
  seq_x[4] = np.array([0, 2], dtype='float32').reshape([2])
  seq_x[5] = np.array([-3, -1], dtype='float32').reshape([2])
  seq_x[6] = np.array([3, 1], dtype='float32').reshape([2])
  seq_x[7] = np.array([-5, -4], dtype='float32').reshape([2])
  seq_x[8] = np.array([-1, 3], dtype='float32').reshape([2])
  seq_x[9] = np.array([-4, -3], dtype='float32').reshape([2])
  # =========================================================
  hy = np.array([-0.998948, -0.995195], dtype='float32').reshape([2])
  cy = np.array([0, 0], dtype='float32').reshape([2])
  seq_y = [None] * 10
  seq_y[0] = np.array([-0.952544, -0.00179997], dtype='float32').reshape([2])
  seq_y[1] = np.array([-0.998906, -0.125215], dtype='float32').reshape([2])
  seq_y[2] = np.array([-0.244914, -0.590022], dtype='float32').reshape([2])
  seq_y[3] = np.array([-0.929613, -0.994685], dtype='float32').reshape([2])
  seq_y[4] = np.array([-0.154062, -0.252855], dtype='float32').reshape([2])
  seq_y[5] = np.array([-0.918886, -0.987749], dtype='float32').reshape([2])
  seq_y[6] = np.array([0.400333, -0.869267], dtype='float32').reshape([2])
  seq_y[7] = np.array([-0.995206, -0.999966], dtype='float32').reshape([2])
  seq_y[8] = np.array([-0.927997, -0.12594], dtype='float32').reshape([2])
  seq_y[9] = np.array([-0.998948, -0.995195], dtype='float32').reshape([2])
  # =========================================================
  np_seq_y, np_hy, np_cy = run(gru_cell, w, b, seq_x, hx, cx)
  for np_sy, sy in zip(np_seq_y, seq_y):
    np.testing.assert_allclose(np_sy, sy, rtol=1e-6, atol=1e-6)
  np.testing.assert_allclose(np_hy, hy, rtol=1e-6, atol=1e-6)
  np.testing.assert_allclose(np_cy, cy, rtol=1e-6, atol=1e-6)
  print("Verify GRU cell done!")


def verify_lstm_cell():
  # =========================================================
  w = [None] * 8
  b = [None] * 8
  w[0] = None
  w[1] = None
  w[2] = None
  w[3] = None
  w[4] = np.array([2, 2, -3, 3], dtype='float32').reshape([2, 2])
  w[5] = np.array([2, 4, -1, 1], dtype='float32').reshape([2, 2])
  w[6] = np.array([-2, 3, 3, -5], dtype='float32').reshape([2, 2])
  w[7] = np.array([3, 4, 3, -3], dtype='float32').reshape([2, 2])
  b[0] = np.array([-1, -5], dtype='float32').reshape([2])
  b[1] = np.array([0, -3], dtype='float32').reshape([2])
  b[2] = np.array([-1, 0], dtype='float32').reshape([2])
  b[3] = np.array([4, 1], dtype='float32').reshape([2])
  b[4] = np.array([-5, 4], dtype='float32').reshape([2])
  b[5] = np.array([-4, 4], dtype='float32').reshape([2])
  b[6] = np.array([1, 2], dtype='float32').reshape([2])
  b[7] = np.array([-2, -1], dtype='float32').reshape([2])
  # =========================================================
  hx = np.array([0, 0], dtype='float32').reshape([2])
  cx = np.array([0, 0], dtype='float32').reshape([2])
  seq_x = [None] * 10
  seq_x[0] = np.array([0, 2], dtype='float32').reshape([2])
  seq_x[1] = np.array([-3, -1], dtype='float32').reshape([2])
  seq_x[2] = np.array([3, 1], dtype='float32').reshape([2])
  seq_x[3] = np.array([-5, -4], dtype='float32').reshape([2])
  seq_x[4] = np.array([-1, 3], dtype='float32').reshape([2])
  seq_x[5] = np.array([-4, -3], dtype='float32').reshape([2])
  seq_x[6] = np.array([4, -4], dtype='float32').reshape([2])
  seq_x[7] = np.array([1, 1], dtype='float32').reshape([2])
  seq_x[8] = np.array([-4, -2], dtype='float32').reshape([2])
  seq_x[9] = np.array([3, 1], dtype='float32').reshape([2])
  # =========================================================
  hy = np.array([0.0470013, 0.355363], dtype='float32').reshape([2])
  cy = np.array([0.0473509, 0.531963], dtype='float32').reshape([2])
  seq_y = [None] * 10
  seq_y[0] = np.array([0, 0.5491], dtype='float32').reshape([2])
  seq_y[1] = np.array([-0.000248474, 0.00493969], dtype='float32').reshape([2])
  seq_y[2] = np.array([0.0471861, 0.373676], dtype='float32').reshape([2])
  seq_y[3] = np.array([-2.03695e-06, 0.000129828], dtype='float32').reshape([2])
  seq_y[4] = np.array([-0.000507384, 0.682047], dtype='float32').reshape([2])
  seq_y[5] = np.array([-0.00011725, 0.000419007], dtype='float32').reshape([2])
  seq_y[6] = np.array([0.118247, -6.0058e-05], dtype='float32').reshape([2])
  seq_y[7] = np.array([0.0120833, 0.307651], dtype='float32').reshape([2])
  seq_y[8] = np.array([-2.3041e-05, 0.00197692], dtype='float32').reshape([2])
  seq_y[9] = np.array([0.0470013, 0.355363], dtype='float32').reshape([2])
  # =========================================================
  np_seq_y, np_hy, np_cy = run(lstm_cell, w, b, seq_x, hx, cx)
  for np_sy, sy in zip(np_seq_y, seq_y):
    np.testing.assert_allclose(np_sy, sy, rtol=1e-6, atol=1e-6)
  np.testing.assert_allclose(np_hy, hy, rtol=1e-6, atol=1e-6)
  np.testing.assert_allclose(np_cy, cy, rtol=1e-6, atol=1e-6)
  print("Verify LSTM cell done!")


verify_relu_cell()
verify_tanh_cell()
verify_gru_cell()
verify_lstm_cell()
