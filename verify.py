import numpy as np


def relu(x):
  return x * (x > 0)


def linear(x, w, b):
  if w is not None:
    x = w @ x
  if b is not None:
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

verify_relu_cell()
verify_tanh_cell()
