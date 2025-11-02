import unittest
import numpy as np
from teenygrad.tensor import Tensor
from teenygrad.nn.lstm import LSTM


class TestRecurrent(unittest.TestCase):
    def test_lstm_forward(self):
        input_size = 10
        hidden_size = 20
        num_layers = 2
        seq_len = 5
        batch_size = 3

        lstm = LSTM(input_size, hidden_size, num_layers)
        x = Tensor.rand(seq_len, batch_size, input_size)

        output, hc = lstm(x)

        self.assertEqual(output.shape, (seq_len, batch_size, hidden_size))
        self.assertEqual(hc.shape, (num_layers, 2 * batch_size, hidden_size))

        output.numpy()
        hc.numpy()

    def test_lstm_backward(self):
        input_size = 10
        hidden_size = 20
        num_layers = 1
        seq_len = 5
        batch_size = 3

        lstm = LSTM(input_size, hidden_size, num_layers, dropout=0.5)
        x = Tensor.rand(seq_len, batch_size, input_size)

        for cell in lstm.cells:
            cell.weights_ih.requires_grad = True
            cell.weights_hh.requires_grad = True
            cell.bias_ih.requires_grad = True
            cell.bias_hh.requires_grad = True

        x.requires_grad = True

        output, _ = lstm(x)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)
        for cell in lstm.cells:
            self.assertIsNotNone(cell.weights_ih.grad)
            self.assertEqual(cell.weights_ih.grad.shape, cell.weights_ih.shape)
            self.assertIsNotNone(cell.weights_hh.grad)
            self.assertEqual(cell.weights_hh.grad.shape, cell.weights_hh.shape)
            self.assertIsNotNone(cell.bias_ih.grad)
            self.assertEqual(cell.bias_ih.grad.shape, cell.bias_ih.shape)
            self.assertIsNotNone(cell.bias_hh.grad)
            self.assertEqual(cell.bias_hh.grad.shape, cell.bias_hh.shape)


if __name__ == '__main__':
    unittest.main()
