from teenygrad.tensor import Tensor


class LSTMCell:
    def __init__(self, input_size, hidden_size, dropout=0.0):
        self.dropout = dropout
        self.weights_ih = Tensor.uniform(hidden_size * 4, input_size)
        self.bias_ih = Tensor.uniform(hidden_size * 4)
        self.weights_hh = Tensor.uniform(hidden_size * 4, hidden_size)
        self.bias_hh = Tensor.uniform(hidden_size * 4)

    def __call__(self, x, hc):
        gates = x.linear(self.weights_ih.T, self.bias_ih) + hc[:x.shape[0]].linear(self.weights_hh.T, self.bias_hh)
        i, f, g, o = gates.chunk(4, 1)
        i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
        c = (f * hc[x.shape[0]:]) + (i * g)
        h = (o * c.tanh()).dropout(self.dropout)
        return Tensor.cat(h, c).realize()


class LSTM:
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = [LSTMCell(input_size if i == 0 else hidden_size, hidden_size, dropout if i != num_layers - 1 else 0) for i in range(num_layers)]

    def do_step(self, x, hc):
        new_hc = [x]
        for i, cell in enumerate(self.cells):
            new_hc.append(cell(new_hc[i][:x.shape[0]], hc[i]))
        return Tensor.stack(new_hc[1:]).realize()

    def __call__(self, x, hc=None):
        if hc is None:
            hc = Tensor.zeros(self.num_layers, 2 * x.shape[1], self.hidden_size, requires_grad=False)
        output = None
        for t in range(x.shape[0]):
            hc = self.do_step(x[t], hc)
            if output is None:
                output = hc[-1:, :x.shape[1]]
            else:
                output = output.cat(hc[-1:, :x.shape[1]], dim=0).realize()
        return output, hc
