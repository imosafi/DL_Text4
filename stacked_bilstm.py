import dynet as dy



class StackedBilstm(object):
    def __init__(self, in_dim, out_dim, mlp_hidden_dim, model):
        # self.builder = dy.VanillaLSTMBuilder(1, in_dim, out_dim, model)
        # self.W1 = model.add_parameters((mlp_hidden_dim, out_dim))
        # self.b1 = model.add_parameters(mlp_hidden_dim)
        # self.W2 = model.add_parameters((2, mlp_hidden_dim))
        # self.b2 = model.add_parameters(2)

    def __call__(self, sequence):
        # lstm = self.builder.initial_state()
        # W1 = self.W1.expr()
        # W2 = self.W2.expr()
        # b1 = self.b1.expr()
        # b2 = self.b2.expr()
        # outputs = lstm.transduce(sequence)
        # result = W2 * (dy.tanh(W1 * outputs[-1]) + b1) + b2
        # return result