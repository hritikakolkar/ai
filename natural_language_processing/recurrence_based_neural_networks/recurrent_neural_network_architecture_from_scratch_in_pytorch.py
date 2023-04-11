import torch
import torch.nn as nn
class RNN(nn.Module):
    def __init__(self, input_size , hidden_size, output_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Input, Hidden and Output Weights
        self.Wih = nn.Parameter(torch.randn(input_size, hidden_size))
        self.Whh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Who = nn.Parameter(torch.randn(hidden_size, output_size))

        # Bias
        self.bh = nn.Parameter(torch.randn(1, hidden_size))
        self.bo = nn.Parameter(torch.randn(1, output_size))


    def forward(self, X, H_0 = None):
        # X has shape (batch_size, sequence_length, input_size)
        batch_size, sequence_length, input_size = X.size()
        if H_0 is None:
            H_t = torch.zeros((batch_size, self.hidden_size))
        else:
            H_t = H_0
        
        outputs = torch.zeros((batch_size, sequence_length, self.output_size))

        for t in range(sequence_length):
            # input at timestep t
            X_t = X[:, t, :]
            # Calculating hidden state at t
            H_t = torch.tanh(torch.mm(X_t, self.Wih) + torch.mm(H_t, self.Whh) + self.bh)
            # Calculating output at time state t
            O_t = torch.mm(H_t, self.Who) + self.bo

            outputs[:, t, :] = O_t

        return outputs, H_t
    
if __name__ == "__main__":
    batch_size = 3 # No of sequences in one batch
    sequence_length = 8 # No of words in each Sequence
    input_size = 4 # No of features of each words aka Word Embedding Size 
    X = torch.randn(batch_size, sequence_length, input_size)

    # initialize RNN
    hidden_size = 10
    output_size = 15
    rnn = RNN(input_size=input_size, hidden_size= hidden_size, output_size= output_size)
 
    # run forward propagation
    outputs, last_hidden_state = rnn(X)

    print("Outputs Size:- ",outputs.size())
    print("Last Hidden State Size:- ",last_hidden_state.size())