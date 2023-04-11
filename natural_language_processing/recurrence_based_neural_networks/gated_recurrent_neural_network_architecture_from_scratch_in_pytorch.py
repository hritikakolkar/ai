import torch
import torch.nn as nn
class GRU(nn.Module):
    def __init__(self, input_size , hidden_size, output_size):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Reset_gate Weights and Bias
        self.Wxr = nn.Parameter(torch.randn(self.input_size, self.hidden_size))
        self.Whr = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size))
        self.br = nn.Parameter(torch.randn(1, self.hidden_size))
        
        # Update Gate Weights and Bias
        self.Wxz = nn.Parameter(torch.randn(self.input_size, self.hidden_size))
        self.Whz = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size))
        self.bz = nn.Parameter(torch.randn(1, self.hidden_size))

        # Candidate Hidden Cell Weights and Bias
        self.Wxh = nn.Parameter(torch.randn(self.input_size, self.hidden_size))
        self.Whh = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size))
        self.bh = nn.Parameter(torch.randn(1, self.hidden_size))

        # For Outputs
        self.Whq = nn.Parameter(torch.randn(self.hidden_size, self.output_size))
        self.bq = nn.Parameter(torch.randn(1, self.output_size))
    
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
            
            # Calculating Reset Gate
            # Rt = σ(Xt*Wxr + Ht−1*Whr + br)
            R_t = torch.sigmoid(torch.mm(X_t, self.Wxr) + torch.mm(H_t, self.Whr) + self.br)

            # Calculating Update Gate
            # Ut = σ(Xt*Wxu + Ht−1*Whu + bu),
            Z_t = torch.sigmoid(torch.mm(X_t, self.Wxz) + torch.mm(H_t, self.Whz) + self.bz)

            # Calculating Candidate Hidden Cell 
            # H˜t = tanh(Xt*Wxh + (Rt ⊙ Ht−1)*Whh + bh),
            H_tilda_t = torch.tanh(torch.mm(X_t, self.Wxh) + torch.mm(torch.multiply(R_t, H_t), self.Whh) + self.bh)

            # Calculating Hiddent State
            # Ht = Zt ⊙ Ht−1 + (1 − Zt) ⊙ H
            H_t = torch.multiply(Z_t, H_t) + torch.multiply((1 - Z_t),H_tilda_t)

            # Output at timestep t
            O_t = torch.mm(H_t, self.Whq) + self.bq
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
    gru = GRU(input_size=input_size, hidden_size= hidden_size, output_size= output_size)
 
    # run forward propagation
    outputs, last_hidden_state = gru(X)

    print("Output Size:- ",outputs.size())
    print("Last Hidden State Size :- ",last_hidden_state.size())