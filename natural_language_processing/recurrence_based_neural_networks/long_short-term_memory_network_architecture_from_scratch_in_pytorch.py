import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size , hidden_size, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Input Gate Weights
        self.Wxi = nn.Parameter(torch.randn(self.input_size, self.hidden_size))
        self.Whi = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size))
        self.bi = nn.Parameter(torch.randn(1, hidden_size))

        # Forget Gate Weights
        self.Wxf = nn.Parameter(torch.randn(self.input_size, self.hidden_size))
        self.Whf = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size))
        self.bf = nn.Parameter(torch.randn(1, self.hidden_size))

        # Output Gate Weights
        self.Wxo = nn.Parameter(torch.randn(self.input_size, self.hidden_size))
        self.Who = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size))
        self.bo = nn.Parameter(torch.randn(1, self.hidden_size))

        # Candidate Memory Cell
        self.Wxc = nn.Parameter(torch.randn(self.input_size, self.hidden_size))
        self.Whc = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size))
        self.bc = nn.Parameter(torch.randn(1, self.hidden_size))

        # For output
        self.Whq = nn.Parameter(torch.randn(self.hidden_size, self.output_size))
        self.bq = nn.Parameter(torch.randn(1, self.output_size))
    
    def forward(self, X, H_0 = None, C_0 = None):
        # X has shape (batch_size, sequence_length, input_size)
        batch_size, sequence_length, input_size = X.size()

        if H_0 is None:
            H_t = torch.zeros((batch_size, self.hidden_size))
        else:
            H_t = H_0

        if C_0 is None:
            C_t = torch.zeros((batch_size, self.hidden_size))
        else:
            C_t = C_0

        outputs = torch.zeros((batch_size, sequence_length, self.output_size))

        for t in range(sequence_length):
            X_t = X[:,t,:]
            
            #Input Gate
            #It = σ(XtWxi + Ht−1Whi + bi) 
            I_t = torch.sigmoid(torch.mm(X_t, self.Wxi) + torch.mm(H_t, self.Whi) + self.bi)
            
            #Forget Gate
            #Ft = σ(XtWxf + Ht−1Whf + bf)
            F_t = torch.sigmoid(torch.mm(X_t, self.Wxf) + torch.mm(H_t, self.Whf) + self.bf)
            
            #Output Gate
            #Ot = σ(XtWxo + Ht−1Who + bo)
            O_t = torch.sigmoid(torch.mm(X_t, self.Wxo) + torch.mm(H_t, self.Who) + self.bo)
            
            #Candidate State Cell
            #C˜t = tanh(Xt*Wxc + Ht−1*Whc + bc)
            C_tilda_t = torch.tanh(torch.mm(X_t,self.Wxc) + torch.mm(H_t, self.Whc) + self.bc)
            
            # Calculating Current Cell State using previous cell state
            #Ct = Ft ⊙ Ct−1 + It ⊙ C˜t
            C_t = torch.multiply(F_t, C_t) + torch.multiply(I_t, C_tilda_t)
            
            # Calculating Current Hidden State using previous hidden state
            #Ht = Ot ⊙ tanh(Ct)
            H_t = torch.multiply(O_t, torch.tanh(C_t))
            
            #Output at time t (didn't mentioned in architecture)
            O_t = torch.mm(H_t, self.Whq) + self.bq
            
            outputs[:, t, :] = O_t
        return outputs, H_t, C_t
    
if __name__ == "main":
    batch_size = 19 # No of sequences in one batch
    sequence_length = 8 # No of words in each Sequence
    input_size = 4 # No of features of each words aka Word Embedding Size 
    X = torch.randn(batch_size, sequence_length, input_size)

    # initialize 
    hidden_size = 10
    output_size = 15
    lstm = LSTM(input_size=input_size, hidden_size= hidden_size, output_size= output_size)

    # run forward propagation
    outputs, last_hidden_state, last_cell_state = lstm(X)

    print("Output Size:- ",outputs.size())
    print("Last Hidden State Size :- ",last_hidden_state.size())
    print("Last Cell State Size :- ",last_cell_state.size())