import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torch.nn.functional as F


class GlobalModel(nn.Module):
    def __init__(self, D, hidden_size, K, sequence_length = 24, output_window_size =24):
        super(GlobalModel, self).__init__()
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.output_window_size = output_window_size

        self.K = K
        self.D = D

        # self.rnn = nn.LSTM(input_size=D, hidden_size=hidden_size, batch_first=True)
        self.rnn = nn.RNN(input_size=D, hidden_size=hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, D*output_window_size, bias=True)
        self.dec = nn.Linear(D, K, bias=True)

        # self.fcW = nn.Linear(K, 1, bias=True)
        self.W = nn.Parameter(torch.randn(K))
        
        # Bias term for the output
        self.b0 = nn.Parameter(torch.randn(output_window_size))

    def forward(self, x): #[batch_size, input_window_size, D]
        # x: [batch_size, input_window_size, D]
        output, _ = self.rnn(x)
        
        # Use the output of the last time step from RNN
        output = self.fc(output[:, -1, :])  # [batch_size, K*output_window_size]
        
        # Reshape output to [batch_size, output_window_size, D]
        output = output.view(-1, self.output_window_size, self.D)
        
        reduced_output = torch.zeros(output.size(0), self.output_window_size, self.K, device=output.device)
        for i in range(self.output_window_size):
            reduced_output[:, i, :] = self.dec(output[:, i, :])  # [batch_size, output_window_size, K]
        
        final_output = torch.matmul(reduced_output, self.W) + self.b0  # [batch_size, output_window_size]

        
        return final_output
    
    def get_weight_matrix(self):
        # access the weight matrix W of the fcW layer
        return self.W
        # return self.fcW.weight


    

# class LocalModel(nn.Module):
#     def __init__(self, D_plus_C, hidden_size, output_window_size=24):
#         super(LocalModel, self).__init__()
#         self.rnn = nn.RNN(input_size=D_plus_C, hidden_size=hidden_size, batch_first=True)
#         # Assuming the output is a single value per sequence, output_window_size=1 for simplicity
#         self.fc = nn.Linear(hidden_size,2*output_window_size, bias=True)
#         self.softplus = nn.Softplus()
#         self.output_window_size = output_window_size

#     def forward(self, x):
#         # x shape: [batch_size, sequence_length, D_plus_C]
#         output, _ = self.rnn(x)
#         # Process the last output of the RNN sequence
#         output = self.fc(output[:, -1, :])
#          # Reshape the output to have separate dimensions for the output_window_size and the 2 output parameters
#         output = output.view(-1, self.output_window_size, 2)  # Shape: [batch_size, output_window_size, 2]
        
#         # Assuming the goal is to predict a positive value (e.g., variance), apply Softplus
#         sigma = self.softplus(output[:,:,0])
#         p = torch.sigmoid(output[:,:,1])

#         return sigma, p


class LocalModel(nn.Module):
    def __init__(self, D_plus_C, hidden_size, output_window_size=24):
        super(LocalModel, self).__init__()
        self.rnn = nn.RNN(input_size=D_plus_C, hidden_size=hidden_size, batch_first=True)
        # Assuming the output is a single value per sequence, output_window_size=1 for simplicity
        self.fc = nn.Linear(hidden_size,3*output_window_size, bias=True)
        self.softplus = nn.Softplus()
        self.output_window_size = output_window_size

    def forward(self, x):
        # x shape: [batch_size, sequence_length, D_plus_C]
        output, _ = self.rnn(x)
        # Process the last output of the RNN sequence
        output = self.fc(output[:, -1, :])
         # Reshape the output to have separate dimensions for the output_window_size and the 2 output parameters
        output = output.view(-1, self.output_window_size, 3)  # Shape: [batch_size, output_window_size, 2]
        
        # Assuming the goal is to predict a positive value (e.g., variance), apply Softplus
        sigma = self.softplus(output[:,:,0])
        p = torch.sigmoid(output[:,:,1])
        local_mu = (output[:,:,2])

        return sigma, p, local_mu

    



class TimeSeriesDatasetSingleSubject(Dataset):
    def __init__(self, X, Y, cat_vars, binary_event, input_window_size, output_window_size):
        self.X = X
        self.Y = Y
        self.cat_vars = cat_vars
        self.binary_event = binary_event
        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        # Compute the total number of sliding windows that can be generated given the stride of 1
        self.total_windows = len(X) - input_window_size - output_window_size + 1

    def __len__(self):
        return self.total_windows

    def __getitem__(self, idx):
        # Calculate window start and end indices for input and output windows dynamically
        input_start = idx
        input_end = idx + self.input_window_size
        output_start = input_end
        output_end = output_start + self.output_window_size
        
        # Slice the data for this window
        X = self.X[input_start:input_end]
        Y = self.Y[input_start:input_end]  # Assuming you want to predict the same window shape as input
        trueX = self.X[output_start:output_end]
        trueY = self.Y[output_start:output_end]
        cat_vars = self.cat_vars[input_start:input_end]
        binary_event = self.binary_event[input_start:input_end]
        true_binary_event = self.binary_event[output_start:output_end]
        
        return X, Y, trueX, trueY, cat_vars, binary_event, true_binary_event

# class TimeSeriesDatasetSingleSubject(Dataset):
#     def __init__(self, X, Y, cat_vars, binary_event, input_window_size, output_window_size):
#         self.input_window_size = input_window_size
#         self.output_window_size = output_window_size
#         # Adjust for sliding window; construct windows for X and Y
#         self.X, self.Y, self.trueX, self.trueY, self.cat_vars, self.binary_event, self.true_binary_event = self.create_windows(X, Y, cat_vars, binary_event)

#     def create_windows(self, X, Y, cat_vars, binary_event):
#         n = len(X)
#         stride = 1  # This can be adjusted if you want non-overlapping windows
#         windows_X, windows_Y, windows_trueX, windows_trueY, windows_cat_vars, windows_binary_event, windows_true_binary_event = [], [], [], [], [], [], []
        
#         for i in range(n - self.input_window_size - self.output_window_size + 1):
#             windows_X.append(X[i:i+self.input_window_size])
#             windows_Y.append(Y[i:i+self.input_window_size])
#             windows_trueX.append(X[i+self.input_window_size:i+self.input_window_size+self.output_window_size])
#             windows_trueY.append(Y[i+self.input_window_size:i+self.input_window_size+self.output_window_size])
#             windows_cat_vars.append(cat_vars[i:i+self.input_window_size])
#             windows_binary_event.append(binary_event[i:i+self.input_window_size])
#             windows_true_binary_event.append(binary_event[i+self.input_window_size:i+self.input_window_size+self.output_window_size])
        
#         return torch.stack(windows_X), torch.stack(windows_Y), torch.stack(windows_trueX), torch.stack(windows_trueY), torch.stack(windows_cat_vars), torch.stack(windows_binary_event), torch.stack(windows_true_binary_event)

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         return self.X[idx], self.Y[idx], self.trueX[idx], self.trueY[idx], self.cat_vars[idx], self.binary_event[idx], self.true_binary_event[idx]



# class TimeSeriesDatasetSingleSubject(torch.utils.data.Dataset):
#     def __init__(self, X, Y, trueX, trueY, cat_vars, binary_event):
#         self.X = X
#         self.Y = Y
#         self.trueX = trueX
#         self.trueY = trueY
#         self.cat_vars = cat_vars
#         self.binary_event = binary_event

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         return self.X[idx], self.Y[idx], self.trueX[idx], self.trueY[idx], self.cat_vars[idx], self.binary_event[idx]


# class TimeSeriesDataset(Dataset): # Flat
#     torch.manual_seed(1)
#     """
#     Args:
#         J (int): Number of subjects.
#         D (int): Number of features.
#         T (int): Number of time points.
#         C (int): Number of categorical variables.
#     """
#     def __init__(self, J, D, T, C):
#         self.X = torch.randn(J * T, D)  # Synthetic features for demonstration
#         self.Y = torch.randn(J * T)  # Synthetic response variable, flattened
#         self.cat_vars = self.generate_cat_vars(J, C, T)
#         # self.subject_ids = self.generate_subject_ids(J, T)
#         self.binary_event = self.generate_binary_event(J, T)  # Generate binary event data
        
#     def generate_cat_vars(self, J, C, T):
#         cat_vars = torch.randint(0, 10, (J, C))
#         cat_vars_repeated = cat_vars.repeat_interleave(T, dim=0)  # Repeat each row T times
#         return cat_vars_repeated
    
#     # def generate_subject_ids(self, J, T):
#     #     subject_ids = torch.arange(J).unsqueeze(-1).repeat(1, T).flatten()
#     #     return subject_ids
    
#     def generate_binary_event(self, J, T):
#         # Generate a binary event for each time step of each subject
#         # This example randomly generates a binary event, adjust as needed
#         binary_event = torch.randint(0, 2, (J * T,))
#         return binary_event
    
#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         return self.X[idx], self.Y[idx], self.cat_vars[idx], self.binary_event[idx]