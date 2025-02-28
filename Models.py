# # # # # # # # # # # # # # # # # # # # # # # # 
#                                             #
#                                             #
#                                             #
#   This .py contains the Models to train     #
#     (MLP, Batch_MLP, CNN, CNN_SelfAtt)      #
#                                             #
#                                             #
#                                             #
# # # # # # # # # # # # # # # # # # # # # # # #
import torch.nn as nn
import torch
import math
class MLP_Model(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) model with dropout regularization.

    This model consists of multiple fully connected layers with `Tanh` activation functions
    and dropout layers to prevent overfitting.

    Parameters:
    -----------
    - input_dim (int): The number of input features.
    - output_dim (int): The number of output features (e.g., number of target variables).

    Architecture:
    -------------
    - Five hidden layers with the following dimensions: 512 → 256 → 128 → 64 → 32
    - Each hidden layer uses `Tanh` as the activation function.
    - A dropout layer with `p=0.3` is applied after each hidden layer to reduce overfitting.
    - The output layer maps the last hidden layer to `output_dim` dimensions.

    Methods:
    --------
    - forward(x): Defines the forward pass of the model.

    Returns:
    --------
    - torch.Tensor: The model output, representing predictions.
    """

    def __init__(self, input_dim, output_dim):
        super(MLP_Model, self).__init__()
        self.hidden1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.hidden2 = nn.Linear(512, 256)
        self.hidden3 = nn.Linear(256, 128)
        self.hidden4 = nn.Linear(128, 64)
        self.hidden5 = nn.Linear(64, 32)
        self.output = nn.Linear(32, output_dim)  # 25 sorties

        self.dropout = nn.Dropout(p=0.3)
        self.activation = nn.Tanh()

    def forward(self, x):
        """
        Forward pass through the MLP model.

        Parameters:
        -----------
        - x (torch.Tensor): The input tensor.

        Returns:
        --------
        - torch.Tensor: The model's predictions.
        """

        x = self.hidden1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.hidden2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.hidden3(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.hidden4(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.hidden5(x)
        x = self.activation(x)
        x = self.output(x)

        return x

"""#### 6 layers MLP with batchs"""

class MLP_Batch_Model(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) model with batch normalization.

    This model is similar to `MLP_Model`, but it includes batch normalization layers
    to stabilize training.

    Parameters:
    -----------
    - input_dim (int): The number of input features.
    - output_dim (int): The number of output features.

    Architecture:
    -------------
    - Six hidden layers with the following dimensions: 1024 → 512 → 256 → 256 → 128 → 64
    - Each hidden layer is followed by a batch normalization layer and `Tanh` activation.
    - A dropout layer (`p=0.3`) is applied after each hidden layer.
    - The output layer maps the last hidden layer to `output_dim` dimensions.

    Methods:
    --------
    - forward(x): Defines the forward pass of the model.

    Returns:
    --------
    - torch.Tensor: The model output, representing predictions.
    """

    def __init__(self, input_dim, output_dim):
        super(MLP_Batch_Model, self).__init__()
        self.hidden1 = nn.Linear(input_dim, 1024)
        self.batchnorm1 = nn.BatchNorm1d(1024)
        self.activation = nn.Tanh()

        self.hidden2 = nn.Linear(1024, 512)
        self.batchnorm2 = nn.BatchNorm1d(512)

        self.hidden3 = nn.Linear(512, 256)
        self.batchnorm3 = nn.BatchNorm1d(256)

        self.hidden4 = nn.Linear(256, 256)
        self.batchnorm4 = nn.BatchNorm1d(256)

        self.hidden5 = nn.Linear(256, 128)
        self.batchnorm5 = nn.BatchNorm1d(128)

        self.hidden6 = nn.Linear(128, 64)
        self.output = nn.Linear(64, output_dim)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        """
        Forward pass through the MLP model with batch normalization.

        Parameters:
        -----------
        - x (torch.Tensor): The input tensor.

        Returns:
        --------
        - torch.Tensor: The model's predictions.
        """

        x = self.hidden1(x)
        x = self.batchnorm1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.hidden2(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.hidden3(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.hidden4(x)
        x = self.batchnorm4(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.hidden5(x)
        x = self.batchnorm5(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.hidden6(x)
        x = self.activation(x)
        x = self.output(x)

        return x

"""CNN Model"""
class CNNModel(nn.Module):
    """
    A CNN model with batch normalization and average pooling.

    Architecture:
    -------------
    - Three convolutional layers with the following channels: 1 → 16 → 32 → 64
    - Each conv layer is followed by a batch normalization layer and `ReLU` activation.
    - Average pooling is applied after CNN
    - Two FC layers  with the following channels: 64 → 32 → 1
    

    Methods:
    --------
    - forward(x): Defines the forward pass of the model.

    Returns:
    --------
    - torch.Tensor: The model output, representing predictions.
    """
    def __init__(self,output_dim):
        super(CNNModel,self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.global_avg_pool(x).squeeze(-1)  # Pooling over time dimension
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


"""CNN/MLP with Self Attention layer"""

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SelfAttention(nn.Module):
    def __init__(self, dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)  # (batch, seq_len, dim)
        K = self.key(x)
        V = self.value(x)
        attn = self.softmax(Q @ K.transpose(-2, -1) / (x.shape[-1] ** 0.5))
        return attn @ V

class CNN_Attention(nn.Module):
    """
    A CNN/MLP combined model with batch normalization, self attention layer and positional encoding.

    Self Attention and positional encoding are added because CNN doesn't detect long-term dependencies (Thus self attention layer)
    and has no sense of order for time series (Thus positional encoding)

    Parameters:
    -----------
    - output_dim (int): The number of output features.
    - seq_length: chosen sequence length for training (336 time steps corresponds to one day, couldn't do more because CUDA goes out of memory )
    - d_model: Model dimension
    - hidden_dim: MLP's hidden layers dimension
    Architecture:
    -------------
    - Four convolutional layers with the following channels: 1 → 4 → 8 → 16 → 64
    - Each conv layer is followed by a batch normalization layer and `ReLU` activation.
    - Average pooling is applied after CNN
    - Positional encoding/Self-Attention
    - Four FC layers for the MLP with the following channels: 336*64 → 128 → 128 → output_dim
    - A dropout layer (`p=0.1`) is applied after each hidden layer of the MLP.
    - The output layer maps the last hidden layer to `output_dim` dimensions.

    Methods:
    --------
    - forward(x): Defines the forward pass of the model.

    Returns:
    --------
    - torch.Tensor: The model output, representing predictions.
    """
    def __init__(self, output_dim, seq_length=336, d_model=64, hidden_dim=128):
        super(CNN_Attention, self).__init__()

        # CNN layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(4)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(8)
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(16)
        self.conv4 = nn.Conv1d(in_channels=16, out_channels=d_model, kernel_size=3, padding=1)  # Match d_model
        self.bn4 = nn.BatchNorm1d(d_model)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(seq_length)  # Keep sequence length consistent

        # Self-Attention Layer and Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model=d_model)
        self.self_attention = SelfAttention(dim=d_model)

        # MLP layers
        self.fc1 = nn.Linear(seq_length * d_model, hidden_dim)  # Flattened size
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.global_avg_pool(x)
        x = x.permute(0, 2, 1)  # Reshape for Self-Attention (batch, seq_length, d_model)
        x = self.positional_encoding(x)
        x = self.self_attention(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.norm(self.fc1(x))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x
