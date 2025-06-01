import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Define the model classes
class GatedLinear(nn.Module):
    def __init__(self, in_features, out_features, gate_hidden=16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_net = nn.Sequential(
            nn.Linear(in_features, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, out_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        gates = self.gate_net(x)
        out = F.linear(x, self.weight, self.bias)
        gated_out = gates * out
        return gated_out

class MemoryModule(nn.Module):
    def __init__(self, input_size, memory_size):
        super().__init__()
        self.memory_size = memory_size
        self.write_net = nn.Sequential(
            nn.Linear(input_size, memory_size),
            nn.Tanh()
        )
        self.read_gate = nn.Sequential(
            nn.Linear(input_size, memory_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        memory = self.write_net(x)
        read_weights = self.read_gate(x)
        read_memory = memory * read_weights
        return read_memory

class GatedMemoryNet(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, output_size):
        super().__init__()
        self.memory = MemoryModule(input_size, memory_size)
        self.gated1 = GatedLinear(input_size + memory_size, hidden_size)
        self.gated2 = GatedLinear(hidden_size, output_size)

    def forward(self, x):
        mem = self.memory(x)
        x_cat = torch.cat([x, mem], dim=-1)
        h = F.relu(self.gated1(x_cat))
        out = self.gated2(h)
        return out

# Generate synthetic data
torch.manual_seed(0)
x_data = torch.randn(200, 10)
true_weights = torch.randn(10, 1)
y_data = x_data @ true_weights + 0.1 * torch.randn(200, 1)

# Initialize model, loss, optimizer
model = GatedMemoryNet(input_size=10, hidden_size=32, memory_size=8, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
losses = []
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    output = model(x_data)
    loss = criterion(output, y_data)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# Plot training loss
plt.plot(losses)
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()
