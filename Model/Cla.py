import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(768,512),
            nn.ReLU(),
            nn.Linear(512,134),
            nn.Softmax()
        )

    def forward(self,x):
        return self.model(x)

