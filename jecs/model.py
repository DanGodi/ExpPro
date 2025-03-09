from torchsummary import summary
import torch.nn as nn

class ShallowMLP(nn.Module):
  def __init__(self, input_dim=4, output_dim=1, first_hidden_dim=128,second_hidden_dim=128,third_hidden_dim=128):
    super(ShallowMLP, self).__init__()

    self.layer1 = nn.Linear(input_dim, first_hidden_dim) 
    self.layer2 = nn.Linear(first_hidden_dim, first_hidden_dim)
    self.layer3 = nn.Linear(first_hidden_dim, second_hidden_dim)

    self.layer4 = nn.Linear(second_hidden_dim, third_hidden_dim) 
    self.layer5 = nn.Linear(third_hidden_dim, output_dim)

    self.relu = nn.ReLU() 

  def forward(self, x):
    x = self.layer1(x)
    x = self.relu(x) 
    x = self.layer2(x)
    x = self.relu(x)
    x= self.layer3(x)
    x = self.relu(x)
    x = self.layer4(x)
    x = self.relu(x)
    out = self.layer5(x)
    
    return out
  
def initialize_model():
    model = ShallowMLP()
    summary(model, (4,))

    return model

