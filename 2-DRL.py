import torch
from torch import nn

from net.renet18 import RestNet18


class DQN_net(nn.Module):
    def __init__(self, in_ch, num_class, apply_mean=True):
        super(DQN_net, self).__init__()
        self.apply_mean = apply_mean
        self.resnet = RestNet18(in_ch)

        self.action_value = nn.Sequential(
            nn.Linear(in_features=512 + num_class * 10, out_features=256),
            nn.ReLU(),
            nn.Linear(256, num_class)
        )

    def forward(self, x, actions_record):
        x = self.resnet(x)
        batch = x.shape[0]
        x = x.view(batch, -1)
        x = torch.cat((actions_record, x), dim=1)

        x = self.action_value(x)
        return x

