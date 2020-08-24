import torch
import torch.nn as nn
from pruning.layers import MaskedConv2d
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = MaskedConv2d(1, 6, kernel_size=3)
        self.conv2 = MaskedConv2d(6, 16, kernel_size=3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def set_masks(self, masks):
        self.conv1.set_mask(torch.from_numpy(masks[0]))
        # self.conv2.set_mask(torch.from_numpy(masks[1]))
