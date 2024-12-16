import torch
import torch.nn as nn


class TrainedDeeponet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t, x_t):
        return t * x_t


class RKnet(nn.Module):
    def __init__(self, m=4, h=0.01):
        super().__init__()
        self.lbd = nn.Parameter(torch.ones(m))
        self.alpha = nn.Parameter(torch.ones(m - 1))
        self.Beta = nn.Parameter(torch.ones([m - 1, m - 1]))

        self.trained_deeponet = TrainedDeeponet()
        self.m = m
        self.h = h

    def forward(self, x):
        K_list = []
        K_1 = self.h * self.trained_deeponet(x[:, 0:1], x[:, 1:2])
        K_list.append(K_1)
        for i in range(self.m - 1):
            t_temp = x[:, 0:1] + self.alpha[i] * self.h
            x_temp = x[:, 1:2] + sum(self.Beta[i][j] * K for j, K in enumerate(K_list))
            K_temp = self.h * self.trained_deeponet(t_temp, x_temp)
            K_list.append(K_temp)
        output = x[:, 1:2] + sum(self.lbd[i] * K for i, K in enumerate(K_list))
        return output


if __name__ == '__main__':
    mdoel = RKnet()
    input = torch.tensor([[1, 1], [1, 2]])
    output = mdoel(input)
    pass
