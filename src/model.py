import torch
import torch.nn as nn
import torch.nn.functional as F


# noinspection PyUnresolvedReferences
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        """With DQN, the Q-network took in a state and output a value for each
        action. This was a convenient way to handle discrete actions, but with
        continuous actions we need to take in both the state and the action and
        output a single number. Thus the input dimension for this network is the
        state size + the action size and the output is one single node
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(in_features=state_size, out_features=100)
        self.fc2 = nn.Linear(in_features=100 + action_size, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=1)

    def forward(self, state, action):
        """Build a network that maps state, actions -> values."""
        result = self.fc1(state)
        result = F.relu(result)
        result = torch.cat((result, action.type(torch.FloatTensor)), dim=1)
        result = self.fc2(result)
        result = F.relu(result)
        result = self.fc3(result)
        return result


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super().__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(in_features=state_size, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=action_size)

    def forward(self, state):
        result = F.relu(self.fc1(state))
        result = F.relu(self.fc2(result))
        return F.tanh(self.fc3(result))  # tanh b/c actions are in [-1, 1]
