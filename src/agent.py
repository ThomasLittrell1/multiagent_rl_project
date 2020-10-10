import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.optim as optim

from src.model import PolicyNetwork, QNetwork

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 10  # How often to learn
START_NOISE_SCALE = 0.2  # Starting normal std
NOISE_DECAY = 0.999  # How much to decay the noise every time step

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# noinspection PyUnresolvedReferences
class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.q_optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Policy Network
        self.policy_network_local = PolicyNetwork(state_size, action_size, seed).to(
            device
        )
        self.policy_network_target = PolicyNetwork(state_size, action_size, seed).to(
            device
        )
        self.policy_optimizer = optim.Adam(
            self.policy_network_local.parameters(), lr=LR
        )

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # Action selection
        self.noise_scale = START_NOISE_SCALE

    def step(self, states, actions, rewards, next_states, dones):

        # With multiple arms we need to save each experience separately in the replay
        # buffer
        for state, action, reward, next_state, done in zip(
            states, actions, rewards, next_states, dones
        ):
            self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                for _ in range(20):
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)

    def act(self, state):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().to(device)
        self.qnetwork_local.eval()
        self.policy_network_local.eval()
        with torch.no_grad():
            action = self.policy_network_local(state).cpu().data.numpy()
        self.qnetwork_local.train()
        self.policy_network_local.train()

        # Add noise to the policy that decays to 0 over time to encourage exploration
        noise = np.random.normal(
            loc=0, scale=self.noise_scale, size=(1, self.action_size)
        )
        action += noise
        self.noise_scale *= NOISE_DECAY

        return np.clip(action, a_min=-1, a_max=1)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Update the Q-network
        argmax_a_next = self.policy_network_target.forward(next_states)
        best_next_Q = self.qnetwork_target.forward(next_states, argmax_a_next)
        Q_target = rewards + gamma * best_next_Q * (1 - dones)

        Q_current = self.qnetwork_local.forward(states, actions)

        self.q_optimizer.zero_grad()
        criterion = torch.nn.MSELoss()
        loss = criterion(Q_current, Q_target.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1)
        self.q_optimizer.step()

        # Update the policy network
        argmax_a = self.policy_network_local.forward(states)
        action_values = self.qnetwork_local.forward(states, argmax_a)

        self.policy_optimizer.zero_grad()
        loss = -action_values.mean()  # Negative b/c we're doing gradient ascent
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network_local.parameters(), 1)
        self.policy_optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        self.soft_update(self.policy_network_local, self.policy_network_target, TAU)

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )


# noinspection PyUnresolvedReferences
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = (
            torch.from_numpy(np.vstack([e.state for e in experiences if e is not None]))
            .float()
            .to(device)
        )
        actions = (
            torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
            )
            .long()
            .to(device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        next_states = (
            torch.from_numpy(
                np.vstack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(device)
        )

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
