import torch 
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from .networks import Actor, IQN
from .replay_buffer import ReplayBuffer, PrioritizedReplay
import numpy as np
import random
import copy
from .ICM import ICM, Inverse, Forward

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size,
                      action_size,
                      n_step,
                      per,
                      distributional,
                      noise_type,
                      random_seed,
                      hidden_size,
                      BUFFER_SIZE = int(1e6),  # replay buffer size
                      BATCH_SIZE = 128,        # minibatch size
                      GAMMA = 0.99,            # discount factor
                      TAU = 1e-3,              # for soft update of target parameters
                      LR_ACTOR = 1e-4,         # learning rate of the actor 
                      LR_CRITIC = 1e-4,        # learning rate of the critic
                      WEIGHT_DECAY = 0,#1e-2        # L2 weight decay
                      LEARN_EVERY = 1,
                      LEARN_NUMBER = 1,
                      EPSILON = 1.0,
                      EPSILON_DECAY = 1,
                      device = "cuda",
                      frames = 100000,
                      worker=1
                      ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.per = per
        self.n_step = n_step
        self.distributional = distributional
        self.GAMMA = GAMMA
        self.TAU = TAU
        self.LEARN_EVERY = LEARN_EVERY
        self.LEARN_NUMBER = LEARN_NUMBER
        self.EPSILON_DECAY = EPSILON_DECAY
        self.device = device
        self.seed = random.seed(random_seed)
        # distributional Values
        self.N = 32
        self.entropy_coeff = 0.001
        self.entropy_tau = 0.03
        self.lo = -1
        self.alpha = 0.9
        
        self.eta = torch.FloatTensor([.1]).to(device)

        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed, hidden_size=hidden_size).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed, hidden_size=hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = IQN(state_size, action_size, layer_size=hidden_size, device=device, seed=random_seed, dueling=None, N=self.N).to(device)
        self.critic_target = IQN(state_size, action_size, layer_size=hidden_size, device=device, seed=random_seed, dueling=None, N=self.N).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
            
        # Noise process
        self.noise_type = noise_type
        if noise_type == "ou":
            self.noise = OUNoise(action_size, random_seed)
            self.epsilon = EPSILON
        else:
            self.epsilon = 0.3
        # Replay memory
        if per:
            self.memory = PrioritizedReplay(BUFFER_SIZE, BATCH_SIZE, device=device, seed=random_seed, gamma=GAMMA, n_step=n_step, parallel_env=worker, beta_frames=frames)
        else:
            self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, n_step=n_step, parallel_env=worker, device=device, seed=random_seed, gamma=GAMMA)

        self.learn = self.learn_distribution
        
    def step(self, state, action, reward, next_state, done, timestamp, writer):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.BATCH_SIZE and timestamp % self.LEARN_EVERY == 0:
            for _ in range(self.LEARN_NUMBER):
                experiences = self.memory.sample()
                
                losses = self.learn(experiences, self.GAMMA)
            writer.add_scalar("Critic_loss", losses[0], timestamp)
            writer.add_scalar("Actor_loss", losses[1], timestamp)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)

        assert state.shape == (state.shape[0],self.state_size), "shape: {}".format(state.shape)
        self.actor_local.eval()
        with torch.no_grad():
                action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            if self.noise_type == "ou":
                action += self.noise.sample() * self.epsilon
            else:
                action += self.epsilon * np.random.normal(0, scale=1)
        return action #np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn_(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, idx, weights = experiences
        # ---------------------------- update critic ---------------------------- #

            # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            actions_next = self.actor_target(next_states.to(self.device))
            Q_targets_next = self.critic_target(next_states.to(self.device), actions_next.to(self.device))
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma**self.n_step * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        if self.per:
            td_error =  Q_targets - Q_expected
            critic_loss = (td_error.pow(2)*weights.to(self.device)).mean().to(self.device)
        else:
            critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)                     
        if self.per:
            self.memory.update_priorities(idx, np.clip(abs(td_error.data.cpu().numpy()),-1,1))
        # ----------------------- update epsilon and noise ----------------------- #
        
        self.epsilon *= self.EPSILON_DECAY
        
        if self.noise_type == "ou": self.noise.reset()
        return critic_loss.detach().cpu().numpy(), actor_loss.detach().cpu().numpy(), icm_loss

    
    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU*local_param.data + (1.0-self.TAU)*target_param.data)


    def learn_distribution(self, experiences, gamma):
            """Update policy and value parameters using given batch of experience tuples.
            Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
            where:
                actor_target(state) -> action
                critic_target(state, action) -> Q-value

            Params
            ======
                experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
                gamma (float): discount factor
            """
            states, actions, rewards, next_states, dones, idx, weights = experiences

            # ---------------------------- update critic ---------------------------- #
            # Get predicted next-state actions and Q values from target models

            # Get max predicted Q values (for next states) from target model
            with torch.no_grad():
                next_actions = self.actor_local(next_states)
                Q_targets_next, _ = self.critic_target(next_states, next_actions, self.N)
                Q_targets_next = Q_targets_next.transpose(1,2)
            # Compute Q targets for current states
            Q_targets = rewards.unsqueeze(-1) + (self.GAMMA**self.n_step * Q_targets_next.to(self.device) * (1. - dones.unsqueeze(-1)))

            # Get expected Q values from local model
            Q_expected, taus = self.critic_local(states, actions, self.N)
            assert Q_targets.shape == (self.BATCH_SIZE, 1, self.N)
            assert Q_expected.shape == (self.BATCH_SIZE, self.N, 1)
    
            # Quantile Huber loss
            td_error = Q_targets - Q_expected
            assert td_error.shape == (self.BATCH_SIZE, self.N, self.N), "wrong td error shape"
            huber_l = calculate_huber_loss(td_error, 1.0)
            quantil_l = abs(taus -(td_error.detach() < 0).float()) * huber_l / 1.0
            
            if self.per:
                critic_loss = (quantil_l.sum(dim=1).mean(dim=1, keepdim=True)*weights.to(self.device)).mean()
            else:
                critic_loss = quantil_l.sum(dim=1).mean(dim=1).mean()
            # Minimize the loss
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            clip_grad_norm_(self.critic_local.parameters(), 1)
            self.critic_optimizer.step()

            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actions_pred = self.actor_local(states)
            actor_loss = -self.critic_local.get_qvalues(states, actions_pred).mean()
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic_local, self.critic_target)
            self.soft_update(self.actor_local, self.actor_target)                     
            if self.per:
                self.memory.update_priorities(idx, np.clip(abs(td_error.sum(dim=1).mean(dim=1,keepdim=True).data.cpu().numpy()),-1,1))
            # ----------------------- update epsilon and noise ----------------------- #
            
            self.epsilon *= self.EPSILON_DECAY
            
            if self.noise_type == "ou": self.noise.reset()
            return critic_loss.detach().cpu().numpy(), actor_loss.detach().cpu().numpy()

        

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


def calculate_huber_loss(td_errors, k=1.0):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    assert loss.shape == (td_errors.shape[0], 32, 32), "huber loss has wrong shape"
    return loss