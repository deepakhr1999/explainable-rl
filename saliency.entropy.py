import gym
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.autograd import grad
from torch.utils.tensorboard import SummaryWriter
from common import SubprocVecEnv

num_envs = 16
env_name = "CartPole-v1"


def make_env():
    def _thunk():
        warnings.filterwarnings("ignore")
        env = gym.make(env_name, new_step_api=False)
        return env

    return _thunk


envs = [make_env() for i in range(num_envs)]
envs = SubprocVecEnv(envs)

env = gym.make(env_name, new_step_api=False)


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        print("Network configuration:")
        print("input_dim:", num_inputs)
        print("hidden_dim:", hidden_size)
        print("output_dim:", num_outputs)

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist = Categorical(probs)
        return dist, value, probs


class Hparams:
    num_inputs = envs.observation_space.shape[0]
    num_outputs = envs.action_space.n
    hidden_size = 64
    lr = 1e-3
    num_steps = 10
    max_frames = 20000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_env(model, transform=None):
    if transform is None:
        state = env.reset()
    else:
        state = transform(env.reset())

    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(Hparams.device)
        dist, _, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        if transform is None:
            state = next_state
        else:
            state = transform(next_state)
        total_reward += reward

    return total_reward


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


class LossObject:
    def __init__(self, experiment):
        self.reset()
        self.frame_idx = 0
        self.writer = SummaryWriter(f"logs/{experiment}/")

    def reset(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.masks = []
        self.grad_reg = []
        self.entropy = 0

    def update(self, log_prob, value, reward, done, grad_reg_entropy, entropy):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(Hparams.device))
        self.masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(Hparams.device))
        self.grad_reg.append(grad_reg_entropy)
        self.entropy += entropy
        self.frame_idx += 1

    def compute_loss(self, next_value, gamma=0.99):
        R = next_value
        returns = []
        for step in reversed(range(len(self.rewards))):
            R = self.rewards[step] + gamma * R * self.masks[step]
            returns.insert(0, R)

        grad_reg = torch.cat(self.grad_reg)
        log_probs = torch.cat(self.log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(self.values)

        advantage = returns - values
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        grad_reg = grad_reg.mean()
        loss_without_critic = actor_loss - 0.001 * self.entropy
        loss = loss_without_critic + 0.5 * critic_loss + grad_reg

        # log actor loss, entropy and grad_reg
        self.writer.add_scalar("actor_loss", actor_loss.item(), self.frame_idx)
        self.writer.add_scalar("entropy", self.entropy.item(), self.frame_idx)
        self.writer.add_scalar("grad_reg", grad_reg.item(), self.frame_idx)
        self.writer.add_scalar(
            "loss_without_critic", loss_without_critic.item(), self.frame_idx
        )

        return loss


def model_env_forward(model, state, envs):
    state = torch.FloatTensor(state).to(Hparams.device)
    state.requires_grad = True
    dist, value, probs = model(state)

    action = dist.sample()
    entropy = dist.entropy().mean()
    next_state, reward, done, _ = envs.step(action.cpu().numpy())

    log_prob = dist.log_prob(action)

    saliency = grad(probs[:, 0].sum(), state, retain_graph=True, create_graph=True)[0]
    saliency = torch.softmax(saliency**2, axis=1)
    grad_reg_entropy = (
        -(saliency * torch.log(saliency)).sum(axis=1).mean(axis=0, keepdim=True)
    )

    return next_state, log_prob, value, reward, done, grad_reg_entropy, entropy


def train_model(experiment):
    transform = None

    # init model and optimizer
    model = ActorCritic(
        num_inputs=Hparams.num_inputs,
        num_outputs=Hparams.num_outputs,
        hidden_size=Hparams.hidden_size,
    ).to(Hparams.device)

    optimizer = optim.Adam(lr=Hparams.lr, params=model.parameters())

    # init env
    max_avg_reward = 0
    state = envs.reset()
    loss_obj = LossObject(experiment)

    while loss_obj.frame_idx < Hparams.max_frames:
        loss_obj.reset()

        for _ in range(Hparams.num_steps):
            state, *update_args = model_env_forward(model, state, envs)
            loss_obj.update(*update_args)

            if loss_obj.frame_idx % 500 == 0:
                # average reward over 500 episodes
                avg_reward = np.mean([test_env(model) for _ in range(500)])
                max_avg_reward = max(max_avg_reward, avg_reward)
                print(f"Frame={loss_obj.frame_idx} => avg_reward={avg_reward:.4f}")
                loss_obj.writer.add_scalar("avg_reward", avg_reward, loss_obj.frame_idx)

        state = torch.FloatTensor(state).to(Hparams.device)
        next_value = model(state)[1]

        loss = loss_obj.compute_loss(next_value)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_reward = np.mean([test_env(model) for _ in range(500)])
    max_avg_reward = max(max_avg_reward, avg_reward)
    return max_avg_reward


avg_reward = train_model("cartpole_v1_grad_reg")
print(avg_reward)
