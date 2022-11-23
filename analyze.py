# #!/home/ubuntu/anaconda3/envs/research/bin/python
# import gym
# import os
# import warnings
# import datetime as dt

# warnings.filterwarnings("ignore")
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.distributions import Categorical
# from torch.autograd import grad
# from torch.utils.tensorboard import SummaryWriter
# from common import SubprocVecEnv
# from argparse import ArgumentParser
# from tqdm import tqdm

# num_envs = 16
# env_name = "CartPole-v1"


# def make_env():
#     def _thunk():
#         warnings.filterwarnings("ignore")
#         env = gym.make(env_name, new_step_api=False)
#         return env

#     return _thunk


# envs = [make_env() for i in range(num_envs)]
# envs = SubprocVecEnv(envs)

# env = gym.make(env_name, new_step_api=False)


# class ActorCritic(nn.Module):
#     def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
#         super(ActorCritic, self).__init__()

#         self.critic = nn.Sequential(
#             nn.Linear(num_inputs, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
#         )

#         self.actor = nn.Sequential(
#             nn.Linear(num_inputs, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, num_outputs),
#             nn.Softmax(dim=1),
#         )

#     def forward(self, x):
#         value = self.critic(x)
#         probs = self.actor(x)
#         dist = Categorical(probs)
#         return dist, value, probs


# class Hparams:
#     num_inputs = envs.observation_space.shape[0]
#     num_outputs = envs.action_space.n
#     hidden_size = 32
#     lr = 1e-3
#     num_steps = 10
#     max_frames = 20000
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(
#         f"Network configuration: input_dim={num_inputs} hidden_units={hidden_size} output_dim={num_outputs}"
#     )


# def test_env(model, transform=None):
#     if transform is None:
#         state = env.reset()
#     else:
#         state = transform(env.reset())

#     done = False
#     total_reward = 0
#     while not done:
#         state = torch.FloatTensor(state).unsqueeze(0).to(Hparams.device)
#         dist, _, _ = model(state)
#         next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
#         if transform is None:
#             state = next_state
#         else:
#             state = transform(next_state)
#         total_reward += reward

#     return total_reward


# def compute_returns(next_value, rewards, masks, gamma=0.99):
#     R = next_value
#     returns = []
#     for step in reversed(range(len(rewards))):
#         R = rewards[step] + gamma * R * masks[step]
#         returns.insert(0, R)
#     return returns


# class LossObject:
#     def __init__(self, experiment_name, entropy_weight, mode="train"):
#         self.reset()
#         self.frame_idx = 0
#         date_part = dt.datetime.today().strftime("%Y-%m-%d")
#         self.writer = SummaryWriter(f"logs/{date_part}_{experiment_name}/")
#         self.entropy_weight = entropy_weight
#         self.mode = mode
#         if mode == "train":
#             self.progress_bar = tqdm(
#                 total=Hparams.max_frames, desc=experiment_name, unit="Frames"
#             )

#     def reset(self):
#         self.log_probs = []
#         self.values = []
#         self.rewards = []
#         self.masks = []
#         self.grad_reg = []
#         self.entropy = 0

#     def update(self, log_prob, value, reward, done, grad_reg_entropy, entropy):
#         if self.mode == "test":
#             # print(type(log_prob), "log_prob")
#             # print(type(value), "value")
#             # print(type(reward), "reward")
#             # print(type(done), "done")
#             # print(type(grad_reg_entropy), "grad_reg_entropy")
#             # print(type(entropy), "entropy")

#             reward = np.array([reward])
#             done = np.array([done])

#         self.log_probs.append(log_prob)
#         self.values.append(value)
#         self.rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(Hparams.device))
#         self.masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(Hparams.device))
#         self.grad_reg.append(grad_reg_entropy)
#         self.entropy += entropy
#         self.frame_idx += 1

#     def compute_loss(self, next_value, gamma=0.99):
#         R = next_value
#         returns = []
#         for step in reversed(range(len(self.rewards))):
#             R = self.rewards[step] + gamma * R * self.masks[step]
#             returns.insert(0, R)

#         grad_reg = torch.cat(self.grad_reg)
#         log_probs = torch.cat(self.log_probs)
#         returns = torch.cat(returns).detach()
#         values = torch.cat(self.values)

#         advantage = returns - values
#         actor_loss = -(log_probs * advantage.detach()).mean()
#         critic_loss = advantage.pow(2).mean()
#         grad_reg_mean = grad_reg.mean()
#         loss_without_critic = actor_loss + self.entropy_weight * self.entropy
#         loss = loss_without_critic + 0.5 * critic_loss + grad_reg_mean

#         # log actor loss, entropy and grad_reg
#         if self.mode == "train":
#             self.writer.add_scalar("actor_loss", actor_loss.item(), self.frame_idx)
#             self.writer.add_scalar("entropy", self.entropy.item(), self.frame_idx)
#             self.writer.add_scalar("grad_reg", grad_reg_mean.item(), self.frame_idx)
#             self.writer.add_scalar(
#                 "loss_without_critic", loss_without_critic.item(), self.frame_idx
#             )
#             return loss
#         return sum(self.rewards).sum().numpy().item(), grad_reg.detach().numpy()


# def model_env_forward(model, state, envs, mode="train"):
#     state = torch.FloatTensor(state).to(Hparams.device)
#     state.requires_grad = True
#     dist, value, probs = model(state)

#     action = dist.sample()
#     entropy = dist.entropy().mean()
#     step_input = action.cpu().numpy()
#     if mode == "test":
#         step_input = step_input[0]

#     next_state, reward, done, _ = envs.step(step_input)

#     log_prob = dist.log_prob(action)

#     saliency = grad(probs[:, 0].sum(), state, retain_graph=True, create_graph=True)[0]
#     saliency = torch.softmax(1.0 * saliency**2, axis=1)
#     if mode == "train":
#         grad_reg_entropy = (
#             -(saliency * torch.log(saliency)).sum(axis=1).mean(axis=0, keepdim=True)
#         )
#     else:
#         grad_reg_entropy = saliency

#     return next_state, log_prob, value, reward, done, grad_reg_entropy, entropy


# def model_ckpt(model, max_avg_reward, path, min_performance_to_save):
#     avg_reward = np.mean([test_env(model) for _ in range(500)])
#     if avg_reward > max_avg_reward:
#         # update max value
#         max_avg_reward = avg_reward
#         # save model
#         if max_avg_reward > min_performance_to_save:
#             torch.save(model, path)
#     return max_avg_reward, avg_reward


# def train_model(args, min_performance_to_save):
#     transform = None
#     date_part = dt.datetime.today().strftime("%Y-%m-%d")
#     os.makedirs(f"model_saves/{date_part}/", exist_ok=True)
#     path = f"model_saves/{date_part}/{args.experiment_basename}.pth"

#     # init model and optimizer
#     model = ActorCritic(
#         num_inputs=Hparams.num_inputs,
#         num_outputs=Hparams.num_outputs,
#         hidden_size=Hparams.hidden_size,
#     ).to(Hparams.device)

#     optimizer = optim.Adam(lr=args.lr, params=model.parameters())

#     # init env
#     max_avg_reward = 0
#     state = envs.reset()
#     loss_obj = LossObject(args.experiment_name, args.ew)

#     while loss_obj.frame_idx < Hparams.max_frames:
#         loss_obj.reset()

#         for _ in range(Hparams.num_steps):
#             state, *update_args = model_env_forward(model, state, envs)
#             loss_obj.update(*update_args)

#             if loss_obj.frame_idx % 500 == 0:
#                 # average reward over 500 episodes
#                 max_avg_reward, avg_reward = model_ckpt(
#                     model, max_avg_reward, path, min_performance_to_save
#                 )
#                 loss_obj.progress_bar.set_postfix(
#                     avg_reward=avg_reward, max_avg_reward=max_avg_reward
#                 )
#                 loss_obj.progress_bar.update(500)
#                 # print(f"Frame={loss_obj.frame_idx} => avg_reward={avg_reward:.4f}")
#                 loss_obj.writer.add_scalar("avg_reward", avg_reward, loss_obj.frame_idx)

#         state = torch.FloatTensor(state).to(Hparams.device)
#         next_value = model(state)[1]

#         loss = loss_obj.compute_loss(next_value)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     loss_obj.progress_bar.close()

#     # post update, save model weights is performance improves
#     max_avg_reward, avg_reward = model_ckpt(
#         model, max_avg_reward, path, min_performance_to_save
#     )

#     return max_avg_reward


# def test_model(args, model):
#     # init env
#     done = False
#     total_reward = 0
#     loss_obj = LossObject(args.experiment_name, args.ew, "test")
#     # state = np.expand_dims(env.reset(), 0)
#     state = env.reset()
#     while not done:
#         state = np.expand_dims(state, 0)
#         state, *update_args = model_env_forward(model, state, env, "test")
#         done = update_args[3]
#         loss_obj.update(*update_args)
#     total_reward, grad_reg = loss_obj.compute_loss(0)
#     # after all states are done
#     return total_reward, grad_reg
from argparse import ArgumentParser
import torch
import numpy as np
from tqdm import tqdm
from common import test_model, ActorCritic

if __name__ == "__main__":
    parser = ArgumentParser(prog="model trainer", description="trains cartpole actor")
    parser.add_argument("experiment_basename", type=str)
    parser.add_argument("lr", type=float, default=1e-3)
    parser.add_argument("ew", type=float, default=0.0)
    args = parser.parse_args()
    args.experiment_name = args.experiment_basename
    path = "model_saves/2022-11-20/grad_reg_neg_1e-1.pth"
    path = "model_saves/2022-11-23/default.pth"
    model = torch.load(path)
    rewards, grads = zip(*[test_model(args, model) for _ in tqdm(range(500))])
    print(np.concatenate(grads, axis=0).shape)
    print(np.mean(rewards))
