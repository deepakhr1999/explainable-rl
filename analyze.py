from argparse import ArgumentParser
import torch
import numpy as np
from tqdm import tqdm
import pickle
from common import test_model, ActorCritic

if __name__ == "__main__":
    parser = ArgumentParser(prog="model trainer", description="trains cartpole actor")
    parser.add_argument("file_path", type=str)
    parser.add_argument("--experiment_name", type=str, default="default")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ew", type=float, default=0.0)
    args = parser.parse_args()
    path = "model_saves/2022-11-20/grad_reg_neg_1e-1.pth"
    path = "model_saves/2022-11-23/default.pth"
    model = torch.load(args.file_path)
    rewards, grads = zip(*[test_model(args, model) for _ in tqdm(range(500))])

    ### show grads
    grads = np.concatenate(grads, axis=0)
    with open(f"{args.experiment_name}.pkl", "wb") as file:
        pickle.dump(grads, file)
    print(grads.shape)
    print(np.mean(rewards))
