from argparse import ArgumentParser
from common import train_model

if __name__ == "__main__":
    parser = ArgumentParser(prog="model trainer", description="trains cartpole actor")
    parser.add_argument("experiment_basename", type=str)
    parser.add_argument("lr", type=float, default=1e-3)
    parser.add_argument("ew", type=float, default=0.0)
    args = parser.parse_args()

    max_avg_reward = 0
    avg_reward = 0
    for run_num in range(5):
        args.experiment_name = f"{args.experiment_basename}_{run_num+1}"
        avg_reward = train_model(args, avg_reward)
        max_avg_reward = max(max_avg_reward, avg_reward)
    print(max_avg_reward)
