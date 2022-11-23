# explainable-rl

## Train
### Args
- experiment_name
- learning rate
- weight for gradient entropy
### Command
```sh
python train.py grad_reg_neg_1e-1 5e-3 " -1e-1"
python train.py default 3e-3 0              
tensorboard --logdir=logs/ --bind_all
```
