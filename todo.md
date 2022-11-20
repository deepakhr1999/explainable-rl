# TODO
## Idea
- multiple models that get good accuracy
- but we need the model with the least weights
- get linearly dependent data by transforming. fetch y by some model
- train some model directly and look at the data gradients (average them)
- train model with saliency and look at the data gradients (average them)
- See there is a difference in grads but not the converged loss

---

## Qualitative evaluation of model
For every experiment,
1. Load weights of each model with best performance over all runs
2. Rollout over 500 episodes, compute datagradients for each frame
3. Plots (each plot accompanied by avg and std values)
   1. Plot distribution of each gradient component over all frames
   2. Plot distribution entropy value of each gradient vector over all frames.
