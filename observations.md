# Observations
## Default setup
Params
- experiment basename = default
- learning rate = 1e-3
- entropy weight = zero

Notes
1. Model converges to decent performance after 12k frames.
2. There is a lot of variation in peak performance of model between runs. It is necessary to run the training script multiple times to achieve good performance.
3. However, within a run, at the point where model reaches peak performance, there is sharp dip in the entropy value. This indicates the model converges to a simple hypothesis (gives weightage only to few features in data)
