## Proximal Policy Optimization with Generalized Advantage Estimation

[By Patrick Coady](https://github.com/pat-coady/trpo/tree/aigym_evaluation)

### Summary (need extend)

## Dependencies

* Python 3.5
* The Usual Suspects: NumPy, matplotlib, scipy
* TensorFlow
* gym - [installation instructions](https://gym.openai.com/docs)
* [MuJoCo](http://www.mujoco.org/)

### Results can be reproduced as follows:

```
./train.py Humanoid-v1 -n 200000
./train.py HumanoidStandup-v1 -n 200000 -b 5
```

### References

1. [Trust Region Policy Optimization](https://arxiv.org/pdf/1502.05477.pdf) (Schulman et al., 2016)
2. [Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/pdf/1707.02286.pdf) (Heess et al., 2017)
3. [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/pdf/1506.02438.pdf) (Schulman et al., 2016)
4. [GitHub Repository with several helpful implementation ideas](https://github.com/joschu/modular_rl) (Schulman)
