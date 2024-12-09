## S-PPO (Smoothed - Proximal Policy Optimization)
<p align="center">
  <img src="./fig/SPPO_training.png?raw=true" width="80%" height="80%" />
</p>

### Links
https://github.com/Trustworthy-ML-Lab/Robust_HighUtil_Smoothed_DRL
https://arxiv.org/pdf/2406.18062

### Describe (need extend)
### Setup (need extend)
```
cd SPPO
git clone https://github.com/KaidiXu/auto_LiRPA
cd auto_LiRPA
git checkout 389dc72fcff606944dca0504cc77f52fef024c4e
python setup.py install
cd ..
pip install -r requirements.txt
```
Then, follow the instructions [here](https://github.com/openai/mujoco-py#install-mujoco) to install mujoco
### Evaluate (need extend)
### Training
To train S-PPO (Vanilla) in the Walker environment, run
```
python run.py --config-path config_walker_sppo.json
```
To train other agents, change the config name. For example, `config_walker_sppo_sgld.json` will train S-PPO (SGLD).
The implementation of S-PPO (WocaR) is in the `src_wocar/` folder, and S-PPO (S-PA-ATLA) is in the `src_pa_atla/` folder.


## TRPO (Trust Region Policy Optimization with Generalized Advantage Estimation)

## Proximal Policy Optimization with Generalized Advantage Estimation


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
[By Patrick Coady](https://github.com/pat-coady/trpo/tree/aigym_evaluation)










train_deterministic_hopper_v4 -> train_deterministic