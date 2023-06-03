# SlowFastSeparation

A python  implementation of the KDD'23 paper: "Learning Slow and Fast System Dynamics via Automatic Separation of Time Scales".

## Requirements
- Python 3.10
- PyTorch==1.12
- scikit-learn==1.1.2
- Numpy
- Scipy
- Matplotlib
- tqdm
- scikit-dimension
- torchdiffeq
- torchsummary

## Usage

### Our Model

**Phase1:** Selecting the appropriate time scale $\tau_s$ and slow dimension *slow_dim* by ID-driven method.

```shell
# switch --phase to 'TimeSelection'
./OURS.sh
```

**Phase2:** Separating the fast and slow components and learning the dynamics.

```shell
# switch --phase to 'LearnDynamics'
# choose appropriate params(--tau_s, --slow_dim and --koopman_dim) by the Phase1
./OURS.sh
```

### Baseline

Train and test models in 1S2F and 2S2F system:

```shell
./LSTM.sh # for LSTM
./TCN.sh # for TCN
./NeuralODE # for Neural ODE
```



We recommend turning on the **--parallel** option to enable parallel execution of programs with different random seeds to improve test efficiency. Please be careful to choose the suitable number of random seeds  **--seed_num** according to your computational and cache resources. The result of the experiment should be an average of multiple random seeds.
