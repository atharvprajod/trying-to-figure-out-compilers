# Contributing to RL-Fusion

Thank you for considering contributing to RL-Fusion! This document outlines the process for contributing to this project.

## Development Setup

1. **Fork and clone the repository**:
    ```
    git clone https://github.com/your-username/rl-fusion.git
    cd rl-fusion
    ```

2. **Create a virtual environment**:
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies in development mode**:
    ```
    pip install -e .
    ```

4. **Install TVM from source** (following the [official instructions](https://tvm.apache.org/docs/install/from_source.html)):
    ```
    # Clone TVM repository
    git clone --recursive https://github.com/apache/tvm.git
    cd tvm
    
    # Build TVM
    mkdir build
    cp cmake/config.cmake build
    cd build
    cmake ..
    make -j8
    
    # Add TVM to PYTHONPATH
    export TVM_HOME=/path/to/tvm
    export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
    ```

## Project Structure

The project has the following structure:

```
.
├── src/
│   └── rl_fusion/
│       ├── environment/    # RL environment that interfaces with TVM
│       ├── models/         # RL models (PPO, DQN) and GNN implementations
│       └── utils/          # Utility functions
├── scripts/                # Training and evaluation scripts
├── tests/                  # Test code
├── results/                # Performance results and visualizations
├── requirements.txt        # Python dependencies
└── README.md               # Project overview
```

## Development Workflow

1. **Create a new branch**:
    ```
    git checkout -b feature/your-feature-name
    ```

2. **Make your changes**:
   - Follow the coding style of the project.
   - Add appropriate tests for your changes.

3. **Run tests**:
    ```
    pytest
    ```

4. **Submit a pull request**:
   - Push your changes to your fork.
   - Create a pull request against the main repository.
   - Describe your changes in detail.

## Running Experiments

To train an RL agent for operator fusion in TVM:

```
python scripts/train_fusion_rl.py --algorithm ppo --use_cost_model --num_iterations 100
```

To evaluate a trained agent:

```
python scripts/evaluate_fusion_rl.py --model_path results/ppo_YYYYMMDD_HHMMSS/best_policy.pth --algorithm ppo --num_episodes 10 --compare_default
```

To run the TVM integration demo:

```
python scripts/tvm_integration_demo.py --model_path results/ppo_YYYYMMDD_HHMMSS/best_policy.pth --algorithm ppo --visualize
```

## Adding New Features

### Adding a new RL algorithm

1. Create a new implementation in `src/rl_fusion/models/`
2. Add the corresponding training loop in `src/rl_fusion/utils/training.py`
3. Update the scripts to support the new algorithm

### Adding support for a new model architecture

1. Update the environment in `src/rl_fusion/environment/fusion_env.py` to handle the new architecture
2. Test and benchmark with the existing scripts

## Reporting Issues

If you encounter any issues or have suggestions for improvements, please [open an issue](https://github.com/your-username/rl-fusion/issues/new) with the following information:

- A clear description of the issue
- Steps to reproduce the problem
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, TVM version, etc.)

## License

By contributing to this project, you agree that your contributions will be licensed under the project's license. 