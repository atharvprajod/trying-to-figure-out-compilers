# Vibe code mess rn: basically RL to help compiler, idk wanna try and pan this out by throwing myself in the deep end

This project explores the use of reinforcement learning to optimize operator fusion scheduling within the Apache TVM compiler, targeting performance improvements on ARM CPUs (specifically Apple M1 chips).

## Project Overview

Deep learning compilers like Apache TVM apply **operator fusion** to combine multiple graph operations into a single kernel, eliminating intermediate memory writes and improving data locality. TVM's current fusion strategy uses generic, rule-based heuristics which are hardware-agnostic. This project aims to improve operator fusion in TVM for ARM CPUs by making fusion decisions **adaptive and learning-based**.

## Project Structure

```
.
├── src/
│   └── rl_fusion/
│       ├── environment/    # RL environment that interfaces with TVM
│       ├── models/         # RL models (PPO, DQN) and GNN implementations
│       └── utils/          # Utility functions
├── tests/                  # Test code
├── scripts/                # Training and evaluation scripts
├── results/                # Performance results and visualizations
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Setup

1. Install TVM from source following the [official instructions](https://tvm.apache.org/docs/install/from_source.html)
2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

[To be added as the implementation progresses]

## License

[Add appropriate license] 
