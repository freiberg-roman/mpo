# MAP Policy Optimisation (MPO)

Inspired by implementation of [daisatojp](https://github.com/daisatojp/mpo).
Basic framework and SAC implementation are mostly taken from [OpenAI SpinningUp](https://github.com/openai/spinningup)

References:

MPO: [link](https://arxiv.org/abs/1806.06920)
SAC: [link](https://arxiv.org/pdf/1801.01290v1.pdf) 
RERPI: [link](https://arxiv.org/abs/1812.02256)

# Installation

Python 3.9+ and working MuJoCo installation is required.
Optional: Create conda environment with 

    conda create -n myenv python=3.9

Standard installation with pip

    git clone https://github.com/freiberg-roman/mpo.git
    cd mpo
    pip install -e ".[dev]"

Test installation by running
    
    python -m mpo.examples.main algorithm=mpo q_learning=retrace overrides=pendulum
    
