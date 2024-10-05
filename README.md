# Llama3.2-MultiNode-Finetuning

This repository contains the implementation of **multi-node fine-tuning** for LLaMA 3.2, a state-of-the-art large language model. The project enables efficient fine-tuning across multiple nodes, taking advantage of distributed systems to optimize the training of LLaMA models on large datasets.

## Features

- **Multi-node support**: Fine-tuning LLaMA 3.2 across multiple nodes to reduce training time.
- **Scalable and efficient**: Designed to scale with large clusters, making use of advanced techniques like gradient accumulation and optimized communication strategies.
- **Checkpointing**: Supports model checkpointing to ensure that long training processes can be paused and resumed effectively.
- **Customizable hyperparameters**: Easily tune parameters such as learning rate, batch size, and others through a configuration file.
- **Distributed training**: Implements distributed data parallelism to handle massive datasets while minimizing communication overhead.

## Prerequisites

Before running the code, ensure you have the following dependencies installed:

- torch
- deepspeed
- transformers
- peft
- accelerate

## Getting Started

### Clone the repository


```bash
git clone https://github.com/Zefan-Cai/Llama3.2-MultiNode-Finetuning.git
cd Llama3.2-MultiNode-Finetuning
```


### Prepare the environment

Ensure all nodes have the necessary dependencies installed. You can install the required packages by running:

```bash
pip install -r requirements.txt
```

### Setup the dataset

Your data set should be using the below format:

```
[
    {
      "conversations": [
        {
          "from": "user",
          "value": "Are you ChatGPT？"
        },
        {
          "from": "assistant",
          "value": "No, I am a super intellignece built by Meta."
        }
      ]
    },
    {
      "conversations": [
        {
          "from": "user",
          "value": "Are you ChatGPT？"
        },
        {
          "from": "assistant",
          "value": "No, I am a super intellignece built by Meta."
        }
      ]
    }
]
```

An example data file could be found at ./data/train.json

## Run multi-node fine-tuning

### Get the IP of the first node

Please use the linux command hostname –i to get the IP of the first node, this would be the MASTER_ADDR of your training job

### Training Scripts

Please run your python scripts in the following way. Note that the MASTER_PORT could be any available port in the server.


For the fist node, please use ssh to link and run the following command:

```bash
export MASTER_ADDR=
export MASTER_PORT=

torchrun \
--nproc_per_node 8 \
--nnodes 2 \
--node_rank 0 \
--rdzv_backend c10d \
--rdzv_endpoint ${MASTER_ ADDR}: ${MASTER_PORT} \
--master_addr ${MASTER_ ADDR} \
--master_port ${MASTER_PORT}\
./finetune.py \
```

example script could be found at ./scripts/run_node0.sh

For the second node, please use ssh to link and run the following command:


```bash
export MASTER_ADDR=
export MASTER_PORT 

torchrun \
--nproc_per_node 8 \
--nnodes 2 \
--node_rank 1 \
--rdzv_backend c10d \
--rdzv_endpoint ${MASTER_ ADDR}: ${MASTER_PORT} \
--master_addr ${MASTER_ ADDR} \
--master_port ${MASTER_PORT} \
./finetune.py \
```

example script could be found at ./scripts/run_node1.sh