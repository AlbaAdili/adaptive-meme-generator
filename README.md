# Teaching Generative AI to Laugh  
## An Adaptive Meme Generation System Balancing Quality and Latency Under Load
---
## Project Overview

This project presents an adaptive generative AI system for meme creation using **Stable Diffusion** with **LoRA fine-tuning**, combined with a **queue-based online controller**.

The main objective is to study and optimize the **latency–quality tradeoff** in real-time generative AI systems under varying workloads.

We analyze:

- How diffusion parameters affect inference latency  
- How LoRA fine-tuning improves semantic alignment  
- How tail latency behaves under congestion  
- How load-aware adaptation improves scalability  

---

##  System Architecture

The system consists of three main components:

### Generative Model Layer

- Base model: `runwayml/stable-diffusion-v1-5`
- Optional quality model: `stabilityai/stable-diffusion-xl-base-1.0`
- LoRA fine-tuning applied to UNet attention layers:
  - `to_q`
  - `to_k`
  - `to_v`
  - `to_out.0`


### Adaptive Controller

- Asynchronous queue-based request handling  
- Two inference modes:
  - **Quality Mode** → Higher diffusion steps
  - **Fast Mode** → Reduced diffusion steps  
- Mode switching based on queue length thresholds  
- Reduces tail latency under high load  


### Evaluation Module

- End-to-end latency logging  
- CLIP similarity scoring  
- Latency histograms  
- Latency–quality tradeoff visualization  
- Base vs LoRA quantitative comparison  

---

##  Installation

### Clone Repository

```bash
git clone https://github.com/AlbaAdili/adaptive-meme-generator.git
cd adaptive-meme-generator
```
### Create Environment
```bash
conda create -n memeai python=3.10
conda activate memeai
```

### Install Dependencies
```bash
pip install torch torchvision
pip install diffusers transformers peft safetensors
pip install matplotlib pandas tqdm
```

### LoRA Fine-Tuning

To train LoRA adapters:
```bash
python models/lora_train.py
```


### Running the Adaptive System
```bash
python app.py
```
