# Required Knowledges for project

> Please note: You can **always** ask LLM for help.
> 
> **Important**: We assume you have understanding of `Linux`(bash, ssh, etc), `Git`, and teamwork basics(eg. how to send `issues`, `PR`(pull request))

Because of the nature of this project, knowledge requirements are divided into two parts, **network** and **ML/DL**.

As year 3+ students, you should at least have good understanding of courses **AI foundations**, **Network security technology** and **Computer networks**.


> Here, requirement means you have used that language in a multi-file project which loc(lines of code) > 100
> 
> For languages like `c/c++`, you should have experience on at least one build systems(`make`, `cmake`, `xmake`, etc)

For programming, `Python` is a must and a low level language(eg.`c`, `c++`, `rust`, `zig`, etc) will help a lot.

So the minimum requirement for programming is: `Python` + a GC language other than `Python`(eg.`JS/TS`, `Java`, `GO`, `C#`, `Lua` etc)

Optional: low level language, experience on **network programming**

## Network

1. **OSI 7 layer model**: You should know:

  - major protocols from L2 to L4, including **packet schema**, **routing**, **De/construction**.

  - Connection definition/tracing. Eg. "how to trace a TCP/UDP connection"

2. **Behaviors and common Algos**: You should know:

  - Boardcast mechanism

  - Common routing algos

  - `send/send_to`, `recv/recv_from` or their equivalents on modern OS kernel

3. **Security**: You should know:

  - Common behaviors of firewall

  - Few packets attacks(like `tcp rst`, `replay`, etc)


## ML/DL

## ML / DL

> This section focuses on **practical, engineering-oriented understanding and usage** of ML/DL,  
> rather than theoretical derivation or cutting-edge research.
>
> You are **NOT required** to be an ML researcher,  
> but you should be able to **read code, run experiments, and explain model behaviors**.

---

### 1. Foundations

> Assumed background: completion of a **Year 3 “AI Foundations” (or equivalent)** course.  
> If you have not formally taken such a course, **understanding the core concepts is sufficient**.  
> No in-depth theoretical derivations are required.

You should know:

- The difference between **Machine Learning (ML)** and **Deep Learning (DL)**
- Learning paradigms:
  - Supervised / Unsupervised / Self-supervised learning
- Basic concepts:
  - Dataset / Batch / Epoch
  - Loss function
  - Optimizer (eg. SGD, Adam)
  - Overfitting / Underfitting
- The difference between training and inference

You should be able to:

- Train a basic model using common frameworks (eg. `PyTorch`, `TensorFlow`)
- Understand and implement a standard training loop:
  - Forward pass → loss computation → backward pass → parameter update

---

### 2. Neural Network Basics

You should know:

- Common network layers:
  - Linear / Fully Connected layers
  - Convolution layers
  - Normalization layers (BatchNorm / LayerNorm)
- Common activation functions:
  - ReLU / LeakyReLU / Sigmoid / Tanh
- The **conceptual role** of backpropagation (no formula derivation required)

You should understand:

- How data flows through a neural network
- How gradients affect parameter updates
- Why deeper networks are harder to train

---

### 3. Generative Models (Overview Level)

You should take a glance at the following models and understand their **core ideas and behavioral characteristics**.

#### GAN (Generative Adversarial Network)

You should know:

- The roles of the Generator and the Discriminator
- The adversarial training process
- What the loss functions roughly represent

You should understand:

- Why GAN training can be unstable
- What *mode collapse* means
- Typical use cases (eg. image generation, data augmentation)

Optional but recommended:

- Run or read code of a simple GAN implementation (eg. DCGAN)

---

#### Diffusion Models

You should know:

- The forward process: gradually adding noise to data
- The reverse process: denoising and sampling
- Why diffusion models can generate high-quality samples

You should understand:

- Differences between diffusion models and GANs in training stability
- Why diffusion sampling is usually slower
- High-level ideas of noise prediction vs data prediction

Optional but recommended:

- Run inference using a pretrained diffusion model
- Understand the role of timestep / scheduler

---

### 4. Engineering Perspective

You should be familiar with:

- Differences between GPU and CPU training / inference
- Basic memory and performance considerations
- Model checkpoint loading and saving
- Reproducibility basics (random seed, configuration, logging)

You should be able to:

- Read and modify existing ML/DL codebases
- Debug common issues:
  - NaN loss
  - No convergence
  - OOM (out-of-memory)
- Integrate ML/DL components into a larger system (eg. networked services, data pipelines)

---

### 5. Relation to This Project

You should understand:

- ML/DL models are treated as **modules**, not black boxes
- Model outputs should be **interpretable or observable** when possible
- ML components may interact with:
  - Network traffic
  - Logs / metrics
  - Online or streaming data

You are expected to:

- Use ML/DL as a **tool**, not an end goal
- Be comfortable combining ML logic with system / network code


Take a glance on `GAN` and `Diffusion`. 
