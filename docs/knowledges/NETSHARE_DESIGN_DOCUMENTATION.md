# NetShare: Design & Implementation Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Data Processing Pipeline](#data-processing-pipeline)
5. [Model Implementation](#model-implementation)
6. [Configuration System](#configuration-system)
7. [Distributed Computing](#distributed-computing)
8. [Field Processing System](#field-processing-system)
9. [Usage Examples](#usage-examples)
10. [Dependencies](#dependencies)

## Overview

NetShare is a GAN-based framework for generating synthetic network traffic traces (packet headers and flow headers) that maintains the statistical properties and privacy characteristics of real network data. The system addresses key challenges in synthetic network data generation including fidelity, scalability, and privacy.

### Key Features
- **GAN-based Generation**: Uses DoppelGANger architecture for realistic network trace generation
- **Multi-format Support**: Handles both PCAP and NetFlow formats
- **Distributed Processing**: Leverages Ray for scalable training and generation
- **Privacy Preservation**: Supports differential privacy (DP) options
- **Flexible Encoding**: Various encoding strategies for different data types
- **Quality Assessment**: Built-in visualization and evaluation tools

## Architecture

NetShare follows a modular, component-based architecture with clear separation of concerns:

```
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────┐
│   Generator     │───▶│ Model Manager Layer  │───▶│    Model        │
│                 │    │ (NetShareManager)    │    │ (DoppelGANger)  │
└─────────────────┘    └──────────────────────┘    └─────────────────┘
         │                        │                         │
         ▼                        ▼                         ▼
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────┐
│ Pre/Post        │    │ Ray Distributed      │    │ Training/       │
│ Processor       │    │ Computing            │    │ Generation      │
│ (NetShare)      │    │ (Parallel Processing)│    │ Pipeline        │
└─────────────────┘    └──────────────────────┘    └─────────────────┘
```

### Component Layers

1. **Generator Layer**: Main orchestration class that manages the complete workflow
2. **Model Manager Layer**: Handles training and generation workflows
3. **Model Layer**: Implements the actual GAN algorithms
4. **Pre/Post Processor Layer**: Handles data preparation and transformation
5. **Ray Layer**: Provides distributed computing capabilities

## Core Components

### Generator Class

The `Generator` class serves as the main entry point and workflow coordinator:

```python
from netshare import Generator

generator = Generator(config="config.json")
generator.train(work_folder="results/")
generator.generate(work_folder="results/")
generator.visualize(work_folder="results/")
```

**Key Methods**:
- `train()`: Preprocesses data and trains the GAN model
- `generate()`: Generates synthetic data using the trained model
- `train_and_generate()`: Executes both training and generation in sequence
- `visualize()`: Creates visual comparisons between real and synthetic data

### Model Manager

The `NetShareManager` handles the training and generation workflows:

- **Training Workflow**: Manages data preprocessing, model training, and checkpointing
- **Generation Workflow**: Handles attribute generation, feature generation, and data reconstruction
- **Chunked Processing**: Splits large datasets into chunks for efficient processing

### Model Implementation

The `DoppelGANgerTorchModel` implements the core GAN architecture:

- **Separate Generators**: Distinct generators for attributes and features
- **Conditional Generation**: Features generated conditioned on attributes
- **Multiple Discriminators**: Separate discriminators for attributes and features
- **Sequence Handling**: Supports variable-length sequences with padding

## Data Processing Pipeline

### Preprocessing Stage

The preprocessing pipeline transforms raw network data into GAN-ready format:

1. **Data Ingestion**: Supports PCAP and CSV formats
2. **Data Chunking**: Splits large datasets by size or time windows
3. **Field Processing**: Applies appropriate encodings to different field types
4. **Normalization**: Normalizes continuous fields to [0,1] range
5. **Encoding**: Converts categorical fields using various strategies

### Field Types and Encodings

NetShare supports multiple field types with specialized processing:

- **Continuous Fields**: Numerical data with min-max normalization
- **Discrete Fields**: Categorical data with one-hot encoding
- **Bit Fields**: Integer data converted to bit representations (e.g., IP addresses)
- **Word2Vec Fields**: Embedding-based representation for categorical data

### Post-processing Stage

The post-processing pipeline reconstructs synthetic data to original format:

1. **Denormalization**: Reverses normalization applied during preprocessing
2. **Decoding**: Converts encoded representations back to original format
3. **Format Conversion**: Outputs data in original format (PCAP/NetFlow)
4. **Quality Assessment**: Evaluates synthetic data quality

## Model Implementation

### DoppelGANger Architecture

The core model implements the DoppelGANger architecture which separates:
- **Attribute Generation**: Static properties of network flows (IP addresses, ports, protocol)
- **Feature Generation**: Time-series data within flows (timestamps, packet sizes)

**Key Components**:
- **Attribute Generator**: Creates static flow properties
- **Feature Generator**: Creates time-series data conditioned on attributes
- **Feature Discriminator**: Distinguishes real vs. synthetic features
- **Attribute Discriminator**: Distinguishes real vs. synthetic attributes

**Training Process**:
- Alternating optimization of generator and discriminators
- Gradient penalty for WGAN-GP stability
- Sequence packing for variable-length sequences

### Model Configuration

The model supports various hyperparameters:
- `batch_size`: Training batch size
- `sample_len`: Length of sequences to generate
- `epochs`: Number of training epochs
- `learning_rates`: Generator and discriminator learning rates
- `network_architecture`: Generator/discriminator layer configurations

## Generation Implementation

### DoppelGANger Generator Architecture

The `DoppelGANgerGenerator` class implements the core generation architecture with separate pathways for attributes and features:

#### Attribute Generation
The generator has two attribute generation pathways:
- **Real Attribute Generator**: Generates static properties that are directly learned from data
- **Additional Attribute Generator**: Generates supplementary attributes conditioned on real attributes

```python
# Real attribute generation pathway
real_attribute_gen_without_last_layer = Sequential(
    Linear(attr_latent_dim, attribute_num_units),
    ReLU(),
    BatchNorm1d(attribute_num_units),
    # ... additional layers
)

# Additional attribute generation pathway (conditioned on real attributes)
addi_attribute_gen_without_last_layer = Sequential(
    Linear(attr_latent_dim + real_attribute_out_dim, attribute_num_units),
    ReLU(),
    BatchNorm1d(attribute_num_units),
    # ... additional layers
)
```

The attribute generation process:
1. Takes random noise (`real_attribute_noise`, `addi_attribute_noise`)
2. Passes through separate neural networks to generate real and additional attributes
3. Uses appropriate activation functions based on output type (Softmax for discrete, Sigmoid/Tanh for continuous)
4. Combines real and additional attributes into the final attribute vector

#### Feature Generation
The feature generation uses an LSTM-based architecture:

```python
# LSTM module for sequential feature generation
lstm_module = LSTM(
    input_size=real_attribute_out_dim + addi_attribute_out_dim + feature_latent_dim,
    hidden_size=feature_num_units,
    num_layers=feature_num_layers,
    batch_first=True
)
```

The feature generation process:
1. Combines attributes with feature noise as input to the LSTM
2. Uses LSTM to generate sequential features over time steps
3. Applies separate output layers for each feature dimension
4. Uses appropriate activation functions based on feature types

#### Generation Process
The `generate` method in the `DoppelGANger` class orchestrates the full generation:

```python
def generate(
    self,
    num_samples,
    given_attribute=None,
    given_attribute_discrete=None,
    return_gen_flag_feature=False,
):
    # Generate noise inputs for the generator
    real_attribute_noise = self._gen_attribute_input_noise(num_samples).to(self.device)
    addi_attribute_noise = self._gen_attribute_input_noise(num_samples).to(self.device)
    feature_input_noise = self._gen_feature_input_noise(num_samples, self.sample_time).to(self.device)
    h0 = Variable(torch.normal(0, 1, (self.generator.feature_num_layers, num_samples, self.generator.feature_num_units))).to(self.device)
    c0 = Variable(torch.normal(0, 1, (self.generator.feature_num_layers, num_samples, self.generator.feature_num_units))).to(self.device)

    # Generate in batches
    generated_data_list = []
    for n_batch in range(num_batches):
        generated_data_list.append(
            self._generate(
                real_attribute_noise=real_attribute_noise[n_batch * self.batch_size: (n_batch + 1) * self.batch_size],
                addi_attribute_noise=addi_attribute_noise[n_batch * self.batch_size: (n_batch + 1) * self.batch_size],
                feature_input_noise=feature_input_noise[n_batch * self.batch_size: (n_batch + 1) * self.batch_size],
                h0=h0[:, n_batch * self.batch_size: (n_batch + 1) * self.batch_size, :],
                c0=c0[:, n_batch * self.batch_size: (n_batch + 1) * self.batch_size, :],
                given_attribute=batch_given_attribute,
                given_attribute_discrete=batch_given_attribute_discrete))
```

### Generation Pipeline

The generation process follows a multi-stage pipeline:

#### 1. Attribute Generation Phase
- Generates static flow properties (source IP, destination IP, ports, protocol)
- Can use pre-generated attributes or generate them from noise
- Handles both real and additional attributes

#### 2. Feature Generation Phase
- Uses LSTM to generate time-series features conditioned on attributes
- Generates sequential data with proper temporal dependencies
- Handles variable-length sequences using generation flags

#### 3. Adaptive Rolling
The generator implements adaptive rolling for efficient sequence generation:

```python
if self.use_adaptive_rolling:
    hn, cn = h0, c0
    feature = []
    batch_size = feature_input.size()[0]
    steps = feature_input.size()[1]
    data = feature_input.unbind(1)
    curr_step = 0
    for xt in data:
        output_per_step, (hn, cn) = self.lstm_module(xt[:, None, :], (hn, cn))
        # Generate features for this time step
        feature_per_step = []
        for feature_layer in self.feature_gen_last_layer:
            feature_sub_output = feature_layer(output_per_step)
            feature_per_step.append(feature_sub_output)
        feature_per_step = torch.cat(feature_per_step, dim=2)

        # Check generation flags to determine if sequence should continue
        gen_flag_per_step = feature_per_step[:, :, self.feature_out_dim - 2:: self.feature_out_dim]
        feature.append(feature_per_step)
        curr_step += 1
        tmp_, _ = torch.min((gen_flag_per_step > 0.5).int(), 2)
        if torch.max(tmp_) == 0:
            # All the gen flag is false in this case
            break
```

#### 4. Generation Flag Processing
The system uses generation flags to handle variable-length sequences:
- Each sequence has a generation flag indicating whether it should continue
- Sequences that should end are masked with zeros
- Proper handling of sequence boundaries and padding

### Data Loading and Preprocessing

The `load_data` function handles data preparation for generation:

```python
def load_data(path, sample_len, flag="train"):
    data_npz = np.load(os.path.join(path, "data_{}.npz".format(flag)))
    with open(os.path.join(path, "data_feature_output.pkl"), "rb") as f:
        data_feature_outputs = pickle.load(f)
    with open(os.path.join(path, "data_attribute_output.pkl"), "rb") as f:
        data_attribute_outputs = pickle.load(f)

    # Pad sequences to be multiples of sample_len
    timeseries_len = data_feature.shape[1]
    ceil_timeseries_len = math.ceil(timeseries_len / sample_len) * sample_len
    data_feature = np.pad(
        data_feature,
        pad_width=((0, 0), (0, ceil_timeseries_len - timeseries_len), (0, 0)),
        mode='constant', constant_values=0)
    data_gen_flag = np.pad(
        data_gen_flag,
        pad_width=((0, 0), (0, ceil_timeseries_len - timeseries_len)),
        mode='constant', constant_values=0)

    return (data_feature, data_attribute, data_gen_flag, data_feature_outputs, data_attribute_outputs)
```

### Normalization and Denormalization

The system implements sophisticated normalization strategies:

#### Self-Normalization
The `normalize_per_sample` function implements per-sample normalization:

```python
def normalize_per_sample(data_feature, data_attribute, data_feature_outputs, data_attribute_outputs, eps=1e-4):
    # Calculate min/max for each sample
    data_feature_min = np.amin(data_feature, axis=1)
    data_feature_max = np.amax(data_feature, axis=1)

    additional_attribute = []
    additional_attribute_outputs = []

    dim = 0
    for output in data_feature_outputs:
        if output.type_ == OutputType.CONTINUOUS:
            for _ in range(output.dim):
                max_ = data_feature_max[:, dim] + eps
                min_ = data_feature_min[:, dim] - eps

                # Store normalization parameters as additional attributes
                additional_attribute.append((max_ + min_) / 2.0)
                additional_attribute.append((max_ - min_) / 2.0)
                # ... add to additional attributes
```

#### Renormalization
The `renormalize_per_sample` function reverses the normalization:

```python
def renormalize_per_sample(data_feature, data_attribute, data_feature_outputs,
                          data_attribute_outputs, gen_flags, num_real_attribute):
    attr_dim = 0
    for i in range(num_real_attribute):
        attr_dim += data_attribute_outputs[i].dim
    attr_dim_cp = attr_dim

    fea_dim = 0
    for output in data_feature_outputs:
        if output.type_ == OutputType.CONTINUOUS:
            for _ in range(output.dim):
                # Retrieve normalization parameters from attributes
                max_plus_min_d_2 = data_attribute[:, attr_dim]
                max_minus_min_d_2 = data_attribute[:, attr_dim + 1]
                attr_dim += 2

                max_ = max_plus_min_d_2 + max_minus_min_d_2
                min_ = max_plus_min_d_2 - max_minus_min_d_2

                max_ = np.expand_dims(max_, axis=1)
                min_ = np.expand_dims(min_, axis=1)

                # Apply denormalization
                data_feature[:, :, fea_dim] = (data_feature[:, :, fea_dim] * (max_ - min_)) + min_
                fea_dim += 1
```

### Distributed Generation Pipeline

The system implements a sophisticated distributed generation pipeline through Ray:

#### Attribute Generation Phase
```python
# Generate attributes in parallel across chunks
objs = []
for config_idx, config in enumerate(configs):
    objs.append(_generate_attr.remote(
        create_new_model=create_new_model,
        configs=configs,
        config_idx=config_idx,
        log_folder=log_folder))
_ = ray.get(objs)
```

#### Attribute Merging
The `_merge_attr` function handles cross-chunk flow consistency:

```python
def _merge_attr(attr_raw_npz_folder, config_group, configs):
    # Process generated attributes from each chunk
    # Handle flows that span multiple chunks
    # Ensure consistency across chunk boundaries
    for chunkid in range(num_chunks):
        raw_attr_chunk = np.load(os.path.join(attr_raw_npz_folder, "chunk_id-{}.npz".format(chunkid)))["data_attribute"]
        raw_attr_discrete_chunk = np.load(os.path.join(attr_raw_npz_folder, "chunk_id-{}.npz".format(chunkid)))["data_attribute_discrete"]

        # Process flows that start in this chunk and continue in later chunks
        for row_idx, row in enumerate(raw_attr_chunk):
            if (row[bit_idx_flagstart] < row[bit_idx_flagstart + 1] and
                row[bit_idx_flagstart + 2 * chunkid + 2] < row[bit_idx_flagstart + 2 * chunkid + 3]):
                # This flow starts in this chunk
                # Process and assign to appropriate chunks
```

#### Feature Generation Phase
After attribute merging, features are generated based on the merged attributes:

```python
# Generate features given the merged attributes
objs = []
for config_idx, config in enumerate(configs):
    objs.append(_generate_given_attr.remote(
        create_new_model=create_new_model,
        configs=configs,
        config_idx=config_idx,
        log_folder=log_folder))
_ = ray.get(objs)
```

### Generation Helper Functions

The system provides several specialized generation functions:

#### `_generate_session`
Handles single-session generation without chunking:

```python
@ray.remote(scheduling_strategy="SPREAD", max_calls=1)
def _generate_session(create_new_model, configs, config_idx, log_folder):
    config = configs[config_idx]
    config["given_data_attribute_flag"] = False
    model = create_new_model(config)
    model.generate(
        input_train_data_folder=config["dataset"],
        input_model_folder=config["result_folder"],
        output_syn_data_folder=config["eval_root_folder"],
        log_folder=log_folder)
```

#### `_generate_attr`
Generates attributes in parallel:

```python
@ray.remote(scheduling_strategy="SPREAD", max_calls=1)
def _generate_attr(create_new_model, configs, config_idx, log_folder):
    config = configs[config_idx]
    config["given_data_attribute_flag"] = False  # Generate attributes from noise
    model = create_new_model(config)
    model.generate(
        input_train_data_folder=config["dataset"],
        input_model_folder=config["result_folder"],
        output_syn_data_folder=config["eval_root_folder"],
        log_folder=log_folder)
```

#### `_generate_given_attr`
Generates features given pre-generated attributes:

```python
@ray.remote(scheduling_strategy="SPREAD", max_calls=1)
def _generate_given_attr(create_new_model, configs, config_idx, log_folder):
    config = configs[config_idx]
    config["given_data_attribute_flag"] = True  # Use given attributes
    model = create_new_model(config)
    model.generate(
        input_train_data_folder=config["dataset"],
        input_model_folder=config["result_folder"],
        output_syn_data_folder=config["eval_root_folder"],
        log_folder=log_folder)
```

This distributed approach allows NetShare to handle large-scale network trace generation while maintaining consistency across chunk boundaries and preserving the statistical properties of the original data.

## Configuration System

NetShare uses a hierarchical configuration system:

## Extending NetShare to Other Protocols

NetShare is designed to be extensible to various network protocols beyond the provided examples (NetFlow and PCAP). This guide outlines how to adapt the system for other protocols.

### Protocol Extension Overview

NetShare's flexibility comes from its modular design that separates:
1. **Data Input/Output**: Handles different data formats (PCAP, CSV, etc.)
2. **Field Processing**: Handles different field types and encodings
3. **Model Architecture**: Generic GAN architecture that works with any structured data
4. **Configuration System**: Flexible configuration that can be adapted to any protocol

### Step-by-Step Guide to Protocol Extension

#### 1. Data Preparation and Format

The first step is to prepare your protocol data in a format that NetShare can process:

- **CSV Format**: Convert your protocol data to CSV format with appropriate columns
- **Column Structure**: Each row should represent a network event (packet, flow, etc.)
- **Required Columns**: At minimum, you need timestamp and identifier columns

Example for a custom protocol:
```csv
timestamp,src_ip,dst_ip,src_port,dst_port,protocol,packet_size,flags,ttl,custom_field1,custom_field2
1609459200,192.168.1.1,192.168.1.2,12345,80,6,1500,2,64,0,1
1609459201,192.168.1.2,192.168.1.1,80,12345,6,500,18,128,1,0
```

#### 2. Configuration Design

Create a configuration file that defines how to process your protocol's fields. The configuration has three main sections:

**Global Configuration**:
```json
{
  "global_config": {
    "original_data_file": "path/to/your/protocol_data.csv",
    "overwrite": true,
    "dataset_type": "custom",  // Can be any string
    "n_chunks": 2,  // Number of chunks for distributed processing
    "dp": false  // Whether to use differential privacy
  },
  ...
}
```

**Pre/Post Processor Configuration**:
Define how each field in your protocol should be processed:

```json
"pre_post_processor": {
  "class": "NetsharePrePostProcessor",
  "config": {
    "timestamp": {
      "column": "timestamp",  // Name of timestamp column
      "generation": true,
      "encoding": "interarrival",  // How to handle time
      "normalization": "ZERO_ONE"
    },
    "word2vec": {
      "vec_size": 10,
      "model_name": "word2vec_vecSize",
      "annoy_n_trees": 100,
      "pretrain_model_path": null
    },
    "metadata": [
      // Static properties that define a flow/session
      {
        "column": "src_ip",
        "type": "integer",
        "encoding": "bit",
        "n_bits": 32
      },
      {
        "column": "dst_ip",
        "type": "integer",
        "encoding": "bit",
        "n_bits": 32
      },
      {
        "column": "protocol",
        "type": "string",
        "encoding": "word2vec_proto"
      }
    ],
    "timeseries": [
      // Dynamic properties that change over time within a flow
      {
        "column": "packet_size",
        "type": "float",
        "normalization": "ZERO_ONE",
        "log1p_norm": true
      },
      {
        "column": "ttl",
        "type": "float",
        "normalization": "ZERO_ONE"
      },
      {
        "column": "flags",
        "type": "integer",
        "encoding": "categorical",
        "choices": [0, 1, 2, 18]  // Specific values for your protocol
      }
    ]
  }
}
```

#### 3. Field Type Selection

Choose appropriate field types based on your protocol's data characteristics:

**Continuous Fields** (`type: "float"`):
- Use for numerical values like packet sizes, TTL, timestamps
- Apply normalization (`ZERO_ONE` or `MINUSONE_ONE`)
- Consider `log1p_norm` for values with large ranges

**Discrete Fields** (`type: "integer"` or `"string"` with `encoding: "categorical"`):
- Use for fields with limited set of values
- Specify `choices` for all possible values
- Good for protocol flags, status codes, etc.

**Bit Fields** (`type: "integer"` with `encoding: "bit"`):
- Use for IP addresses (32-bit) or MAC addresses (48-bit)
- Converts integers to bit representations
- Preserves bit-level patterns

**Word2Vec Fields** (`encoding: "word2vec_*"`):
- Use for categorical fields with semantic relationships
- Good for protocol types, service names, etc.
- Creates embeddings that capture relationships between categories

#### 4. Custom Field Processing

If your protocol has unique field types not covered by existing field processors, you can extend the system:

```python
from netshare.utils.field import Field

class CustomProtocolField(Field):
    def __init__(self, custom_param, *args, **kwargs):
        super(CustomProtocolField, self).__init__(*args, **kwargs)
        self.custom_param = custom_param
        self.dim_x = custom_param  # Define output dimension

    def normalize(self, x):
        # Implement normalization logic for your custom field
        # Return normalized values suitable for neural network input
        pass

    def denormalize(self, norm_x):
        # Implement denormalization logic
        # Convert neural network output back to original format
        pass

    def getOutputType(self):
        # Define output type for neural network
        from netshare.utils.output import OutputType, Output, Normalization
        return Output(
            type_=OutputType.CONTINUOUS,  # or OutputType.DISCRETE
            dim=self.dim_x,
            normalization=Normalization.ZERO_ONE
        )
```

#### 5. Model Configuration

Adjust the model parameters based on your protocol's complexity:

```json
"model": {
  "class": "DoppelGANgerTorchModel",
  "config": {
    "batch_size": 100,  // Adjust based on your data size
    "sample_len": [1, 5, 10],  // Sequence lengths to generate
    "epochs": 40,  // Training epochs
    "extra_checkpoint_freq": 1,
    "epoch_checkpoint_freq": 5,
    // Additional GAN hyperparameters
    "g_lr": 0.0002,  // Generator learning rate
    "d_lr": 0.0002,  // Discriminator learning rate
    "d_rounds": 1,   // Discriminator training rounds per generator round
    "g_rounds": 1    // Generator training rounds
  }
}
```

#### 6. Example: Extending to DNS Protocol

Here's a complete example for a DNS protocol extension:

**Sample DNS Data**:
```csv
timestamp,src_ip,dst_ip,src_port,dst_port,query_type,query_name,response_code,ttl,packet_size
1609459200,192.168.1.1,8.8.8.8,12345,53,A,www.example.com,0,300,120
1609459201,192.168.1.1,8.8.8.8,12346,53,AAAA,www.example.com,0,300,140
```

**DNS Configuration**:
```json
{
  "global_config": {
    "original_data_file": "data/dns_data.csv",
    "overwrite": true,
    "dataset_type": "dns",
    "n_chunks": 2,
    "dp": false
  },
  "pre_post_processor": {
    "class": "NetsharePrePostProcessor",
    "config": {
      "timestamp": {
        "column": "timestamp",
        "generation": true,
        "encoding": "interarrival",
        "normalization": "ZERO_ONE"
      },
      "word2vec": {
        "vec_size": 20,
        "model_name": "dns_word2vec",
        "annoy_n_trees": 100,
        "pretrain_model_path": null
      },
      "metadata": [
        {
          "column": "src_ip",
          "type": "integer",
          "encoding": "bit",
          "n_bits": 32
        },
        {
          "column": "dst_ip",
          "type": "integer",
          "encoding": "bit",
          "n_bits": 32
        },
        {
          "column": "dst_port",
          "type": "integer",
          "encoding": "word2vec_port"
        }
      ],
      "timeseries": [
        {
          "column": "query_type",
          "type": "string",
          "encoding": "word2vec_dns_type"
        },
        {
          "column": "response_code",
          "type": "integer",
          "encoding": "categorical",
          "choices": [0, 1, 2, 3, 4, 5]
        },
        {
          "column": "ttl",
          "type": "float",
          "normalization": "ZERO_ONE",
          "min_x": 0,
          "max_x": 86400
        },
        {
          "column": "packet_size",
          "type": "float",
          "normalization": "ZERO_ONE",
          "log1p_norm": true
        }
      ]
    }
  },
  "model": {
    "class": "DoppelGANgerTorchModel",
    "config": {
      "batch_size": 50,
      "sample_len": [1, 3, 5],
      "epochs": 60
    }
  }
}
```

#### 7. Validation and Testing

After configuring for your protocol:

1. **Validate Configuration**: Ensure all column names match your data
2. **Test Preprocessing**: Run preprocessing to verify data is processed correctly
3. **Monitor Training**: Check loss curves and generated samples
4. **Evaluate Quality**: Use built-in visualization tools to compare real vs synthetic data
5. **Adjust Parameters**: Fine-tune based on results

#### 8. Best Practices for Protocol Extension

- **Start Simple**: Begin with a subset of important fields before adding complexity
- **Data Quality**: Ensure your input data is clean and representative
- **Field Grouping**: Group static properties in `metadata` and dynamic properties in `timeseries`
- **Normalization**: Use appropriate normalization for different data types
- **Chunk Size**: Adjust `n_chunks` based on your data size and available resources
- **Validation**: Always validate generated data against domain knowledge

By following this guide, you can adapt NetShare to generate synthetic data for virtually any network protocol while leveraging its powerful GAN-based generation capabilities and distributed processing framework.

### Detailed Protocol Extension Guide

#### Data Requirements for New Protocols

**Required Data Format:**
NetShare requires structured data in CSV format with the following characteristics:

**Basic Requirements:**
- **Timestamp Column**: A timestamp field to establish temporal relationships
- **Identifier Columns**: Fields that can group related events into flows/sessions
- **Feature Columns**: Various protocol-specific fields to capture the behavior

**Data Quality Requirements:**
- **Completeness**: All required fields must be present for each record
- **Consistency**: Data types should be consistent across the dataset
- **Temporal Order**: Records should be ordered chronologically if time relationships are important

**Example Data Structure:**
```csv
timestamp,src_ip,dst_ip,src_port,dst_port,protocol,packet_size,flags,ttl,custom_field1,custom_field2
1609459200,192.168.1.1,192.168.1.2,12345,80,6,1500,2,64,0,1
1609459201,192.168.1.2,192.168.1.1,80,12345,6,500,18,128,1,0
```

**Data Preprocessing Requirements:**
- **Minimum Dataset Size**: At least 10,000+ records recommended for stable training
- **Feature Distribution**: Sufficient variation in values to learn meaningful patterns
- **Flow Grouping**: Clear way to group related records (e.g., by source/destination IP pairs)

#### Training Process for New Protocols

**1. Data Preparation Phase:**
The training process begins with data preprocessing:

```python
# The Generator class handles the complete workflow
generator = Generator(config="your_protocol_config.json")
generator.train(work_folder="results/your_protocol/")
```

**Preprocessing Steps:**
1. **Data Loading**: Load CSV data and validate column structure
2. **Flow Grouping**: Group related records based on metadata fields
3. **Normalization**: Apply appropriate normalization to continuous fields
4. **Encoding**: Convert categorical fields using specified encoding strategies
5. **Chunking**: Split large datasets into manageable chunks for distributed processing

**2. Model Training Phase:**
The DoppelGANger model trains in adversarial fashion:

**Training Components:**
- **Attribute Generator**: Learns to generate static flow properties
- **Feature Generator**: Learns to generate time-series features conditioned on attributes
- **Feature Discriminator**: Distinguishes real vs. synthetic features
- **Attribute Discriminator**: Distinguishes real vs. synthetic attributes

**Training Process:**
1. **Initialization**: Set up neural networks with specified architecture
2. **Adversarial Training**: Alternate between discriminator and generator updates
3. **Gradient Penalty**: Apply WGAN-GP for stable training
4. **Checkpointing**: Save model states at specified intervals
5. **Monitoring**: Track loss metrics and generate samples for validation

**3. Training Configuration Parameters:**

**Critical Training Parameters:**
- `epochs`: Number of complete passes through the dataset (start with 40-100)
- `batch_size`: Number of samples processed together (adjust based on GPU memory)
- `sample_len`: Length of sequences to generate (affects temporal dependencies)
- `learning_rates`: Generator and discriminator learning rates (typically 0.0002)
- `d_rounds/g_rounds`: Ratio of discriminator to generator updates (usually 1:1)

**Advanced Parameters:**
- `d_gp_coe`: Gradient penalty coefficient for discriminator stability
- `num_packing`: Number of sequences packed together for training
- `g_attr_d_coe`: Weight for attribute discriminator loss in generator

#### Pre/Post Processing Modifications

**Understanding the Pre/Post Processing Pipeline:**

The pre/post processing system is highly configurable and typically doesn't require code modifications for new protocols. However, understanding the components helps with configuration:

**Preprocessing Components:**
- **Data Ingestion**: Handles CSV files and converts PCAP if needed
- **Field Processing**: Applies normalization and encoding based on configuration
- **Flow Identification**: Groups related records using metadata fields
- **Data Chunking**: Splits data for distributed processing

**Postprocessing Components:**
- **Denormalization**: Reverses normalization applied during preprocessing
- **Decoding**: Converts encoded representations back to original format
- **Format Conversion**: Outputs data in desired format
- **Quality Assessment**: Evaluates synthetic data quality

**Custom Pre/Post Processing (When Needed):**

If your protocol requires special preprocessing that isn't covered by existing field types, you can extend the system:

**Creating Custom Pre/Post Processor:**
```python
from netshare.pre_post_processors.pre_post_processor import PrePostProcessor

class CustomProtocolPrePostProcessor(PrePostProcessor):
    def _pre_process(self, input_folder, output_folder, log_folder):
        # Custom preprocessing logic for your protocol
        # Load and validate your data
        # Apply custom transformations
        # Save processed data in NetShare format
        pass

    def _post_process(self, input_folder, output_folder,
                      pre_processed_data_folder, log_folder):
        # Custom postprocessing logic
        # Convert generated data back to your protocol format
        # Apply any protocol-specific transformations
        pass
```

**Registering Custom Processor:**
```json
{
  "pre_post_processor": {
    "class": "CustomProtocolPrePostProcessor",
    "config": {
      // Your custom configuration
    }
  }
}
```

**Field Processing Extensions:**

For new field types, extend the field processing system:

```python
from netshare.utils.field import Field
from netshare.utils.output import OutputType, Output, Normalization

class CustomProtocolField(Field):
    def __init__(self, custom_param, *args, **kwargs):
        super(CustomProtocolField, self).__init__(*args, **kwargs)
        self.custom_param = custom_param
        self.dim_x = custom_param  # Define output dimension

    def normalize(self, x):
        # Implement normalization logic for your custom field
        # Return normalized values suitable for neural network input
        # Example: custom encoding for protocol-specific values
        normalized_values = self.custom_encoding(x)
        return normalized_values

    def denormalize(self, norm_x):
        # Implement denormalization logic
        # Convert neural network output back to original format
        original_values = self.custom_decoding(norm_x)
        return original_values

    def getOutputType(self):
        # Define output type for neural network
        return Output(
            type_=OutputType.CONTINUOUS,  # or OutputType.DISCRETE
            dim=self.dim_x,
            normalization=Normalization.ZERO_ONE
        )
```

#### Step-by-Step Implementation Guide

**Step 1: Data Preparation**
1. **Collect Protocol Data**: Gather representative samples of your protocol traffic
2. **Format Conversion**: Convert to CSV with appropriate columns
3. **Data Validation**: Ensure data quality and completeness
4. **Flow Identification**: Determine how to group related records

**Step 2: Configuration Design**
1. **Identify Metadata Fields**: Static properties that define flows/sessions
2. **Identify Timeseries Fields**: Dynamic properties that change over time
3. **Select Field Types**: Choose appropriate field types and encodings
4. **Configure Parameters**: Set model and training parameters

**Step 3: Initial Training**
1. **Start Small**: Begin with a subset of important fields
2. **Monitor Training**: Watch loss curves and sample quality
3. **Adjust Parameters**: Fine-tune based on training behavior
4. **Validate Results**: Check synthetic data quality

**Step 4: Iterative Improvement**
1. **Add Complexity**: Gradually add more fields and complexity
2. **Optimize Performance**: Adjust hyperparameters for better results
3. **Validate Use Cases**: Test synthetic data for intended applications
4. **Document Findings**: Record successful configurations and parameters

#### Example: Extending to a Custom IoT Protocol

**Sample IoT Data:**
```csv
timestamp,device_id,device_type,location,temperature,humidity,pressure,battery_level,status_code,event_type
1609459200,iot_001,sensor,room_1,23.5,45.2,1013.25,87,200,reading
1609459260,iot_001,sensor,room_1,23.7,45.1,1013.20,87,200,reading
```

**IoT Protocol Configuration:**
```json
{
  "global_config": {
    "original_data_file": "data/iot_data.csv",
    "overwrite": true,
    "dataset_type": "iot",
    "n_chunks": 1,
    "dp": false
  },
  "pre_post_processor": {
    "class": "NetsharePrePostProcessor",
    "config": {
      "timestamp": {
        "column": "timestamp",
        "generation": true,
        "encoding": "interarrival",
        "normalization": "ZERO_ONE"
      },
      "word2vec": {
        "vec_size": 15,
        "model_name": "iot_word2vec",
        "annoy_n_trees": 100,
        "pretrain_model_path": null
      },
      "metadata": [
        {
          "column": "device_id",
          "type": "string",
          "encoding": "word2vec_device"
        },
        {
          "column": "device_type",
          "type": "string",
          "encoding": "categorical"
        },
        {
          "column": "location",
          "type": "string",
          "encoding": "word2vec_location"
        }
      ],
      "timeseries": [
        {
          "column": "temperature",
          "type": "float",
          "normalization": "ZERO_ONE",
          "min_x": -40.0,
          "max_x": 85.0
        },
        {
          "column": "humidity",
          "type": "float",
          "normalization": "ZERO_ONE"
        },
        {
          "column": "pressure",
          "type": "float",
          "normalization": "ZERO_ONE",
          "log1p_norm": true
        },
        {
          "column": "battery_level",
          "type": "float",
          "normalization": "ZERO_ONE"
        },
        {
          "column": "status_code",
          "type": "integer",
          "encoding": "categorical",
          "choices": [200, 201, 400, 401, 404, 500]
        },
        {
          "column": "event_type",
          "type": "string",
          "encoding": "categorical"
        }
      ]
    }
  },
  "model": {
    "class": "DoppelGANgerTorchModel",
    "config": {
      "batch_size": 64,
      "sample_len": [1, 5, 10],
      "epochs": 80,
      "extra_checkpoint_freq": 1,
      "epoch_checkpoint_freq": 10,
      "g_lr": 0.0002,
      "d_lr": 0.0002
    }
  }
}
```

#### Training Monitoring and Troubleshooting

**Monitoring Training Progress:**
- **Loss Curves**: Monitor generator and discriminator losses for stability
- **Sample Quality**: Regularly inspect generated samples for realism
- **Convergence**: Look for stable loss values indicating proper training

**Common Issues and Solutions:**
- **Mode Collapse**: Increase discriminator capacity or adjust learning rates
- **Poor Quality**: Increase training epochs or adjust model architecture
- **Memory Issues**: Reduce batch size or increase chunking
- **Temporal Issues**: Adjust sample_len or sequence modeling approach

#### Validation and Quality Assessment

**Built-in Validation Tools:**
NetShare provides several tools for assessing synthetic data quality:
- **Statistical Comparison**: Compare distributions of real vs. synthetic data
- **Visualization**: Side-by-side plots of real and generated data
- **Downstream Tasks**: Test synthetic data on intended applications

**Custom Validation:**
For protocol-specific validation, consider:
- **Protocol Compliance**: Verify generated data follows protocol specifications
- **Behavioral Patterns**: Check for realistic temporal and behavioral patterns
- **Anomaly Detection**: Test if synthetic data can be distinguished from real data

## Data Requirements for GAN and Diffusion Models

Understanding the data requirements is crucial for training both GAN and diffusion models effectively in NetShare. The quality and structure of your training data directly impacts the quality of generated synthetic network traces.

### General Data Requirements

#### Data Format and Structure
Both GAN and diffusion models in NetShare require structured data in CSV format with the following characteristics:

**Basic Requirements:**
- **Timestamp Column**: A timestamp field to establish temporal relationships between network events
- **Identifier Columns**: Fields that can group related events into coherent flows or sessions (e.g., source/destination IP pairs, port numbers)
- **Feature Columns**: Various protocol-specific fields that capture the behavior and characteristics of network traffic
- **Consistent Schema**: All records must follow the same column structure

**Example Data Structure:**
```csv
timestamp,src_ip,dst_ip,src_port,dst_port,protocol,packet_size,flags,ttl,custom_field1,custom_field2
1609459200,192.168.1.1,192.168.1.2,12345,80,6,1500,2,64,0,1
1609459201,192.168.1.2,192.168.1.1,80,12345,6,500,18,128,1,0
```

#### Data Quality Requirements
- **Completeness**: All required fields must be present for each record with minimal missing values
- **Consistency**: Data types should be consistent across the dataset (e.g., all IP addresses in the same format)
- **Temporal Order**: Records should be ordered chronologically when temporal relationships are important
- **Representativeness**: Data should be representative of the network behavior you want to model

### Specific Requirements for GAN Models

#### Dataset Size and Diversity
- **Minimum Size**: At least 10,000+ records recommended for stable GAN training
- **Feature Distribution**: Sufficient variation in values to learn meaningful patterns and avoid mode collapse
- **Temporal Patterns**: Multiple examples of similar temporal patterns to learn from
- **Flow Characteristics**: Diverse examples of different flow types and behaviors

#### GAN-Specific Considerations
- **Balanced Distributions**: Avoid highly imbalanced categorical variables that can cause mode collapse
- **Normalization Range**: Features should be normalized to appropriate ranges (typically [0,1] or [-1,1])
- **Sequence Length**: For time-series data, consistent sequence lengths help with training stability

### Specific Requirements for Diffusion Models

#### Dataset Size and Diversity
- **Minimum Size**: Similar to GANs, 10,000+ records recommended, but diffusion models can sometimes work with smaller datasets due to their training stability
- **Temporal Continuity**: Diffusion models benefit from continuous temporal sequences to learn denoising patterns
- **Multi-Modal Data**: Diffusion models handle multi-modal distributions better than GANs, so diverse data types are beneficial

#### Diffusion-Specific Considerations
- **Noise Robustness**: Diffusion models are more robust to some types of noise in the data
- **Normalization**: Works well with normalized data in [0,1] or [-1,1] ranges
- **Temporal Dependencies**: Can better capture long-term temporal dependencies than GANs

### Data Preprocessing Requirements

#### Flow Grouping
Both models require clear identification of related network events:

**For Network Flows:**
- Group packets by source/destination IP pairs
- Group by source/destination port combinations
- Consider protocol-specific flow identification methods

**For Time-Series Segments:**
- Define appropriate time windows for chunking
- Maintain temporal continuity within chunks
- Handle cross-chunk flows appropriately

#### Feature Engineering
**Continuous Features:**
- Numerical values like packet sizes, TTL, timestamps
- Apply appropriate normalization (min-max, log1p, etc.)
- Handle outliers that could affect training

**Categorical Features:**
- Protocol types, status codes, flags
- Use appropriate encoding (one-hot, embedding, etc.)
- Ensure sufficient examples for each category

**Special Fields:**
- IP addresses: Use bit encoding for 32-bit representation
- Port numbers: Can use categorical or embedding approaches
- Protocol identifiers: Word2Vec or categorical encoding

### Data Quality Assessment

#### Statistical Properties
Before training, verify that your data has:
- **Sufficient Variance**: Features should have meaningful variation
- **Representative Distributions**: Data should represent real-world scenarios
- **Temporal Patterns**: Time-based relationships should be preserved
- **Cross-Feature Correlations**: Important relationships between features should exist

#### Data Validation Steps
1. **Distribution Analysis**: Check histograms and statistical properties
2. **Missing Value Assessment**: Identify and handle missing data appropriately
3. **Outlier Detection**: Identify extreme values that might affect training
4. **Temporal Consistency**: Verify chronological ordering and time gaps

### Recommended Data Collection Strategies

#### For Network Traffic Modeling
- **Capture Duration**: Collect data over sufficient time periods to capture various network behaviors
- **Traffic Diversity**: Include different types of network traffic (web, streaming, file transfer, etc.)
- **Load Variations**: Capture data during different load conditions (peak, off-peak)
- **Event Coverage**: Include various network events and scenarios

#### For Protocol-Specific Modeling
- **Protocol Variants**: Include different versions and variants of the protocol
- **Usage Patterns**: Capture various usage patterns and configurations
- **Error Conditions**: Include examples of error conditions and unusual behaviors
- **Normal Operations**: Focus on normal operational patterns

### Data Preparation Workflow

#### Step 1: Data Collection
- Gather representative network traffic data
- Ensure compliance with privacy and security policies
- Document data sources and collection methodology

#### Step 2: Data Cleaning
- Remove or handle missing values
- Correct obvious errors or inconsistencies
- Normalize timestamps and formats

#### Step 3: Feature Selection
- Identify relevant features for your use case
- Remove redundant or irrelevant features
- Consider privacy implications of sensitive fields

#### Step 4: Data Splitting
- Create train/validation/test splits if needed
- Ensure temporal consistency in splits
- Maintain similar distributions across splits

By following these data requirements, you can prepare high-quality datasets that will enable both GAN and diffusion models in NetShare to learn meaningful patterns and generate realistic synthetic network traces.

## Replacing GAN with Diffusion Model

NetShare's modular architecture allows for replacing the GAN model with a diffusion model. This section provides a comprehensive guide for implementing this change while maintaining the existing preprocessing, postprocessing, and distributed computing infrastructure.

### Understanding the Architecture Change

The current NetShare architecture uses a GAN-based approach (DoppelGANger) with:
- Generator networks for attributes and features
- Discriminator networks for adversarial training
- WGAN-GP loss function with gradient penalty

A diffusion model would replace this with:
- Forward diffusion process that gradually adds noise
- Reverse denoising process that learns to remove noise
- Time-conditional neural networks for generation

### Diffusion Model Architecture Design

#### Core Components

**Diffusion Process:**
- Forward process: Gradually corrupts data with Gaussian noise
- Reverse process: Learns to denoise and reconstruct original data
- Time embedding: Encodes the diffusion timestep for conditional generation

**Neural Network Architecture:**
- Time-conditional U-Net or Transformer for sequence modeling
- Separate networks for attribute and feature generation (similar to DoppelGANger)
- Attention mechanisms for capturing temporal dependencies

#### Implementation Strategy

**1. Diffusion Model Base Class:**
```python
from netshare.models.model import Model
import torch
import torch.nn as nn

class DiffusionModel(Model):
    def __init__(self, config):
        super(DiffusionModel, self).__init__(config)
        self.timesteps = config.get('timesteps', 1000)
        self.beta_start = config.get('beta_start', 1e-4)
        self.beta_end = config.get('beta_end', 0.02)

        # Define noise schedule
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Neural network for denoising
        self.denoise_network = self._build_network(config)

    def _build_network(self, config):
        # Build time-conditional network for denoising
        # Can be U-Net, Transformer, or other architecture
        pass
```

**2. Attribute Diffusion Network:**
```python
class AttributeDiffusionNetwork(nn.Module):
    def __init__(self, input_dim, time_embed_dim, config):
        super().__init__()
        self.time_embedding = nn.Linear(time_embed_dim, time_embed_dim)
        # Attribute-specific processing layers
        self.attribute_processor = nn.Sequential(
            nn.Linear(input_dim + time_embed_dim, config['hidden_dim']),
            nn.ReLU(),
            # Additional layers based on config
        )

    def forward(self, x, timesteps):
        # Process attributes with time conditioning
        time_embed = self.time_embedding(timesteps)
        combined = torch.cat([x, time_embed], dim=-1)
        return self.attribute_processor(combined)
```

**3. Feature Diffusion Network:**
```python
class FeatureDiffusionNetwork(nn.Module):
    def __init__(self, feature_dim, time_embed_dim, config):
        super().__init__()
        self.time_embedding = nn.Linear(time_embed_dim, time_embed_dim)
        # Use LSTM/Transformer for sequential features
        self.feature_processor = nn.LSTM(
            input_size=feature_dim + time_embed_dim,
            hidden_size=config['hidden_dim'],
            num_layers=config['num_layers'],
            batch_first=True
        )

    def forward(self, x, timesteps):
        # Process sequential features with time conditioning
        batch_size, seq_len, feat_dim = x.shape
        time_embed = self.time_embedding(timesteps)
        # Expand time embedding to match sequence length
        time_embed_seq = time_embed.unsqueeze(1).expand(-1, seq_len, -1)
        combined = torch.cat([x, time_embed_seq], dim=-1)
        output, _ = self.feature_processor(combined)
        return output
```

### Integration with NetShare Architecture

#### Model Manager Integration

The `NetShareManager` would need minimal changes to support diffusion models:

```python
# In netshare/model_managers/netshare_manager/netshare_manager.py
def _train(self, input_train_data_folder, output_model_folder, log_folder,
           create_new_model, model_config):
    # The training workflow remains largely the same
    # Only the model implementation changes
    configs = _load_config(
        config_dict={
            **self._config,
            **model_config},
        input_train_data_folder=input_train_data_folder,
        output_model_folder=output_model_folder)

    configs, config_group_list = _configs2configsgroup(
        configs=configs,
        generation_flag=False)

    # Training logic remains the same, model implementation changes
    pass
```

#### Training Process for Diffusion Models

**Forward Diffusion Process:**
```python
def add_noise(self, original_data, timesteps):
    """Add noise to data according to the forward diffusion process"""
    sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
    sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod).sqrt()

    noise = torch.randn_like(original_data)
    sqrt_alpha_t = sqrt_alphas_cumprod[timesteps].view(-1, 1, 1)
    sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1)

    return sqrt_alpha_t * original_data + sqrt_one_minus_alpha_t * noise, noise
```

**Training Loop:**
```python
def _train(self, input_train_data_folder, output_model_folder, log_folder):
    # Load preprocessed data
    data_feature, data_attribute, data_gen_flag, data_feature_outputs, data_attribute_outputs = load_data(...)

    optimizer = torch.optim.Adam(self.denoise_network.parameters(), lr=self.config['lr'])

    for epoch in range(self.config['epochs']):
        for batch_idx, (batch_features, batch_attributes) in enumerate(dataloader):
            # Sample random timesteps
            timesteps = torch.randint(0, self.timesteps, (batch_features.size(0),), device=batch_features.device)

            # Add noise to data
            noisy_features, target_features_noise = self.add_noise(batch_features, timesteps)
            noisy_attributes, target_attributes_noise = self.add_noise(batch_attributes, timesteps)

            # Predict noise
            pred_features_noise = self.denoise_network(batch_features, timesteps, batch_attributes)
            pred_attributes_noise = self.attribute_network(batch_attributes, timesteps)

            # Compute loss (MSE between predicted and actual noise)
            feature_loss = F.mse_loss(pred_features_noise, target_features_noise)
            attribute_loss = F.mse_loss(pred_attributes_noise, target_attributes_noise)

            total_loss = feature_loss + attribute_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
```

### Configuration Changes

The configuration system would need to support diffusion-specific parameters:

```json
{
  "model": {
    "class": "DiffusionModel",
    "config": {
      "timesteps": 1000,
      "beta_start": 1e-4,
      "beta_end": 0.02,
      "batch_size": 64,
      "epochs": 100,
      "lr": 2e-4,
      "network_config": {
        "hidden_dim": 256,
        "num_layers": 4,
        "time_embed_dim": 128
      }
    }
  }
}
```

### Generation Process with Diffusion Models

**Reverse Diffusion Process:**
```python
@torch.no_grad()
def generate(self, num_samples, conditioning_data=None):
    # Start with random noise
    shape = (num_samples, self.feature_dim, self.sequence_length)
    x = torch.randn(shape, device=self.device)

    # Iteratively denoise
    for i in range(self.timesteps - 1, -1, -1):
        t = torch.full((num_samples,), i, device=self.device, dtype=torch.long)

        # Predict noise
        predicted_noise = self.denoise_network(x, t, conditioning_data)

        # Apply reverse diffusion step
        x = self._reverse_diffusion_step(x, predicted_noise, t)

    return x

def _reverse_diffusion_step(self, x, predicted_noise, t):
    # Apply the reverse diffusion formula
    # This involves computing the mean and variance based on the model's prediction
    pass
```

### Pre/Post Processing Considerations

The pre/post processing pipeline remains largely unchanged since diffusion models work with the same data format as GANs:

**Normalization Compatibility:**
- Diffusion models work well with normalized data (typically [-1, 1] or [0, 1])
- Existing normalization strategies in NetShare are compatible
- Self-normalization approach can be maintained

**Temporal Dependencies:**
- Diffusion models can capture temporal dependencies through attention mechanisms
- LSTM-based architectures can maintain sequential modeling
- Generation flags can be handled similarly to GAN approach

### Implementation Steps

**Step 1: Create Diffusion Model Class**
- Extend the base `Model` class
- Implement forward/reverse diffusion processes
- Create attribute and feature diffusion networks

**Step 2: Integrate with Model Manager**
- Ensure compatibility with existing training/generation workflows
- Maintain distributed computing support through Ray

**Step 3: Update Configuration System**
- Add diffusion-specific parameters
- Maintain backward compatibility with GAN models

**Step 4: Testing and Validation**
- Verify that diffusion model produces realistic network traces
- Compare quality metrics with GAN baseline
- Ensure distributed generation pipeline works correctly

### Advantages of Diffusion Models

**Quality Improvements:**
- Better sample diversity compared to GANs
- More stable training without mode collapse
- Deterministic generation process

**Architecture Benefits:**
- Variational lower bound provides training objective
- No need for discriminator networks
- Better handling of multi-modal distributions

**Practical Considerations:**
- Potentially longer generation time due to iterative process
- Higher memory requirements during training
- More hyperparameters to tune (timesteps, noise schedule)

### Challenges and Solutions

**Computational Complexity:**
- Diffusion models require multiple forward passes for generation
- Solution: Use accelerated sampling techniques (DDIM, PLMS)

**Temporal Modeling:**
- Maintaining temporal consistency in network traces
- Solution: Use attention mechanisms and temporal conditioning

**Conditional Generation:**
- Generating features conditioned on attributes
- Solution: Use cross-attention between attribute and feature networks

By following this guide, you can successfully replace the GAN model in NetShare with a diffusion model while maintaining the existing architecture's strengths in distributed computing, preprocessing, and postprocessing.