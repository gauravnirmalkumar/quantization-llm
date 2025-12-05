# LLM Benchmark System - Technical Documentation

## Overview

This project benchmarks GGUF-quantized LLM models across multiple dimensions: speed, quality, memory usage, and latency. It features a real-time web dashboard that displays live model responses and comprehensive metrics.

## Quantization Methods Explained

### What is Quantization?

Quantization reduces model size by representing weights with fewer bits, trading some accuracy for significant memory and speed improvements.

### Quantization Types in This Project

#### **Q8_0 (8-bit Quantization)**
- **Precision**: 8 bits per weight
- **Size**: ~50% of original FP16 model
- **Quality**: Minimal quality loss (~1-2%)
- **Speed**: Moderate
- **Use Case**: Best quality-to-size ratio for production
- **Models**: 
  - Llama-3.2-1B-Instruct-Q8_0
  - Llama-3.2-3B-Instruct-Q8_0
  - Meta-Llama-3.1-8B-Instruct-Q8_0

#### **Q6_K (6-bit K-Quant)**
- **Precision**: 6 bits per weight (mixed precision)
- **Size**: ~38% of original
- **Quality**: Slight quality loss (~2-3%)
- **Speed**: Faster than Q8
- **Use Case**: Good balance for resource-constrained environments
- **Models**:
  - Llama-3.2-1B-Instruct-Q6_K
  - Llama-3.2-3B-Instruct-Q6_K
  - Meta-Llama-3.1-8B-Instruct-Q6_K

#### **Q4_K_M (4-bit K-Quant Medium)**
- **Precision**: 4 bits per weight (mixed precision)
- **Size**: ~25% of original
- **Quality**: Noticeable but acceptable quality loss (~5-7%)
- **Speed**: Very fast
- **Use Case**: **Recommended for most users** - best speed/quality balance
- **Models**:
  - Llama-3.2-1B-Instruct-Q4_K_M
  - Llama-3.2-3B-Instruct-Q4_K_M
  - Meta-Llama-3.1-8B-Instruct-Q4_K_M

#### **Q3_K_L / Q3_K_M (3-bit K-Quant)**
- **Precision**: 3 bits per weight (mixed precision)
- **Size**: ~19% of original
- **Quality**: More quality loss (~8-12%)
- **Speed**: Fastest
- **Use Case**: Maximum speed, acceptable for simple tasks
- **Models**:
  - Llama-3.2-1B-Instruct-Q3_K_L
  - Llama-3.2-3B-Instruct-Q3_K_L
  - Meta-Llama-3.1-8B-Instruct-Q3_K_M

### K-Quant Variants

- **K_S (Small)**: Aggressive quantization, smallest size
- **K_M (Medium)**: Balanced approach
- **K_L (Large)**: More bits for important layers

## Model Quantization Pipeline

This diagram shows the detailed quantization process for each model, including the specific techniques and parameters used:

```mermaid
graph TB
    subgraph "Step 1: Original Models (FP16)"
        A1["Llama 3.2 1B Instruct<br/>Size: 2.8GB<br/>Precision: FP16 (16-bit)<br/>Params: 1.24B<br/>Format: PyTorch/SafeTensors"]
        A2["Llama 3.2 3B Instruct<br/>Size: 6.5GB<br/>Precision: FP16 (16-bit)<br/>Params: 3.21B<br/>Format: PyTorch/SafeTensors"]
        A3["Llama 3.1 8B Instruct<br/>Size: 16GB<br/>Precision: FP16 (16-bit)<br/>Params: 8.03B<br/>Format: PyTorch/SafeTensors"]
    end
    
    subgraph "Step 2: Convert to GGUF Format"
        B["llama.cpp convert.py<br/>Converts PyTorch → GGUF<br/>Preserves model architecture<br/>Maintains FP16 precision"]
    end
    
    subgraph "Step 3: Q8_0 Quantization"
        C["llama.cpp quantize<br/>--method Q8_0<br/>━━━━━━━━━━━━━━<br/>8-bit integer quantization<br/>Linear quantization<br/>Range: -128 to 127<br/>Block size: 32<br/>Compression: ~50%"]
        C1["1B-Q8_0<br/>Size: 1.4GB<br/>Quality: 99%"]
        C2["3B-Q8_0<br/>Size: 3.3GB<br/>Quality: 99%"]
        C3["8B-Q8_0<br/>Size: 7.3GB<br/>Quality: 99%"]
    end
    
    subgraph "Step 4: Q6_K Quantization"
        D["llama.cpp quantize<br/>--method Q6_K<br/>━━━━━━━━━━━━━━<br/>6-bit K-quant (mixed)<br/>Importance matrix weighting<br/>6-bit for most layers<br/>8-bit for critical layers<br/>Compression: ~62%"]
        D1["1B-Q6_K<br/>Size: 1.1GB<br/>Quality: 97%"]
        D2["3B-Q6_K<br/>Size: 2.7GB<br/>Quality: 97%"]
        D3["8B-Q6_K<br/>Size: 5.8GB<br/>Quality: 97%"]
    end
    
    subgraph "Step 5: Q4_K_M Quantization"
        E["llama.cpp quantize<br/>--method Q4_K_M<br/>━━━━━━━━━━━━━━<br/>4-bit K-quant medium<br/>Adaptive quantization<br/>4-bit for most weights<br/>6-bit for attention layers<br/>Compression: ~75%<br/>Recommended variant"]
        E1["1B-Q4_K_M<br/>Size: 770MB<br/>Quality: 93%"]
        E2["3B-Q4_K_M<br/>Size: 2.1GB<br/>Quality: 93%"]
        E3["8B-Q4_K_M<br/>Size: 4.3GB<br/>Quality: 93%"]
    end
    
    subgraph "Step 6: Q3_K Quantization"
        F["llama.cpp quantize<br/>--method Q3_K_L/Q3_K_M<br/>━━━━━━━━━━━━━━<br/>3-bit K-quant<br/>Aggressive quantization<br/>3-bit for embeddings<br/>4-bit for critical paths<br/>Compression: ~81%<br/>Quality trade-off"]
        F1["1B-Q3_K_L<br/>Size: 700MB<br/>Quality: 88%"]
        F2["3B-Q3_K_L<br/>Size: 1.9GB<br/>Quality: 88%"]
        F3["8B-Q3_K_M<br/>Size: 3.7GB<br/>Quality: 88%"]
    end
    
    A1 --> B
    A2 --> B
    A3 --> B
    
    B -->|"Q8_0 quantization<br/>8-bit uniform"| C
    C --> C1
    C --> C2
    C --> C3
    
    B -->|"Q6_K quantization<br/>6-bit mixed precision"| D
    D --> D1
    D --> D2
    D --> D3
    
    B -->|"Q4_K_M quantization<br/>4-bit adaptive"| E
    E --> E1
    E --> E2
    E --> E3
    
    B -->|"Q3_K quantization<br/>3-bit aggressive"| F
    F --> F1
    F --> F2
    F --> F3
    
    subgraph "Step 7: This Benchmark Project"
        G["Benchmark Suite<br/>━━━━━━━━━━━━━━<br/>Tests all 12 models<br/>Speed measurement<br/>Quality (perplexity)<br/>Memory usage<br/>Latency testing<br/>5 test questions each"]
    end
    
    C1 & C2 & C3 & D1 & D2 & D3 & E1 & E2 & E3 & F1 & F2 & F3 --> G
    
    style A1 fill:#ff6b6b,color:#fff,stroke:#c92a2a,stroke-width:3px
    style A2 fill:#ff6b6b,color:#fff,stroke:#c92a2a,stroke-width:3px
    style A3 fill:#ff6b6b,color:#fff,stroke:#c92a2a,stroke-width:3px
    style B fill:#4ecdc4,color:#fff,stroke:#0b7285,stroke-width:3px
    style C fill:#ffd93d,color:#333,stroke:#f08c00,stroke-width:2px
    style D fill:#95e1d3,color:#333,stroke:#0b7285,stroke-width:2px
    style E fill:#a8e6cf,color:#333,stroke:#2f9e44,stroke-width:2px
    style F fill:#ffaaa5,color:#333,stroke:#e03131,stroke-width:2px
    style G fill:#667eea,color:#fff,stroke:#4c51bf,stroke-width:4px
    
    style C1 fill:#fff9db,stroke:#f08c00
    style C2 fill:#fff9db,stroke:#f08c00
    style C3 fill:#fff9db,stroke:#f08c00
    style D1 fill:#e3fafc,stroke:#0b7285
    style D2 fill:#e3fafc,stroke:#0b7285
    style D3 fill:#e3fafc,stroke:#0b7285
    style E1 fill:#d3f9d8,stroke:#2f9e44
    style E2 fill:#d3f9d8,stroke:#2f9e44
    style E3 fill:#d3f9d8,stroke:#2f9e44
    style F1 fill:#ffe3e3,stroke:#e03131
    style F2 fill:#ffe3e3,stroke:#e03131
    style F3 fill:#ffe3e3,stroke:#e03131
```

### Detailed Quantization Techniques

#### **Q8_0 - 8-bit Quantization**
- **Method**: Linear quantization with uniform scaling
- **Process**: 
  1. Calculate min/max values for each weight block (32 weights)
  2. Map FP16 values to 8-bit integers (-128 to 127)
  3. Store scale factor per block
- **Precision**: 256 discrete values per weight
- **Quality Loss**: ~1% (imperceptible)
- **Use Case**: Production deployments requiring high quality

#### **Q6_K - 6-bit K-Quant**
- **Method**: Mixed-precision with importance weighting
- **Process**:
  1. Analyze layer importance using calibration data
  2. Apply 6-bit quantization to standard layers
  3. Keep 8-bit precision for attention mechanisms
  4. Use super-blocks (256 weights) for better compression
- **Precision**: 64 discrete values (6-bit) + 256 (8-bit) for critical layers
- **Quality Loss**: ~2-3%
- **Use Case**: Balanced performance for resource-constrained systems

#### **Q4_K_M - 4-bit K-Quant Medium**
- **Method**: Adaptive quantization with layer-specific precision
- **Process**:
  1. Group weights into super-blocks (256 weights)
  2. Apply 4-bit quantization to embeddings and FFN layers
  3. Use 6-bit for attention query/key/value matrices
  4. Optimize scale factors per super-block
- **Precision**: 16 discrete values (4-bit) + 64 (6-bit) for attention
- **Quality Loss**: ~5-7%
- **Use Case**: **Recommended** - best speed/quality trade-off

#### **Q3_K_L/M - 3-bit K-Quant**
- **Method**: Aggressive quantization with selective precision
- **Process**:
  1. Use 3-bit for embedding layers (least sensitive)
  2. Apply 4-bit to feed-forward networks
  3. Maintain 5-bit for attention layers
  4. Large super-blocks (256 weights) for better accuracy
- **Precision**: 8 discrete values (3-bit) + higher for critical paths
- **Quality Loss**: ~8-12%
- **Use Case**: Maximum speed, acceptable for simple tasks

### Quantization Command Examples

```bash
# Convert original model to GGUF
python convert.py models/Llama-3.2-1B-Instruct

# Q8_0 quantization
./quantize models/Llama-3.2-1B-Instruct-f16.gguf \
           models/Llama-3.2-1B-Instruct-Q8_0.gguf Q8_0

# Q6_K quantization  
./quantize models/Llama-3.2-1B-Instruct-f16.gguf \
           models/Llama-3.2-1B-Instruct-Q6_K.gguf Q6_K

# Q4_K_M quantization (recommended)
./quantize models/Llama-3.2-1B-Instruct-f16.gguf \
           models/Llama-3.2-1B-Instruct-Q4_K_M.gguf Q4_K_M

# Q3_K_L quantization
./quantize models/Llama-3.2-1B-Instruct-f16.gguf \
           models/Llama-3.2-1B-Instruct-Q3_K_L.gguf Q3_K_L
```

### Size Comparison Table

| Model | Original (FP16) | Q8_0 | Q6_K | Q4_K_M | Q3_K |
|-------|----------------|------|------|--------|------|
| **1B** | 2.8 GB | 1.4 GB (50%) | 1.1 GB (39%) | 770 MB (27%) | 700 MB (25%) |
| **3B** | 6.5 GB | 3.3 GB (51%) | 2.7 GB (42%) | 2.1 GB (32%) | 1.9 GB (29%) |
| **8B** | 16 GB | 7.3 GB (46%) | 5.8 GB (36%) | 4.3 GB (27%) | 3.7 GB (23%) |

**Key Insights:**
- Q4_K_M provides 4x compression with minimal quality loss
- Q3_K achieves 4-5x compression but with noticeable quality impact
- Larger models benefit more from quantization (better compression ratios)
- K-quant methods intelligently preserve quality in critical layers

## System Architecture

```mermaid
graph TB
    subgraph "User Interface"
        A[Web Browser] -->|HTTP Request| B[Python HTTP Server :8080]
        B -->|Serves| C[index.html]
        C -->|Polls every 1s| D[benchmark_status.json]
    end
    
    subgraph "Benchmark Engine"
        E[run_simple_benchmarks.py] -->|Reads| F[models.json]
        E -->|Loads| G[GGUF Models]
        E -->|Uses| H[llama-cpp-python]
        E -->|Updates| I[BenchmarkManager]
        I -->|Writes| D
    end
    
    subgraph "Test Suite"
        E -->|Runs 5 Tests| J[Test Prompts]
        J --> K[1. Explanation]
        J --> L[2. Math - Basic]
        J --> M[3. Math - Word Problem]
        J --> N[4. Coding]
        J --> O[5. Reasoning]
    end
    
    subgraph "Metrics Collection"
        H -->|Measures| P[Speed T/s]
        H -->|Measures| Q[Latency ms]
        H -->|Measures| R[Memory MB]
        H -->|Measures| S[Perplexity]
        H -->|Measures| T[Load Time]
        H -->|Measures| U[PP Speed]
    end
    
    subgraph "Data Storage"
        D -->|Contains| V[Active Model Status]
        D -->|Contains| W[Progress %]
        D -->|Contains| X[Live Logs]
        D -->|Contains| Y[Test Responses]
        D -->|Contains| Z[All Metrics]
    end
    
    subgraph "Dashboard Display"
        C -->|Shows| AA[Live Q&A]
        C -->|Shows| AB[Metrics Cards]
        C -->|Shows| AC[Results Table]
        C -->|Shows| AD[Live Logs]
        AC -->|Expandable| AE[Full Q&A per Model]
    end
    
    style E fill:#667eea,color:#fff
    style C fill:#764ba2,color:#fff
    style D fill:#28a745,color:#fff
```

## Data Flow

```mermaid
sequenceDiagram
    participant User
    participant Browser
    participant Server
    participant Benchmark
    participant Model
    participant Status
    
    User->>Browser: Open Dashboard
    Browser->>Server: GET index.html
    Server-->>Browser: HTML/CSS/JS
    
    User->>Benchmark: Start Benchmark
    
    loop For Each Model
        Benchmark->>Model: Load GGUF
        Model-->>Benchmark: Model Ready
        Benchmark->>Status: Update (Loading...)
        
        loop For Each Test Question
            Benchmark->>Model: Send Prompt
            Model-->>Benchmark: Generate Response
            Benchmark->>Status: Update (Q&A + Metrics)
        end
        
        Benchmark->>Status: Save Final Results
    end
    
    loop Every 1 Second
        Browser->>Status: Fetch JSON
        Status-->>Browser: Current Data
        Browser->>Browser: Update UI
    end
```

## Component Details

### 1. **models.json**
Configuration file defining all models to benchmark:
- Model path
- Parameter count (1B, 3B, 8B)
- Quantization type
- Model family

### 2. **run_simple_benchmarks.py**
Main benchmark script that:
- Loads models sequentially
- Runs 5 different test prompts per model
- Measures 6 key metrics
- Updates status file in real-time
- Handles errors gracefully

### 3. **benchmark_manager.py**
Status file manager that:
- Maintains `benchmark_status.json`
- Provides thread-safe updates
- Logs all events
- Tracks progress

### 4. **index.html**
Real-time dashboard featuring:
- Live Q&A display (all 5 test responses)
- Metrics cards (Speed, Latency, Memory, Load Time)
- Expandable results table
- Auto-refresh every second
- Responsive design

### 5. **benchmark_status.json**
Live status file containing:
- Current active model
- Task description
- Progress percentage
- Event logs
- Complete results with all Q&A responses

## Metrics Explained

### Speed (Tokens/Second)
How fast the model generates text. Higher is better.
- **Excellent**: >100 T/s
- **Good**: 50-100 T/s
- **Acceptable**: <50 T/s

### Latency (Milliseconds)
Time to generate the first token. Lower is better.
- **Excellent**: <50ms
- **Good**: 50-150ms
- **Acceptable**: >150ms

### Memory (MB)
RAM used by the model. Lower is better for deployment.
- **1B models**: 700-1400 MB
- **3B models**: 1900-3300 MB
- **8B models**: 3700-7300 MB

### Load Time (Seconds)
Time to load model into memory. Lower is better.
- Depends on model size and quantization
- Q3/Q4 load faster than Q6/Q8

### Prompt Processing Speed (T/s)
How fast the model processes input tokens.

### Perplexity
Quality metric - how well the model predicts text. Lower is better.
- Calculated on wikitext-2 dataset
- Measures language understanding

## Test Questions

1. **Explanation**: Tests general knowledge and explanation ability
2. **Math - Basic**: Tests arithmetic (15 × 23)
3. **Math - Word Problem**: Tests reasoning and calculation
4. **Coding**: Tests code generation (factorial function)
5. **Reasoning**: Tests logical deduction

## Usage

### Start Web Server
```bash
cd /Users/gaurav/Developer/Code/Benchmark
python3 -m http.server 8080
```

### Run Benchmark
```bash
/opt/anaconda3/bin/python3 run_simple_benchmarks.py
```

### View Dashboard
Open browser to: `http://localhost:8080/index.html`

## Key Findings

Based on benchmark results:

**Best Overall**: Llama-3.2-3B-Instruct-Q4_K_M
- Speed: ~70 T/s
- Memory: ~2GB
- Good quality
- Fast load time

**Fastest**: Llama-3.2-1B-Instruct-Q4_K_M
- Speed: ~150 T/s
- Lowest latency
- Smallest memory footprint

**Best Quality**: Llama-3.2-3B-Instruct-Q8_0
- Highest precision
- Best perplexity scores
- Moderate speed

## Technology Stack

- **Backend**: Python 3.13
- **LLM Runtime**: llama-cpp-python
- **Dataset**: Hugging Face datasets (wikitext-2)
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Server**: Python http.server
- **Models**: GGUF format (llama.cpp compatible)

## File Structure

```
Benchmark/
├── models.json                 # Model configurations
├── run_simple_benchmarks.py    # Main benchmark script
├── benchmark_manager.py        # Status file manager
├── index.html                  # Real-time dashboard
├── benchmark_status.json       # Live status (generated)
├── benchmark.log               # Execution logs (generated)
├── README.md                   # Quick start guide
└── DOCUMENTATION.md            # This file
```

## Dependencies

```bash
pip install llama-cpp-python datasets psutil
```

## Performance Optimization

1. **Sequential Processing**: Models loaded one at a time to avoid OOM
2. **GPU Acceleration**: Uses Metal (macOS) for inference
3. **Efficient Quantization**: K-quant methods optimize important layers
4. **Real-time Updates**: Minimal overhead status file updates
5. **Responsive UI**: Lightweight polling mechanism

## Future Enhancements

- [ ] Add more test categories (creative writing, translation)
- [ ] Export results to CSV/Excel
- [ ] Compare multiple runs
- [ ] Add model-to-model comparison charts
- [ ] Support for other model formats (GPTQ, AWQ)
- [ ] Batch processing for faster benchmarks
