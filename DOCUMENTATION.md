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
