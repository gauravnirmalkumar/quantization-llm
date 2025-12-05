# How to Run the LLM Benchmark

## Quick Start

### 1. Start the Web Server
Open Terminal and run:
```bash
cd /Users/gaurav/Developer/Code/Benchmark
python3 -m http.server 8080
```
Keep this terminal window open.

### 2. View the Dashboard
Open your browser and go to:
```
http://localhost:8080/index.html
```

### 3. Run the Benchmark
Open a **new** Terminal window and run:
```bash
cd /Users/gaurav/Developer/Code/Benchmark
/opt/anaconda3/bin/python3 run_simple_benchmarks.py
```

The dashboard will automatically update every second showing:
- Current model being tested
- Progress percentage
- Live logs
- Results table

## What the Benchmark Does

1. **Loads Dataset**: Downloads wikitext-2 from Hugging Face (first time only)
2. **Tests Each Model**: For each of the 12 models:
   - Loads the model onto your GPU
   - Measures **Speed**: How fast it generates tokens (tokens/second)
   - Measures **Quality**: Perplexity score (lower = better)
3. **Updates Dashboard**: Real-time progress visible in browser

## Benchmark Takes About 5-10 Minutes

Each model takes 15-30 seconds to test.

## Files Created

- `benchmark_status.json` - Live status (updated during run)
- `benchmark.log` - Detailed logs

## To Stop

Press `Ctrl+C` in the terminal running the benchmark.

## Already Completed!

The benchmark already ran and completed all 12 models. The results are in `benchmark_status.json`.

Just open the dashboard at `http://localhost:8080/index.html` to see the results!
