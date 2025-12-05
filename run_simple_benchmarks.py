import json
import time
import math
import os
import sys
import psutil
from datasets import load_dataset
from llama_cpp import Llama
from benchmark_manager import BenchmarkManager

def load_models():
    with open("models.json", "r") as f:
        return json.load(f)

def get_memory_usage():
    """Get current process memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def main():
    manager = BenchmarkManager()
    manager.log("Starting Enhanced Benchmark Suite with Multiple Test Questions...")
    
    # Multiple test prompts for different capabilities
    test_prompts = [
        {
            "category": "Explanation",
            "prompt": "Explain quantum computing in simple terms:",
            "max_tokens": 150
        },
        {
            "category": "Math - Basic",
            "prompt": "What is 15 * 23? Show your work step by step.",
            "max_tokens": 100
        },
        {
            "category": "Math - Word Problem",
            "prompt": "If a train travels 120 km in 2 hours, what is its average speed in km/h?",
            "max_tokens": 80
        },
        {
            "category": "Coding",
            "prompt": "Write a Python function to calculate the factorial of a number:",
            "max_tokens": 120
        },
        {
            "category": "Reasoning",
            "prompt": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
            "max_tokens": 100
        }
    ]
    
    # Load Dataset for perplexity
    manager.update_status(task="Loading Dataset...", progress=5)
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n".join(dataset["text"][:10])
        manager.log(f"Loaded wikitext ({len(text)} chars)")
    except Exception as e:
        manager.log(f"Error loading dataset: {e}")
        return

    models = load_models()
    total_models = len(models)
    
    for idx, model in enumerate(models):
        model_name = os.path.basename(model['path']).replace('.gguf', '')
        manager.update_status(model=model_name, task="Loading Model...", progress=10)
        manager.log(f"Processing {model_name}...")
        
        try:
            # Measure memory before loading
            mem_before = get_memory_usage()
            
            # Load Model
            load_start = time.time()
            llm = Llama(
                model_path=model['path'],
                n_gpu_layers=99,
                n_ctx=2048,
                verbose=False,
                logits_all=True
            )
            load_time = time.time() - load_start
            mem_after = get_memory_usage()
            model_size_mb = mem_after - mem_before
            
            manager.log(f"Model loaded in {load_time:.2f}s, using {model_size_mb:.0f}MB")
            
            # Test all prompts and collect responses
            all_responses = []
            total_tokens = 0
            total_time = 0
            
            for test_idx, test in enumerate(test_prompts):
                progress = 20 + (test_idx * 15)
                manager.update_status(task=f"Testing: {test['category']}...", progress=progress)
                
                # Stream tokens for real-time display
                manager.log(f"Q[{test['category']}]: {test['prompt'][:60]}...")
                
                start_time = time.time()
                response_text = ""
                token_count = 0
                
                # Create streaming generator
                stream = llm.create_completion(
                    prompt=test['prompt'],
                    max_tokens=test['max_tokens'],
                    temperature=0.7,
                    stream=True
                )
                
                # Stream tokens and update status
                for output in stream:
                    if 'choices' in output and len(output['choices']) > 0:
                        token = output['choices'][0].get('text', '')
                        if token:
                            response_text += token
                            token_count += 1
                            
                            # Update streaming status every few tokens
                            if token_count % 5 == 0:
                                manager.update_status(
                                    task=f"Generating: {test['category']} ({token_count} tokens)...",
                                    progress=progress
                                )
                                # Log partial response for live display
                                preview = response_text[-100:] if len(response_text) > 100 else response_text
                                manager.log(f"STREAM: ...{preview}")
                
                elapsed = time.time() - start_time
                
                total_tokens += token_count
                total_time += elapsed
                
                # Log final response
                manager.log(f"A: {response_text[:100]}...")
                
                all_responses.append({
                    "category": test['category'],
                    "prompt": test['prompt'],
                    "response": response_text,
                    "tokens": token_count,
                    "time_s": round(elapsed, 2)
                })

            
            # Calculate average speed
            avg_speed = total_tokens / total_time if total_time > 0 else 0
            
            # Test latency (Time to First Token)
            manager.update_status(task="Measuring Latency...", progress=85)
            start_ttft = time.time()
            llm.create_completion(prompt="Hello", max_tokens=1)
            ttft = time.time() - start_ttft
            
            # Test Perplexity
            manager.update_status(task="Calculating Perplexity...", progress=90)
            output_with_logprobs = llm.create_completion(
                prompt=text[:500],
                max_tokens=128,
                logprobs=1,
                echo=True
            )
            
            token_logprobs = output_with_logprobs['choices'][0]['logprobs']['token_logprobs']
            valid_logprobs = [lp for lp in token_logprobs if lp is not None]
            if valid_logprobs:
                avg_logprob = sum(valid_logprobs) / len(valid_logprobs)
                perplexity = math.exp(-avg_logprob)
            else:
                perplexity = 0.0
            
            # Test Prompt Processing Speed
            manager.update_status(task="Testing Prompt Processing...", progress=95)
            long_prompt = text[:1000]
            pp_start = time.time()
            llm.create_completion(prompt=long_prompt, max_tokens=1)
            pp_time = time.time() - pp_start
            pp_tokens = len(long_prompt.split())
            pp_speed = pp_tokens / pp_time if pp_time > 0 else 0
            
            # Save Result with all metrics and responses
            result = {
                "name": model_name,
                "params": model['params'],
                "quant": model['quant'],
                "speed": round(avg_speed, 2),
                "perplexity": round(perplexity, 2),
                "latency_ms": round(ttft * 1000, 2),
                "memory_mb": round(model_size_mb, 0),
                "load_time_s": round(load_time, 2),
                "pp_speed": round(pp_speed, 2),
                "test_responses": all_responses,
                "sample_prompt": all_responses[0]['prompt'],
                "sample_response": all_responses[0]['response']
            }
            
            manager.add_result(result)
            manager.log(f"✓ {model_name}: Speed={avg_speed:.1f} T/s, PPL={perplexity:.1f}, Latency={ttft*1000:.0f}ms, Mem={model_size_mb:.0f}MB")
            
            # Cleanup
            del llm
            
        except Exception as e:
            manager.log(f"Error benchmarking {model_name}: {e}")
            import traceback
            manager.log(f"Traceback: {traceback.format_exc()[:200]}")
        
        # Update Progress
        overall_progress = int(((idx + 1) / total_models) * 100)
        manager.update_status(progress=overall_progress)
        time.sleep(1)

    manager.update_status(model="All Done", task="Completed", progress=100)
    manager.log("All benchmarks completed with multiple test questions.")

if __name__ == "__main__":
    main()
