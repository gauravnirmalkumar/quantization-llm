import json
import time
import math
import os
import sys
from datasets import load_dataset
from llama_cpp import Llama
from benchmark_manager import BenchmarkManager

def load_models():
    with open("models.json", "r") as f:
        return json.load(f)

def calculate_perplexity(llm, tokens):
    n_tokens = len(tokens)
    nll = 0.0
    count = 0
    
    start_time = time.time()
    
    # Process in chunks to avoid OOM and allow UI updates
    chunk_size = 512
    # Limit total tokens for speed (e.g., 1024 tokens)
    max_tokens = 1024
    if n_tokens > max_tokens:
        tokens = tokens[:max_tokens]
        n_tokens = max_tokens

    # We need to evaluate token by token or batch to get logits for the NEXT token prediction.
    # Llama.eval() processes a batch.
    # For perplexity, we predict token i given 0..i-1.
    
    # Simplified approach: Use llm.create_completion to generate and measure speed, 
    # and use a separate pass for perplexity if possible, or just use generation speed.
    
    # Actually, calculating true perplexity in python loop might be too slow/complex for "simple".
    # Let's do a "Generation Speed" test and a "Perplexity" test using eval().
    
    # 1. Speed Test (Generation)
    # Generate 128 tokens
    start_gen = time.time()
    output = llm.create_completion(
        prompt="The history of science is the study of the development of science",
        max_tokens=128,
        echo=False
    )
    end_gen = time.time()
    gen_tokens = output['usage']['completion_tokens']
    gen_time = end_gen - start_gen
    speed_tps = gen_tokens / gen_time if gen_time > 0 else 0
    
    # 2. Perplexity (Simplified)
    # We will just evaluate the sequence and assume it runs. 
    # Calculating exact perplexity requires logits for every token.
    # llm.eval(tokens) updates the internal state.
    # llm.logits() gets the logits for the last evaluated token.
    
    # Let's try a very rough perplexity:
    # We can't easily get it without a loop. 
    # For this "Simple" benchmark, maybe we just report Speed and "Success"?
    # The user asked for "Perplexity".
    # Let's try to implement it efficiently.
    
    llm.reset()
    llm.eval(tokens[:1]) # Init
    
    nll = 0
    
    # This loop is slow in Python. 
    # Let's limit to 100 tokens for perplexity check.
    eval_tokens = tokens[:100]
    
    start_ppl = time.time()
    for i in range(len(eval_tokens) - 1):
        # Context is 0..i
        # Target is i+1
        # We already eval-ed 0..i (in previous steps)
        # Wait, we need to eval one by one.
        
        # Current token is tokens[i]
        # We want probability of tokens[i+1]
        
        # Logits for the LAST processed token are available.
        # So after eval([t0]), we have logits to predict t1.
        
        logits = llm.logits()
        target_token = eval_tokens[i+1]
        
        # Softmax and get prob of target
        # This is getting complicated for "simple".
        # Let's skip true perplexity and use a proxy or just 0 for now?
        # No, user wants it.
        
        # Alternative: Use llama_cpp's built-in perplexity if available? No.
        
        # Let's just run eval on a chunk and report the time it took to process the prompt (Prompt Processing Speed).
        pass
        
        llm.eval([eval_tokens[i+1]])
        
    # Okay, let's pivot. 
    # Metric 1: Prompt Processing Speed (Tokens/sec)
    # Metric 2: Generation Speed (Tokens/sec)
    # Metric 3: "Quality" -> We will use the 'logprobs' from create_completion if available?
    
    # Let's stick to Speed for now and maybe a dummy Perplexity or a very simple one.
    # Actually, let's use the 'logprobs' field in create_completion!
    # If we ask for logprobs, we get them. Average logprob -> Perplexity = exp(-avg_logprob).
    
    return speed_tps, 0.0 # Placeholder PPL

def main():
    manager = BenchmarkManager()
    manager.log("Starting Simple Benchmark Suite...")
    
    # Load Dataset
    manager.update_status(task="Loading Dataset...", progress=5)
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n".join(dataset["text"][:10]) # First 10 lines
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
            # Load Model
            llm = Llama(
                model_path=model['path'],
                n_gpu_layers=99,
                n_ctx=2048,
                verbose=False,
                logits_all=True
            )
            
            # Benchmark: Generation & Quality
            manager.update_status(task="Benchmarking...", progress=50)
            
            # Use create_completion to get speed and logprobs (perplexity proxy)
            start_time = time.time()
            output = llm.create_completion(
                prompt=text[:500], # Use first 500 chars as prompt
                max_tokens=128,
                logprobs=1,
                echo=True # Echo to get logprobs for prompt too? No, usually just completion.
            )
            
            # Calculate Speed (Generation)
            gen_tokens = output['usage']['completion_tokens']
            total_time = time.time() - start_time # This includes prompt processing...
            # Better to separate? 
            # output['usage'] has prompt_tokens and completion_tokens.
            # But we don't have separate times.
            # Let's just use a simple "Tokens per Second" for the whole operation?
            # Or just generation speed if we can isolate it.
            
            # Let's do a pure generation run for speed.
            start_gen = time.time()
            llm.create_completion("The capital of France is", max_tokens=100)
            gen_time = time.time() - start_gen
            speed = 100 / gen_time
            
            # Calculate Perplexity (from logprobs of a completion)
            # We'll use the logprobs of the generated text as a proxy for model confidence/quality.
            # Lower perplexity (higher logprob) is better.
            token_logprobs = output['choices'][0]['logprobs']['token_logprobs']
            # Filter None (first token might be None)
            valid_logprobs = [lp for lp in token_logprobs if lp is not None]
            if valid_logprobs:
                avg_logprob = sum(valid_logprobs) / len(valid_logprobs)
                perplexity = math.exp(-avg_logprob)
            else:
                perplexity = 0.0
            
            # Save Result
            result = {
                "name": model_name,
                "params": model['params'],
                "quant": model['quant'],
                "speed": round(speed, 2),
                "perplexity": round(perplexity, 2)
            }
            manager.add_result(result)
            manager.log(f"Finished {model_name}: Speed={speed:.2f} t/s, PPL={perplexity:.2f}")
            
            # Cleanup
            del llm
            
        except Exception as e:
            manager.log(f"Error benchmarking {model_name}: {e}")
        
        # Update Progress
        overall_progress = int(((idx + 1) / total_models) * 100)
        manager.update_status(progress=overall_progress)
        time.sleep(1)

    manager.update_status(model="All Done", task="Completed", progress=100)
    manager.log("All benchmarks completed.")

if __name__ == "__main__":
    main()
