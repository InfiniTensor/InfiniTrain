import subprocess
import re
import time

# Configuration
GPT2_BIN = "./build/gpt2"
GPT2_INPUT_BIN = "/data/shared/InfiniTrain-dev/data/llmc/gpt2/tinyshakespeare/tiny_shakespeare_train.bin"
GPT2_TOKENIZER_BIN = "/data/shared/InfiniTrain-dev/data/llmc/gpt2/gpt2_tokenizer.bin"

LLAMA3_BIN = "./build/llama3"
LLAMA3_INPUT_BIN = "/data/shared/InfiniTrain-dev/data/llmc/llama3/tinyshakespeare/tiny_shakespeare_train.bin"
LLAMA3_TOKENIZER_BIN = "/data/shared/InfiniTrain-dev/data/llmc/llama3/llama3_tokenizer.bin"
LLAMA3_LLMC_BIN = "/data/shared/InfiniTrain-dev/data/llmc/llama3/llama3.2_1B_fp32.bin"

OUTPUT_FILE = "performance_report.md"

def run_experiment(name, model_type, flash, seq_len, num_steps=20, batch_size=4):
    print(f"Running {name}...")
    total_batch_size = batch_size * seq_len
    
    if model_type == "gpt2":
        cmd = [
            GPT2_BIN,
            f"-input_bin={GPT2_INPUT_BIN}",
            f"-tokenizer_bin={GPT2_TOKENIZER_BIN}",
            f"-flash={str(flash).lower()}",
            f"-sequence_length={seq_len}",
            f"-num_iteration={num_steps}",
            f"-batch_size={batch_size}",
            f"-total_batch_size={total_batch_size}",
            "-model=d12",  # GPT-2 124M equivalent
            "-overfit_single_batch=false", # Use real data
            "-freq_generate_txt=9999" # Disable text generation to prevent OOM/overhead
        ]
    elif model_type == "llama3":
         cmd = [
            LLAMA3_BIN,
            f"-input_bin={LLAMA3_INPUT_BIN}",
            f"-tokenizer_bin={LLAMA3_TOKENIZER_BIN}",
            f"-llmc_filepath={LLAMA3_LLMC_BIN}",
            f"-flash={str(flash).lower()}",
            f"-sequence_length={seq_len}",
            f"-num_iteration={num_steps}",
            f"-batch_size={batch_size}",
            f"-total_batch_size={total_batch_size}",
            "-overfit_single_batch=false", # Use real data
            "-freq_generate_txt=9999" # Disable text generation to prevent OOM/overhead
        ]
    else:
        print(f"Unknown model type: {model_type}")
        return None
    
    print("Command:", " ".join(cmd))
    
    start_time = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    all_output = []
    losses = []
    throughputs = [] # tok/s
    mem_usages = [] # MB
    
    # Regex for log line:
    # step 1/10 | train loss 5.358501 | lr 1.00e-04 | (123.45 ms | 8320 tok/s | peak used: 1234 MB ...
    pattern = re.compile(r"step\s+(\d+)/\d+\s+\|\s+train loss\s+([0-9.]+)\s+.*\|\s+([0-9.]+)\s+tok/s\s+\|\s+peak used:\s+(\d+)\s+MB")
    
    for line in proc.stdout:
        line = line.rstrip("\n")
        all_output.append(line)
        match = pattern.search(line)
        if match:
            step = int(match.group(1))
            loss = float(match.group(2))
            tps = float(match.group(3))
            mem = int(match.group(4))
            losses.append(loss)
            throughputs.append(tps)
            mem_usages.append(mem)
            print(f"[{name}] step={step} loss={loss:.6f} tps={tps:.0f} peak_mem={mem}MB")

    returncode = proc.wait()
    end_time = time.time()

    if returncode != 0:
        print(f"Error running {name}:")
        print("\n".join(all_output[-50:]))
        return None

    print(f"Completed {name} in {end_time - start_time:.2f}s")
    
    return {
        "losses": losses,
        "throughputs": throughputs,
        "mem_usages": mem_usages,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_steps": num_steps,
        "avg_tps": sum(throughputs[5:]) / len(throughputs[5:]) if len(throughputs) > 5 else 0, # Skip first 5 steps
        "peak_mem": max(mem_usages) if mem_usages else 0
    }

def generate_report(results):
    with open(OUTPUT_FILE, "w") as f:
        f.write("# FlashAttention Integration Performance Report\n\n")
        
        # AC1: Precision Alignment
        f.write("## 1. Precision Alignment (AC1)\n")
        base = results.get("Baseline-1024")
        flash = results.get("Flash-1024")
        steps_1024 = min(len(base["losses"]) if base else 0, len(flash["losses"]) if flash else 0)
        f.write(f"Comparing Training Loss for first {steps_1024} steps (SeqLen=1024).\n\n")
        
        if base and flash:
            f.write("| Step | Baseline Loss | Flash Loss | Diff |\n")
            f.write("|---|---|---|---|\n")
            max_diff = 0
            for i in range(min(len(base['losses']), len(flash['losses']))):
                diff = abs(base['losses'][i] - flash['losses'][i])
                max_diff = max(max_diff, diff)
                if i < 10 or i % 10 == 0: # Print first 10 and every 10th
                    f.write(f"| {i+1} | {base['losses'][i]:.6f} | {flash['losses'][i]:.6f} | {diff:.6e} |\n")
            f.write(f"\n**Max Difference**: {max_diff:.6e}\n")
            if max_diff < 1e-4:
                 f.write("**Status**: PASS (Difference within expected floating point error)\n")
            else:
                 f.write("**Status**: WARNING (Difference > 1e-4)\n")
        else:
            f.write("Missing data for comparison.\n")

        # AC2: Performance
        f.write("\n## 2. Performance Comparison (AC2)\n")
        f.write("| Configuration | Seq Len | Batch Size | Avg Tokens/s | Peak Memory (MB) | Speedup |\n")
        f.write("|---|---|---|---|---|---|\n")
        
        configs = [
            ("Baseline-1024", "Flash-1024", 1024),
            ("Baseline-2048", "Flash-2048", 2048),
            ("LLaMA3-Base-1024", "LLaMA3-Flash-1024", 1024)
        ]
        
        for base_key, flash_key, seq_len in configs:
            b_res = results.get(base_key)
            f_res = results.get(flash_key)
            
            if b_res:
                f.write(f"| Baseline | {seq_len} | {b_res['batch_size']} | {b_res['avg_tps']:.0f} | {b_res['peak_mem']} | 1.0x |\n")
            if f_res:
                speedup = f_res['avg_tps'] / b_res['avg_tps'] if b_res and b_res['avg_tps'] > 0 else 0
                f.write(f"| FlashAttn | {seq_len} | {f_res['batch_size']} | {f_res['avg_tps']:.0f} | {f_res['peak_mem']} | {speedup:.2f}x |\n")

if __name__ == "__main__":
    results = {}
    
    # 1. GPT-2 Seq Len 1024
    results["Baseline-1024"] = run_experiment("Baseline-1024", "gpt2", flash=False, seq_len=1024, num_steps=20, batch_size=2)
    results["Flash-1024"] = run_experiment("Flash-1024", "gpt2", flash=True, seq_len=1024, num_steps=20, batch_size=2)
    
    # 2. GPT-2 Seq Len 2048 (AC2)
    results["Baseline-2048"] = run_experiment("Baseline-2048", "gpt2", flash=False, seq_len=2048, num_steps=20, batch_size=1)
    results["Flash-2048"] = run_experiment("Flash-2048", "gpt2", flash=True, seq_len=2048, num_steps=20, batch_size=1)

    # 3. LLaMA-3 Seq Len 1024 (AC3)
    results["LLaMA3-Base-1024"] = run_experiment("LLaMA3-Base-1024", "llama3", flash=False, seq_len=1024, num_steps=10, batch_size=1)
    results["LLaMA3-Flash-1024"] = run_experiment("LLaMA3-Flash-1024", "llama3", flash=True, seq_len=1024, num_steps=10, batch_size=1)
    
    generate_report(results)
    print(f"Report generated at {OUTPUT_FILE}")
