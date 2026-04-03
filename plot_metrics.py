import os
import re
import matplotlib.pyplot as plt

LOG_DIR = 'scripts/logs_flash'

def parse_log(filepath):
    steps = []
    losses = []
    memories = []
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return steps, losses, memories
    
    with open(filepath, 'r') as f:
        for line in f:
            # Example format: 
            # E20260315 22:08:49.372466 ... step 1/20 | train loss 4.372168 | ... peak used: 30023 MB
            match = re.search(r'step\s+(\d+)/.*\btrain loss\s+([\d\.]+).*?peak used:\s+(\d+)\s+MB', line)
            if match:
                steps.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                memories.append(int(match.group(3)))
    return steps, losses, memories

def plot_loss(model, seq):
    base_log = os.path.join(LOG_DIR, f'{model}_baseline_fp32_{seq}.log')
    flash_log = os.path.join(LOG_DIR, f'{model}_flash_fp32_{seq}.log')
    
    b_steps, b_losses, _ = parse_log(base_log)
    f_steps, f_losses, _ = parse_log(flash_log)
    
    if not b_steps or not f_steps:
        return
    
    plt.figure(figsize=(8, 5))
    plt.plot(b_steps, b_losses, label='Baseline', marker='o', linewidth=2)
    plt.plot(f_steps, f_losses, label='FlashAttention', marker='x', linestyle='--', linewidth=2)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Train Loss', fontsize=12)
    plt.title(f'{model.upper()} (Seq={seq.replace("seq", "")}) FP32 Train Loss Comparison', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('docs', 'images', f'{model}_loss_curve.png'))
    plt.close()

def plot_memory(model):
    seqs = ['seq64', 'seq256', 'seq512']
    seq_ints = [64, 256, 512]
    base_mems = []
    flash_mems = []
    
    for seq in seqs:
        b_log = os.path.join(LOG_DIR, f'{model}_baseline_fp32_{seq}.log')
        f_log = os.path.join(LOG_DIR, f'{model}_flash_fp32_{seq}.log')
        
        _, _, b_m = parse_log(b_log)
        _, _, f_m = parse_log(f_log)
        
        # Take the maximum/peak memory of the whole run
        base_mems.append(max(b_m) if b_m else 0)
        flash_mems.append(max(f_m) if f_m else 0)
        
    plt.figure(figsize=(8, 5))
    plt.plot(seq_ints, base_mems, label='Baseline', marker='o', linewidth=2)
    plt.plot(seq_ints, flash_mems, label='FlashAttention', marker='x', linestyle='--', linewidth=2)
    plt.xlabel('Sequence Length', fontsize=12)
    plt.ylabel('Peak Memory Used (MB)', fontsize=12)
    plt.title(f'{model.upper()} Peak Memory Usage vs Sequence Length', fontsize=14)
    
    # Annotate points
    for i, seq in enumerate(seq_ints):
        plt.annotate(f"{base_mems[i]}", (seq, base_mems[i] + 200), ha='center', fontsize=10)
        plt.annotate(f"{flash_mems[i]}", (seq, flash_mems[i] - 600), ha='center', fontsize=10)
        
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join('docs', 'images', f'{model}_memory_curve.png'))
    plt.close()

if __name__ == '__main__':
    plot_loss('llama3', 'seq256')
    plot_loss('gpt2', 'seq256')
    plot_memory('llama3')
    plot_memory('gpt2')
    print("Plots generated successfully in docs/images/")
