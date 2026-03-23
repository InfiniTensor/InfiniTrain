import matplotlib.pyplot as plt
import re

def parse_log(filepath):
    steps = []
    losses = []
    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(r'step\s+(\d+)/\d+\s+\|\s+train loss\s+([\d.]+)', line)
            if match:
                steps.append(int(match.group(1)))
                losses.append(float(match.group(2)))
    return steps, losses

baseline_log = '/home/simon_chou/InfiniTrain/cuda-report/cuda-report-img/judge_gpt2_baseline_v7_bs2.log'
flash_log = '/home/simon_chou/InfiniTrain/cuda-report/cuda-report-img/judge_gpt2_flash_v7_bs2.log'

b_steps, b_losses = parse_log(baseline_log)
f_steps, f_losses = parse_log(flash_log)

plt.figure(figsize=(10, 6))
plt.plot(b_steps, b_losses, label='Baseline (FP32)', marker='o', linestyle='-', markersize=4)
plt.plot(f_steps, f_losses, label='FlashAttention (FP32)', marker='x', linestyle='--', markersize=4)

plt.title('GPT-2 Training Loss Alignment (SeqLen=1024)')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.savefig('/home/simon_chou/InfiniTrain/cuda-report/cuda-report-img/gpt2_loss_curve.png', dpi=300)
print("Chart generated: gpt2_loss_curve.png")
