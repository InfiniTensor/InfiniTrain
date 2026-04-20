from transformers import AutoTokenizer
import struct

tokenizer = AutoTokenizer.from_pretrained("/var/qy_home/jiyiming/Qwen3-8B")

with open("input.txt", "r") as f:
    text = f.read()

ids = tokenizer.encode(text)

with open("tiny_shakespeare_qwen3.bin", "wb") as f:
    for tid in ids:
        f.write(struct.pack("I", tid))

print(f"Wrote {len(ids)} tokens")
