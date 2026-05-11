from safetensors.torch import load_file
import os, json

hf_path = '../../../Qwen3-8B'
index_path = os.path.join(hf_path, 'model.safetensors.index.json')
with open(index_path) as f:
    index = json.load(f)

# 所有包含 'norm' 的 key
norm_keys = sorted([k for k in index['weight_map'].keys() if 'norm' in k.lower()])
print('=== All norm keys ===')
for k in norm_keys:
    print(f'  {k}')

# 检查有没有 q_norm
qk_keys = sorted([k for k in index['weight_map'].keys() if 'q_norm' in k.lower() or 'k_norm' in k.lower()])
print(f'\nq_norm/k_norm keys count: {len(qk_keys)}')
for k in qk_keys[:5]:
    print(f'  {k}')

# layer 0 的所有 key
l0 = sorted([k for k in index['weight_map'].keys() if 'layers.0' in k])
print(f'\nLayer 0 keys ({len(l0)} total):')
for k in l0:
    print(f'  {k}')
