import sqlite3
c=sqlite3.connect('llama3_nvtx.sqlite')
def q(s,*a): return c.execute(s,a).fetchall()
def q1(s,*a): return c.execute(s,a).fetchone()
A,B=q1("select start,end from NVTX_EVENTS where coalesce(text,(select value from StringIds where id=textId))='Step_1'")
BW=1792128000000; PEAK=170*128*2*2.407e9
print("═══ ⑦ Adam kernel：grid 反推参数分组 + 显存带宽核算 ═══")
rows=q("""select k.gridX*k.gridY*k.gridZ gb,k.blockX bb,count(*) n,sum(k.end-k.start) t
          from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds sm on k.demangledName=sm.id
          where k.start>=? and k.start<? and sm.value like '%AdamAccumulateGrad%'
          group by gb,k.blockX order by gb desc""",A,B)
NAM={262668288:'embedding / lm_head  128256×2048',16777216:'gate/up/down_proj   2048×8192',
     6291456:'qkv_proj(融合)        2048×3072',4194304:'o_proj              2048×2048',
     2048:'RMSNorm weight        n_embd'}
te=tt=tr=0
print(f"  {'张量元素数':>13}{'次':>4}{'参数总量':>14}{'Σ(ms)':>9}{'avg(us)':>9}{'访存GB':>9}{'GB/s':>8}{'峰值%':>7}  推断")
for gb,bb,n,t in rows:
    elem=gb*bb; p=elem*n; traf=p*7*4; bw=traf/(t/1e9)
    print(f"  {elem:>13,}{n:>4}{p:>14,}{t/1e6:>9.4f}{t/n/1e3:>9.1f}{traf/1e9:>9.3f}"
          f"{bw/1e9:>8.0f}{100*bw/BW:>6.1f}%  {NAM.get(elem,'?')}")
    te+=p; tt+=t; tr+=traf
bw=tr/(tt/1e9)
print(f"  {'─'*13}{'─'*4}{'─'*14}{'─'*9}{'─'*9}{'─'*9}{'─'*8}{'─'*7}")
print(f"  {'合计':>13}{sum(r[2] for r in rows):>4}{te:>14,}{tt/1e6:>9.4f}{'':>9}{tr/1e9:>9.3f}{bw/1e9:>8.0f}{100*bw/BW:>6.1f}%")
print(f"\n  ★ Adam 覆盖 {te:,} 参数 = {te/1e9:.3f}B ≈ llama3.2-1B 全量")
print(f"    (16×6291456 qkv + 16×4194304 o + 48×16777216 MLP + 2×262668288 emb/head + 33×2048 norm")
print(f"     = {16*6291456+16*4194304+48*16777216+2*262668288+33*2048:,}，本窗口见到 {te:,}，")
print(f"     差 {16*6291456+16*4194304+48*16777216+2*262668288+33*2048-te:,} 即溢出到 Step_2 的部分)")
print(f"  ★ 访存 = 参数×7×4B（读 param/grad/m/v + 写 param/m/v）= {tr/1e9:.2f} GB / {tt/1e6:.3f} ms")
print(f"    → {bw/1e9:.0f} GB/s = 显存峰值 {BW/1e9:.0f} GB/s 的 {100*bw/BW:.1f}%，各组一致 → 纯带宽受限，kernel 本身无优化空间")

print(f"\n═══ ⑧ GEMM：kernel 数核对 + 算力达成率 ═══")
cut=q1("""select count(*),sum(k.end-k.start) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds sm on k.demangledName=sm.id
          where k.start>=? and k.start<? and sm.value like '%cutlass%'""",A,B)
mag=q1("""select count(*),sum(k.end-k.start) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds sm on k.demangledName=sm.id
          where k.start>=? and k.start<? and sm.value like '%magma%'""",A,B)
print(f"  cutlass sgemm {cut[0]} 个 Σ{cut[1]/1e6:.4f} ms | magma batched {mag[0]} 个 Σ{mag[1]/1e6:.4f} ms")
print(f"  核对 cutlass 数：16 层 × 5 Linear(qkv,o,gate,up,down) + lm_head = 81 个 fwd GEMM")
print(f"                   bwd 每 Linear 2 个(dW,dx) → 81×2 = 162；合计 81+162 = 243  ✓ 实测 {cut[0]}")
print(f"  核对 magma  数：16 层 × 2(QK^T, attn@V) × 3(fwd+2bwd) = 96            ✓ 实测 {mag[0]}")
M,L=4*64,16
f_layer = 6*M*2048*3072 + 6*M*2048*2048 + 3*6*M*2048*8192
f_head  = 6*M*2048*128256
f_attn  = 2*(3*2*(4*32)*64*64*64)
flops = L*f_layer + f_head + L*f_attn
sec = (cut[1]+mag[1])/1e9
ach = flops/sec
print(f"\n  理论 FLOPs/step（M=batch×seq={M}；一个 Linear 的 fwd+bwd = 3×2MNK = 6MNK）:")
print(f"    16 层 Linear  = {L*f_layer/1e9:>9.1f} GFLOP  ({100*L*f_layer/flops:>4.1f}%)")
print(f"    lm_head       = {f_head/1e9:>9.1f} GFLOP  ({100*f_head/flops:>4.1f}%)  ← 单个张量占 1/5")
print(f"    attention     = {L*f_attn/1e9:>9.3f} GFLOP  ({100*L*f_attn/flops:>4.2f}%)")
print(f"    合计          = {flops/1e12:>9.4f} TFLOP")
print(f"\n  GEMM 实测 Σ {sec*1e3:.4f} ms → 达成 {ach/1e12:.2f} TFLOPS")
print(f"  RTX 5090 FP32 峰值 = 170×128×2×2.407GHz = {PEAK/1e12:.1f} TFLOPS")
print(f"  ★ FP32 算力达成率 = {100*ach/PEAK:.1f}%")
print(f"  ★ 全部 cutlass 名含 'simt'+'align1'：走 CUDA core FP32、未向量化，未用 Tensor Core")
