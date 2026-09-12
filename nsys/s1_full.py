import sqlite3, re, math
c = sqlite3.connect('llama3_nvtx.sqlite')
def q(s,*a): return c.execute(s,a).fetchall()
def q1(s,*a): return c.execute(s,a).fetchone()
A,B = q1("""select start,end from NVTX_EVENTS
            where coalesce(text,(select value from StringIds where id=textId))='Step_1'""")
W = B-A
SM, BW, MAXW, MAXB = 170, 1792128000000, 48, 24
TPS = MAXW*32                      # 每 SM 最大线程

def classify(n):
    n = n.replace('void ','').strip()
    if 'cutlass_' in n:
        m = re.search(r'cutlass_(\d+)_(\w+)_(\w+)_(\d+x\d+)_(\d+x\d+)_(\w+)_align(\d+)', n)
        if m: return f"GEMM cutlass_{m.group(3)}[{m.group(4)},{m.group(6)},align{m.group(7)}]"
    if 'magma_sgemmEx' in n: return "GEMM magma_batched_sgemm"
    n2 = n.replace('<unnamed>::','')
    kname = n2.split('<')[0].split('::')[-1]
    m = re.search(r'kernels::cuda::(\w+?)(Forward|Backward)\(', n)
    if m and kname.startswith(('Binary','Unary','Generic')):
        v = 'vec' if 'Vectorized' in kname else ('nb' if 'NoBroadcast' in kname else 'bcast')
        return f"{m.group(1)}{m.group(2)} [{kname.replace('Kernel','')}/{v}]"
    if 'GenericReduce' in kname:
        m2 = re.search(r'::(\w+)Finalize', n)
        return f"Reduce<{m2.group(1) if m2 else '?'}> [{kname.replace('Kernel','')}]"
    return kname

K = q("""select k.start,k.end,k.gridX*k.gridY*k.gridZ,k.blockX*k.blockY*k.blockZ,
                k.registersPerThread,k.staticSharedMemory,k.correlationId,sm.value
         from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds sm on k.demangledName=sm.id
         where k.start>=? and k.start<? order by k.start""", A, B)
TOT = sum(e-s for s,e,*_ in K)
iv=sorted((s,e) for s,e,*_ in K); mg=[]
for s,e in iv:
    if mg and s<=mg[-1][1]: mg[-1][1]=max(mg[-1][1],e)
    else: mg.append([s,e])
UNION=sum(e-s for s,e in mg)

print("╔══════════════════════════════════════════════════════════════════════════╗")
print(f"║  Step_1 窗口 CUDA HW 统计（已排除 warmup：Step_0 及其之前 227 个 kernel）      ")
print("╚══════════════════════════════════════════════════════════════════════════╝")
print(f"  窗口 {W/1e6:.3f} ms | kernel {len(K)} 个 | Σ设备 {TOT/1e6:.3f} ms | 并集 {UNION/1e6:.3f} ms | 单 stream 无并发")

# ── 分类聚合 ──
G={}
for s,e,gb,bb,rp,shm,cid,nm in K:
    g=G.setdefault(classify(nm),{'n':0,'t':0,'fill':0.0,'tail':0.0,'gb':[],'bb':bb})
    g['n']+=1; g['t']+=e-s; g['gb'].append(gb)
    bps=min(MAXB, TPS//bb) if bb else 1
    cap=SM*bps
    g['fill'] += (e-s)*min(gb,cap)/cap
    g['tail'] += (e-s)*(gb/(math.ceil(gb/cap)*cap))
print(f"\n═══ ① 各类 kernel 占用时间（按 Σ 设备时间排序）═══")
print(f"  {'#':<3}{'kernel 类别':<48}{'次数':>5}{'Σ(ms)':>9}{'占比':>7}{'avg(us)':>9}{'SM填充':>8}")
for i,(nm,g) in enumerate(sorted(G.items(),key=lambda x:-x[1]['t']),1):
    print(f"  {i:<3}{nm[:47]:<48}{g['n']:>5}{g['t']/1e6:>9.4f}{100*g['t']/TOT:>6.2f}%"
          f"{g['t']/g['n']/1e3:>9.2f}{100*g['fill']/g['t']:>7.1f}%")

# ── 大类汇总 ──
CAT={'GEMM(cutlass+magma)':['GEMM'],'Optimizer(Adam)':['AdamAccumulate'],'GradAccum(SGD/累加)':['AccumulateGradKernel'],
     'Elementwise(逐元素)':['Mul','Add','Sub','Div','Sigmoid','Rsqrt','Pow','AddScalar','MulScalar'],
     'Reshape/搬运':['FillKernel','TransposeForward','Slice','Stack','RepeatInterleave','Triu'],
     'Reduce/Norm':['Reduce','GenericReduce'],'Embedding':['Embedding'],'Softmax':['Softmax'],
     'CrossEntropy':['CrossEntropy'],'Mask':['Mask']}
print(f"\n═══ ② 按功能大类汇总 ═══")
print(f"  {'大类':<26}{'次数':>6}{'Σ(ms)':>10}{'占比':>8}{'SM填充率':>10}")
acc={}
for nm,g in G.items():
    hit='其他'
    for cat,pats in CAT.items():
        if any(p in nm for p in pats): hit=cat; break
    a=acc.setdefault(hit,[0,0,0.0]); a[0]+=g['n']; a[1]+=g['t']; a[2]+=g['fill']
for cat,(n,t,f) in sorted(acc.items(),key=lambda x:-x[1][1]):
    print(f"  {cat:<26}{n:>6}{t/1e6:>10.4f}{100*t/TOT:>7.2f}%{100*f/t:>9.1f}%")
print(f"  {'─'*26}{'─'*6}{'─'*10}{'─'*8}{'─'*10}")
print(f"  {'合计':<26}{len(K):>6}{TOT/1e6:>10.4f}{100.0:>7.2f}%{100*sum(g['fill'] for g in G.values())/TOT:>9.1f}%")
