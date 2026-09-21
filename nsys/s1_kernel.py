import sqlite3, re
c = sqlite3.connect('llama3_nvtx.sqlite')
def q(s,*a): return c.execute(s,a).fetchall()
def q1(s,*a): return c.execute(s,a).fetchone()
S = {i:v for i,v in c.execute("select id,value from StringIds")}
A,B = q1("""select start,end from NVTX_EVENTS
            where coalesce(text,(select value from StringIds where id=textId))='Step_1'""")
W = B-A
st = q("""select coalesce(text,(select value from StringIds where id=textId)) t,start,end
          from NVTX_EVENTS where t like 'Step_%' or t in ('Forward','Backward','Optimizer','ZeroGrad','LossReadback')
          order by start""")
S0 = [x for x in st if x[0]=='Step_0'][0]
warm_end = S0[1]

def clean(n):
    n = n.replace('void ','').strip()
    m = re.search(r'cutlass_(\d+)_(\w+?)_(\w+?)_(\d+x\d+x\d+)_(\w+?)_align', n)
    if m: return f"cutlass_{m.group(2)}_{m.group(3)}_{m.group(4)}_{m.group(5)}"
    m = re.search(r'cublas\w*', n, re.I)
    if m: return m.group(0)
    base = n.split('<')[0]
    base = base.split('(')[0]
    return base.split('::')[-1]

K = q("""select k.start,k.end,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,
                k.registersPerThread,k.staticSharedMemory,k.correlationId,sm.value
         from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds sm on k.demangledName=sm.id
         where k.start>=? and k.start<? order by k.start""", A, B)
tot = sum(e-s for s,e,*_ in K)

# 并发检测（是否多 stream 重叠）
iv = sorted((s,e) for s,e,*_ in K); merged=[]
for s,e in iv:
    if merged and s<=merged[-1][1]: merged[-1][1]=max(merged[-1][1],e)
    else: merged.append([s,e])
union = sum(e-s for s,e in merged)
nstream = q1("select count(distinct streamId) from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and start<?",A,B)[0]

print(f"════════ Step_1 窗口 CUDA HW kernel 统计（排除 warmup）════════")
print(f"  窗口宽度        {W/1e6:.3f} ms")
print(f"  kernel 总数     {len(K)}")
print(f"  Σ 设备时间      {tot/1e6:.3f} ms")
print(f"  区间并集        {union/1e6:.3f} ms   (Σ/并集 = {tot/union:.4f} → {'无并发' if tot/union<1.001 else '有重叠'})")
print(f"  使用的 stream   {nstream}")
print(f"  GPU 时间占用率  {100*union/W:.2f}%   (空闲 {100*(1-union/W):.2f}% = {(W-union)/1e6:.3f} ms)")

# 提交归属
own = {'warmup(Step_0前)':0,'Step_0(溢出)':0,'Step_1(本步)':0}
ownt= {'warmup(Step_0前)':0,'Step_0(溢出)':0,'Step_1(本步)':0}
for s,e,gx,gy,gz,bx,by,bz,rp,shm,cid,nm in K:
    r = q1("select start from CUPTI_ACTIVITY_KIND_RUNTIME where correlationId=?",cid)
    ts = r[0] if r else s
    k = 'warmup(Step_0前)' if ts<warm_end else ('Step_0(溢出)' if ts<A else 'Step_1(本步)')
    own[k]+=1; ownt[k]+=e-s
print(f"\n  ── 按【提交时刻】归属（揭示跨 step 溢出）──")
for k in own:
    print(f"     {k:<18} {own[k]:>5} 个 kernel   Σ {ownt[k]/1e6:>8.3f} ms   ({100*ownt[k]/tot:>5.1f}% 设备时间)")

print(f"\n════════ 各类 kernel 占用时间（按 Σ 设备时间排序）════════")
G={}
for s,e,gx,gy,gz,bx,by,bz,rp,shm,cid,nm in K:
    g=G.setdefault(clean(nm),[0,0,gx*gy*gz,bx*by*bz,[]])
    g[0]+=1; g[1]+=e-s; g[4].append((e-s,gx*gy*gz,bx*by*bz,rp,shm))
print(f"  {'#':<3}{'kernel 类别':<40}{'次数':>6}{'Σ时间(ms)':>11}{'占比':>8}{'avg(us)':>9}{'blocks':>10}{'thr/blk':>8}")
for i,(nm,(n,t,gb,bb,_)) in enumerate(sorted(G.items(),key=lambda x:-x[1][1]),1):
    print(f"  {i:<3}{nm[:39]:<40}{n:>6}{t/1e6:>11.4f}{100*t/tot:>7.2f}%{t/n/1e3:>9.2f}{gb:>10}{bb:>8}")
print(f"\n  合计 {len(G)} 类，{len(K)} 个 kernel，Σ {tot/1e6:.3f} ms")
