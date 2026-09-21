import sqlite3, math
c=sqlite3.connect('llama3_nvtx.sqlite')
def q(s,*a): return c.execute(s,a).fetchall()
def q1(s,*a): return c.execute(s,a).fetchone()
A,B=q1("select start,end from NVTX_EVENTS where coalesce(text,(select value from StringIds where id=textId))='Step_1'")
W=B-A; SM,BW,MAXW,MAXB=170,1792128000000,48,24; TPS=MAXW*32
CK={1:'HtoD',2:'DtoH',8:'DtoD',9:'HtoH',0:'Unknown'}

print("═══ ③ 各类 memory 操作占用时间与实测带宽 ═══")
print(f"  {'操作':<10}{'次数':>6}{'Σ时间(ms)':>11}{'Σ字节':>14}{'实测带宽':>13}{'参考上限':>12}{'达成率':>9}{'avg(us)':>9}")
M=q("""select copyKind,count(*),sum(end-start),sum(bytes),min(bytes),max(bytes)
       from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and start<? group by copyKind""",A,B)
mtot=mt=0
for ck,n,t,by,mn,mx in sorted(M,key=lambda r:-r[2]):
    bw=by/(t/1e9)
    ref = BW if ck==8 else 63.0e9      # DtoD 走显存；HtoD/DtoH 走 PCIe5.0 x16 理论 63 GB/s
    refn= 'PCIe5.0x16' if ck!=8 else '显存1.79TB/s'
    print(f"  {CK.get(ck,ck):<10}{n:>6}{t/1e6:>11.4f}{by:>14,}{bw/1e9:>10.2f}GB/s{refn:>12}{100*bw/ref:>8.2f}%{t/n/1e3:>9.2f}")
    mtot+=t; mt+=by
Z=q1("select count(*),sum(end-start),sum(bytes) from CUPTI_ACTIVITY_KIND_MEMSET where start>=? and start<?",A,B)
if Z[0]:
    bw=Z[2]/(Z[1]/1e9)
    print(f"  {'memset':<10}{Z[0]:>6}{Z[1]/1e6:>11.4f}{Z[2]:>14,}{bw/1e9:>10.2f}GB/s{'显存1.79TB/s':>12}{100*bw/BW:>8.2f}%{Z[1]/Z[0]/1e3:>9.2f}")
    mtot+=Z[1]; mt+=Z[2]
print(f"  {'─'*10}{'─'*6}{'─'*11}{'─'*14}{'─'*13}")
print(f"  {'合计':<10}{sum(r[1] for r in M)+Z[0]:>6}{mtot/1e6:>11.4f}{mt:>14,}")
print(f"\n  memory 操作 Σ {mtot/1e6:.4f} ms = 窗口 {W/1e6:.3f} ms 的 {100*mtot/W:.2f}%，是 kernel 设备时间的 {100*mtot/72373800:.2f}%")

print(f"\n═══ ④ memcpy 按字节数细分（揭示每一次传输的语义）═══")
print(f"  {'方向':<6}{'字节':>10}{'次数':>6}{'Σ(ms)':>9}{'实测带宽':>13}  推断语义")
SEM={16:'StackForward 的 2 个指针数组 (sizeof(void*)*2)',32:'2 个 float 标量参数',
     24:'3 个 float 标量参数',40:'5 个 float 标量参数',96:'12 个 float 标量/Adam 超参',
     1024:'CrossEntropy 的 per-token loss (bs=256 × 4B)',4:'标量 loss 回读',
     2048:'输入 batch 数据',8192:'x/y 输入张量 (4*64*8B?)'}
for ck,by,n,t in q("""select copyKind,bytes,count(*),sum(end-start) from CUPTI_ACTIVITY_KIND_MEMCPY
                      where start>=? and start<? group by copyKind,bytes order by sum(end-start) desc limit 14""",A,B):
    print(f"  {CK.get(ck,ck):<6}{by:>10,}{n:>6}{t/1e6:>9.4f}{by/(t/1e9)/1e9:>10.2f}GB/s  {SEM.get(by,'?')}")

print(f"\n═══ ⑤ GPU 利用率三层分解 ═══")
K=q("""select k.start,k.end,k.gridX*k.gridY*k.gridZ,k.blockX*k.blockY*k.blockZ,sm.value
       from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds sm on k.demangledName=sm.id
       where k.start>=? and k.start<?""",A,B)
TOT=sum(e-s for s,e,*_ in K)
iv=sorted((s,e) for s,e,*_ in K); mg=[]
for s,e in iv:
    if mg and s<=mg[-1][1]: mg[-1][1]=max(mg[-1][1],e)
    else: mg.append([s,e])
UNION=sum(e-s for s,e in mg)
fill=0.0; tail=0.0; thr=0.0
for s,e,gb,bb,nm in K:
    bps=min(MAXB,TPS//bb) if bb else 1; cap=SM*bps
    fill+=(e-s)*min(gb,cap)/cap
    tail+=(e-s)*(gb/(math.ceil(gb/cap)*cap))
    thr+=(e-s)*min(gb*bb,cap*TPS)/ (cap*TPS)
print(f"  【层次1 时间占用率】GPU 有 kernel 在跑的时间 / 窗口")
print(f"      = {UNION/1e6:.3f} / {W/1e6:.3f} = {100*UNION/W:.2f}%    (空闲 {(W-UNION)/1e6:.3f} ms)")
print(f"  【层次2 SM 填充率】时间加权 min(blocks, 容量)/容量，容量 = 170 SM × blocksPerSm")
print(f"      = {100*fill/TOT:.2f}%    → kernel 在跑时，平均只占满这么多比例的 SM 驻留槽位")
print(f"  【层次3 线程填充率】时间加权 min(blocks×threads, 容量×1536)/(容量×1536)")
print(f"      = {100*thr/TOT:.2f}%")
print(f"  【wave 量化效率】blocks/(ceil(blocks/容量)×容量)，反映尾波浪费")
print(f"      = {100*tail/TOT:.2f}%")
print(f"\n  → 有效算力利用 ≈ 时间占用率 × SM 填充率 = {100*UNION/W:.1f}% × {100*fill/TOT:.1f}% = {100*UNION/W*fill/TOT:.1f}%")

print(f"\n═══ ⑥ SM 填充率最低的 kernel（浪费最严重）═══")
G={}
for s,e,gb,bb,nm in K:
    bps=min(MAXB,TPS//bb) if bb else 1; cap=SM*bps
    g=G.setdefault(nm,[0,0,0.0,gb,bb,cap]); g[0]+=1; g[1]+=e-s; g[2]+=(e-s)*min(gb,cap)/cap
print(f"  {'kernel(截断)':<62}{'n':>4}{'Σ(ms)':>8}{'SM填充':>8}{'blocks':>9}{'容量':>7}")
for nm,(n,t,f,gb,bb,cap) in sorted(G.items(),key=lambda x:x[1][2]/x[1][1])[:9]:
    print(f"  {nm.replace('void infini_train::kernels::cuda::','').replace('<unnamed>::','')[:61]:<62}"
          f"{n:>4}{t/1e6:>8.4f}{100*f/t:>7.1f}%{gb:>9}{cap:>7}")
