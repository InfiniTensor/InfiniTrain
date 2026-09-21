import sqlite3
c = sqlite3.connect('llama3_nvtx.sqlite')
def q(s,*a): return c.execute(s,a).fetchall()
S = {i:v for i,v in c.execute("select id,value from StringIds")}
a,b = q("select start,end from NVTX_EVENTS where coalesce(text,(select value from StringIds where id=textId))='Step_2'")[0]
CK = {0:'HtoH',1:'HtoD',2:'DtoH',3:'HtoA',4:'AtoH',8:'DtoD'}

ev = []
for s,e,n in q("""select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k
                  join StringIds s on k.demangledName=s.id where k.start>=? and k.start<?""",a,b):
    ev.append((s,e,'K',n))
for s,e,ck,by in q("select start,end,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and start<?",a,b):
    ev.append((s,e,'M',f"{CK.get(ck,ck)}-{by}B"))
for s,e in q("select start,end from CUPTI_ACTIVITY_KIND_MEMSET where start>=? and start<?",a,b):
    ev.append((s,e,'S','memset'))
ev.sort()

merged = []
for s,e,k,n in ev:
    if merged and s <= merged[-1][1]:
        merged[-1][1] = max(merged[-1][1], e)
    else:
        merged.append([s,e,(k,n)])
gaps = [(merged[i][1], merged[i+1][0], merged[i][2], merged[i+1][2])
        for i in range(len(merged)-1) if merged[i+1][0] > merged[i][1]]
gaps.sort(key=lambda g: -(g[1]-g[0]))
idle = sum(g[1]-g[0] for g in gaps)
span = b-a
print(f"=== step3 窗口 {span/1e6:.3f} ms | GPU 活动段 {len(merged)} | gap {len(gaps)} 个 ===")
print(f"    GPU 空闲合计 = {idle/1e6:.3f} ms = {100*idle/span:.1f}% 窗口\n")
print(f"=== 最大的 10 个 GPU 空隙 ===")
print(f"  {'#':<3}{'起点(ms)':>9}{'空隙(ms)':>10}  {'空隙前最后一个活动':<40}{'空隙后第一个活动'}")
for i,(gs,ge,pv,nx) in enumerate(gaps[:10],1):
    print(f"  {i:<3}{(gs-a)/1e6:>9.3f}{(ge-gs)/1e6:>10.4f}  {pv[0]+':'+pv[1][:36]:<40}{nx[0]+':'+nx[1][:34]}")

print(f"\n=== 最大 5 个 gap 期间 host 在做什么 ===")
for i,(gs,ge,pv,nx) in enumerate(gaps[:5],1):
    print(f"\n  --- gap#{i}  起点 +{(gs-a)/1e6:.3f} ms, 持续 {(ge-gs)/1e6:.4f} ms ---")
    print(f"      前: {pv[0]}:{pv[1][:60]}")
    print(f"      后: {nx[0]}:{nx[1][:60]}")
    H = {}
    for nid,s2,e2 in q("select nameId,start,end from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and start<?",gs,ge):
        nm = S.get(nid,str(nid)); x = H.setdefault(nm,[0,0]); x[0]+=1; x[1]+=e2-s2
    if not H: print("      host CUDA API: 无 —— GPU 纯空闲，主机在做纯 CPU 工作")
    for nm,(n,t) in sorted(H.items(),key=lambda x:-x[1][1])[:4]:
        print(f"      host API {nm:<36} n={n:<4} Σ={t/1e3:>9.1f} us  avg={t/n/1e3:.1f} us")
    # 跨 gap 的长阻塞 API
    lg = q("""select r.start,r.end,s.value from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id
              where r.start<? and r.end>? order by (r.end-r.start) desc limit 3""",ge,gs)
    for s2,e2,nm in lg:
        if e2-s2 > (ge-gs)*0.5:
            print(f"      ★跨整个 gap 的阻塞调用: {nm}  时长 {(e2-s2)/1e6:.4f} ms (起 +{(s2-a)/1e6:.3f})")
