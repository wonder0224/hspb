"""
실재 셀 포화 — 자질 격자 위 의미 분화의 유계성 (508, noun.person 루트)
=====================================================================
사용자 정식화: 각 자질=토큰영역. k자질→2^k 비트열. 공집합 비트열 존재.
  의미 분화 = 비어있지 않은(실재) 비트열 수. 공집합은 분화 아님.
  자질 늘리면 2^k 폭발하나 실재 셀은 토큰수에 막혀 포화 = 유계 의미분화.
측정 (라벨·임베딩 안 씀, 순수 조합):
 1) 자질 순번 k 늘리며 실재 셀(distinct 비트열) 수 곡선.
 2) 새 자질이 실재셀을 진짜 쪼개나(실재→실재+실재) vs 공집합분기(실재→실재+공집합).
 3) 실재 셀 포화 k. 2^k 대비 실재 비율(공집합이 얼마나 지배하나).
도메인 3개 다.
"""
import numpy as np, json
from sklearn.cluster import KMeans

def run(dom, root_ss):
    if dom=='508':
        EMB=np.load('/mnt/user-data/uploads/news_legal_embeddings.npy').astype(np.float64)
        U=json.load(open('/mnt/user-data/uploads/news_legal_units.json'))
        def ss(u):
            l=u.get('l1',''); return l.split('|') if l else []
    elif dom=='diverse':
        EMB=np.load('/mnt/user-data/uploads/news_diverse_embeddings.npy').astype(np.float64)
        U=json.load(open('/mnt/user-data/uploads/news_diverse_units.json'))
        def ss(u): return u.get('ontological',[]) or []
    elif dom=='onco':
        EMB=np.load('/mnt/user-data/uploads/oncology_embeddings.npy').astype(np.float64)
        U=json.load(open('/mnt/user-data/uploads/oncology_units.json'))
        def ss(u): return u.get('l1',[]) or []
    N=EMB.shape[0]
    root=[i for i in range(N) if root_ss in ss(U[i])]
    if len(root)<20: return None
    km=KMeans(n_clusters=200,random_state=42,n_init=3).fit(EMB)
    pr=km.cluster_centers_; pr/=(np.linalg.norm(pr,axis=1,keepdims=True)+1e-12)
    s=EMB@pr.T; Z=(s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-12)
    Bd=(Z>1.5).astype(np.int8)
    Rt=Bd[root]   # len(root) x 200
    # 변별 순번: 루트 균형 큰 순
    def bal(c):
        on=int(Rt[:,c].sum()); return min(on,len(root)-on)
    active=[c for c in range(200) if 0<int(Rt[:,c].sum())<len(root)]
    order=sorted(active,key=lambda c:-bal(c))

    nroot=len(root)
    cells_curve=[]; real_split=[]; empty_split=[]
    keys=[() for _ in range(nroot)]   # 각 토큰 비트열(누적)
    prev_cells=1
    for k,c in enumerate(order,1):
        bits=Rt[:,c]
        newkeys=[keys[i]+(int(bits[i]),) for i in range(nroot)]
        cells=len(set(newkeys))
        # 직전 셀 중 몇 개가 진짜 둘로(실재+실재), 몇 개가 공집합분기(실재 그대로)
        from collections import defaultdict
        prev_groups=defaultdict(list)
        for i in range(nroot): prev_groups[keys[i]].append(i)
        rs=es=0
        for g,mem in prev_groups.items():
            on=sum(1 for i in mem if bits[i]>0); off=len(mem)-on
            if on>0 and off>0: rs+=1   # 실재 분할
            # else: 이 자질이 이 셀 안 가름(공집합 분기 효과)
        cells_curve.append(cells); real_split.append(rs)
        keys=newkeys
        if cells>=nroot: break   # 모든 토큰 distinct = 완전 포화
    # 포화 k: 셀 증가가 멈추는(또는 거의) 지점
    cc=np.array(cells_curve)
    sat=None
    for kk in range(len(cc)-5):
        if cc[kk+5]-cc[kk] <= 1: sat=kk+1; break
    return dict(dom=dom,root_ss=root_ss,nroot=nroot,
                n_feats=len(order),
                cells_curve=cc.tolist(),
                final_cells=int(cc[-1]),
                saturation_k=sat,
                real_splits=real_split)

# 루트: 각 도메인 최대 supersense
configs=[('508','noun.person'),('diverse','noun.group'),('onco','noun.state')]
results=[]
for dom,rss in configs:
    r=run(dom,rss)
    if r is None: continue
    results.append(r)
    cc=r['cells_curve']
    print("="*56)
    print(f"{dom} / 루트 {rss}: {r['nroot']}토큰, 발견자질 {r['n_feats']}개")
    print("="*56)
    print(f"  실재 셀 수 곡선 (자질 k개일 때 distinct 비트열):")
    marks=[1,2,3,5,10,20,30,50,len(cc)]
    for k in marks:
        if k<=len(cc):
            twok = 2**k if k<=30 else float('inf')
            ratio = cc[k-1]/twok if k<=30 else 0
            print(f"    k={k:3d}: 실재셀 {cc[k-1]:3d} / 2^k={twok if k<=20 else '큼':>10} "
                  f"(실재비율 {ratio:.2e})" if k<=20 else
                  f"    k={k:3d}: 실재셀 {cc[k-1]:3d}")
    print(f"  최종 실재셀 {r['final_cells']} (토큰 {r['nroot']}개 상한)")
    print(f"  셀 증가 포화 k ≈ {r['saturation_k']}")
    print(f"  → 2^k는 폭발하나 실재셀은 {r['final_cells']}에서 멈춤 = 의미분화 유계\n")

json.dump(results,open('/home/claude/realcells.json','w'),ensure_ascii=False,indent=2)
print("[saved] realcells.json")
