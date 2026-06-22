"""
의미 분기 다이어그램 — 기저자질 1개 루트, 발견자질로 분기 추적 (508)
=====================================================================
루트 = noun.person 토큰 집합(자명한 기저 분류).
발견자질을 응집/변별 순으로 적용하며 이 집합이 어떻게 갈라지나 추적.
각 분기 노드: 토큰 수, 대표 텍스트, 4축 라벨 분포(독립 검증 — 자질에 안 쓴 축).
출력: 트리 구조 + 각 잎의 4축 순도(유효한 의미무리인가).
"""
import numpy as np, json
from sklearn.cluster import KMeans
from collections import Counter

EMB=np.load('/mnt/user-data/uploads/news_legal_embeddings.npy').astype(np.float64)
UNITS=json.load(open('/mnt/user-data/uploads/news_legal_units.json'))
N=EMB.shape[0]
def ss_of(u):
    l=u.get('l1',''); return l.split('|') if l else []

ROOT_SS='noun.person'
root=[i for i,u in enumerate(UNITS) if ROOT_SS in ss_of(UNITS[i])]
print(f"루트 = {ROOT_SS}: {len(root)} 토큰")

# 발견자질 (전체에서)
km=KMeans(n_clusters=200,random_state=42,n_init=3).fit(EMB)
pr=km.cluster_centers_; pr/=(np.linalg.norm(pr,axis=1,keepdims=True)+1e-12)
s=EMB@pr.T; Z=(s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-12)
Bd=(Z>1.5).astype(np.int8)

# 루트 토큰 안에서 변별력 있는 발견자질 선택: 루트를 가장 균형있게 가르는 것부터
def axis_dist(idx, axis):
    c=Counter()
    for i in idx:
        v=UNITS[i].get(axis,'')
        if v: c[v]+=1
    return c

def describe(idx, maxn=5):
    txts=[UNITS[i]['text'] for i in idx[:maxn]]
    actor=axis_dist(idx,'actor')
    return txts, dict(actor.most_common(3))

# 루트를 발견자질로 순차 분기 (가장 루트를 잘 가르는 자질 순)
root_set=set(root)
def split_quality(idx, feat):
    on=[i for i in idx if feat[i]>0]
    if not on or len(on)==len(idx): return 0
    return min(len(on),len(idx)-len(on))  # 균형도

# 상위 분기 자질 5개로 트리 (이진 분기 누적)
print("\n=== 분기 다이어그램 (noun.person 루트) ===\n")
# 루트의 4축 라벨 (분기 전)
print(f"[ROOT] noun.person, n={len(root)}")
print(f"  actor 분포: {dict(axis_dist(root,'actor').most_common(6))}")
print(f"  대표: {[UNITS[i]['text'] for i in root[:6]]}")

# 순차 분기: 매 단계 현재 잎들 중 가장 큰 잎을 가장 잘 가르는 발견자질로 분기
leaves=[root]   # 각 잎 = 토큰 인덱스 리스트
feat_used=[]
for depth in range(4):
    # 가장 큰 잎
    big=max(leaves,key=len)
    if len(big)<8: break
    # 그 잎을 가장 균형있게 가르는 발견자질
    best_c,best_q=None,0
    for c in range(Bd.shape[1]):
        q=split_quality(big,Bd[:,c])
        if q>best_q: best_q,best_c=q,c
    if best_c is None: break
    feat=Bd[:,best_c]; feat_used.append(best_c)
    on=[i for i in big if feat[i]>0]; off=[i for i in big if feat[i]==0]
    leaves.remove(big); leaves.extend([on,off])
    print(f"\n[분기 {depth+1}] 발견자질#{best_c}로 n={len(big)} → on={len(on)}, off={len(off)}")
    ton,aon=describe(on); toff,aoff=describe(off)
    print(f"   ON  (n={len(on)}): {ton}")
    print(f"        actor: {aon}")
    print(f"   OFF (n={len(off)}): {toff}")
    print(f"        actor: {aoff}")

# 최종 잎들의 4축 순도
print("\n=== 최종 잎(의미 무리) 4축 순도 ===")
def purity(idx,axis):
    d=axis_dist(idx,axis)
    if not d: return 0,None
    tot=sum(d.values()); top=d.most_common(1)[0]
    return top[1]/tot, top[0]
for k,leaf in enumerate(sorted(leaves,key=len,reverse=True)):
    if len(leaf)<3: continue
    pa,la=purity(leaf,'actor'); pt,lt=purity(leaf,'topic')
    print(f"  잎{k} n={len(leaf):3d}: actor순도 {pa:.2f}({la}) | topic순도 {pt:.2f}({lt})")
    print(f"        대표: {[UNITS[i]['text'] for i in leaf[:4]]}")

print("\n→ 잎들이 actor로 깨끗이 갈리면(judge/attorney/defendant) 발견자질이")
print("  noun.person 안을 유효한 의미무리로 재분할 = 기저(person)+발견(actor세분) 통합 유효.")

json.dump(dict(root_size=len(root),feats=feat_used,
               leaves=[{'n':len(l),'actor':dict(axis_dist(l,'actor').most_common(3)),
                        'texts':[UNITS[i]['text'] for i in l[:5]]} for l in leaves]),
          open('/home/claude/diagram_person_508.json','w'),ensure_ascii=False,indent=2)
print("[saved] diagram_person_508.json")
