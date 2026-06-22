"""
G절 실험 — 발견 군집 자질 ↔ WordNet supersense 흡수율 (508 legal)
====================================================================
설계 확정사항 (LEXICON G절):
- 병렬 발견: 군집은 임베딩에서 F1대로 독립 발견(supersense/4축 안 봄), 사후에 흡수율 측정.
- 고유 기저 = l1 16 supersense (universal·fixed). 4축은 흡수율 분모 아님(해석 대조용).
- purity 기반 흡수율 + 우연보정(쏠림 편향 제거). raw도 같이 보고.
- multi-supersense 토큰(116개)은 각 supersense에 1/k 분할 귀속.

이 실험이 답하는 것: 발견 군집이 supersense 하나로 우연이상 모이나?
  높은 보정흡수율 → 갈래1(상보, WordNet이 굵은 변별 흡수)
  낮은 보정흡수율 → 갈래2(직교, 군집이 supersense보다 잘아 거의 다시 함)
"""
import numpy as np, json
from scipy.stats import spearmanr
from collections import defaultdict

np.random.seed(42)
EMB = np.load('/mnt/user-data/uploads/news_legal_embeddings.npy').astype(np.float64)
UNITS = json.load(open('/mnt/user-data/uploads/news_legal_units.json'))
N = EMB.shape[0]
assert N == len(UNITS) == 508

# ---- supersense 귀속 행렬 (분할 귀속) ----
# S[i, s] = 토큰 i가 supersense s에 귀속되는 가중치(합=1)
all_ss = []
tok_ss = []
for u in UNITS:
    parts = u['l1'].split('|') if u['l1'] else []
    tok_ss.append(parts)
    all_ss.extend(parts)
SS = sorted(set(all_ss))
ss_idx = {s:j for j,s in enumerate(SS)}
S = np.zeros((N, len(SS)))
for i, parts in enumerate(tok_ss):
    if parts:
        w = 1.0/len(parts)
        for p in parts:
            S[i, ss_idx[p]] += w
# 기저비율 (chance): 각 supersense의 전체 질량 / N
base_rate = S.sum(axis=0) / N   # len(SS)

print(f"[setup] N={N}, supersenses={len(SS)}")
print(f"[setup] base rates: " + ", ".join(f"{s}={base_rate[ss_idx[s]]:.3f}" for s in SS[:5]) + " ...")

# ============================================================
# F1 발견 절차 (step2 재현): prototype 후보 풀 + z이진화 + greedy ρ 포화
# ============================================================
def discover_features(emb, k_pool=200, z_thresh=1.5, eps=0.002, max_pairs=15000, seed=42):
    """
    1) prototype 후보 풀 k개: 임베딩을 KMeans로 k개 군집 중심(후보 자질).
    2) 각 토큰을 후보별 유사도 → row-wise z-score → z>z_thresh면 그 후보자질 active(이진).
    3) greedy: 변별보존 ρ(binary Jaccard vs cosine)를 가장 올리는 자질 순서로 누적,
       ρ 증가분 < eps 되면 포화 → 그때까지 선택된 자질 수 = 발견 고정폭.
    반환: 선택된 후보 인덱스 리스트, 최종 이진행렬(N x 선택수), ρ 궤적
    """
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(seed)
    n = emb.shape[0]
    # 1) 후보 풀
    km = KMeans(n_clusters=k_pool, random_state=seed, n_init=3)
    km.fit(emb)
    protos = km.cluster_centers_
    protos /= (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-12)
    # 2) 유사도 → row-wise z → 이진
    sim = emb @ protos.T               # n x k_pool  (코사인, 임베딩 정규화됨)
    mu = sim.mean(axis=1, keepdims=True)
    sd = sim.std(axis=1, keepdims=True) + 1e-12
    Z = (sim - mu) / sd
    B_full = (Z > z_thresh).astype(np.int8)   # n x k_pool 이진
    # 참조 코사인 (쌍 표본) + 랭크 사전계산 (spearman = rank상의 pearson)
    pairs = rng.integers(0, n, size=(max_pairs, 2))
    pairs = pairs[pairs[:,0] != pairs[:,1]]
    cos_ref = np.einsum('ij,ij->i', emb[pairs[:,0]], emb[pairs[:,1]])
    from scipy.stats import rankdata
    cos_rank = rankdata(cos_ref)
    cos_rank = (cos_rank - cos_rank.mean())
    cos_rank_norm = cos_rank / (np.linalg.norm(cos_rank) + 1e-12)
    pi, pj = pairs[:,0], pairs[:,1]

    def rho_of(trial_cols):
        Bsub = B_full[:, trial_cols]
        a = Bsub[pi]; b = Bsub[pj]
        inter = (a & b).sum(axis=1)
        union = (a | b).sum(axis=1)
        j = np.where(union>0, inter/union, 0.0)
        if j.std() < 1e-12:
            return -1.0
        jr = rankdata(j); jr = jr - jr.mean()
        jn = jr / (np.linalg.norm(jr) + 1e-12)
        return float(jn @ cos_rank_norm)

    selected = []
    rho_traj = []
    cur_rho = 0.0
    active_cands = [c for c in range(k_pool) if B_full[:,c].sum() > 0]
    remaining = active_cands
    while remaining:
        best_c, best_rho = None, cur_rho
        for c in remaining:
            r = rho_of(selected + [c])
            if r > best_rho:
                best_rho, best_c = r, c
        if best_c is None:
            break
        gain = best_rho - cur_rho
        selected.append(best_c)
        remaining.remove(best_c)
        rho_traj.append(best_rho)
        if gain < eps and len(selected) >= 3:
            break
        cur_rho = best_rho
    B_sel = B_full[:, selected]
    return selected, B_sel, rho_traj

# ============================================================
# 흡수율: 발견 자질(=선택된 prototype 군집)이 supersense로 환원되나
# ============================================================
def absorption(B_sel, S, base_rate, ss_list, ss_idx, adj_thresh=0.2):
    """
    각 발견자질 f(이진열): 그 자질을 켠 토큰들의 supersense 질량 분포.
      raw_purity = max_s (그 토큰들의 s 귀속질량 평균)
      adj_purity = raw_purity - base_rate[argmax]   (우연보정)
    흡수율 = adj_purity >= adj_thresh 인 발견자질의 비율
    반환: 발견자질별 (top supersense, raw, adj), 요약
    """
    results = []
    F = B_sel.shape[1]
    for f in range(F):
        on = B_sel[:, f].astype(bool)
        if on.sum() == 0:
            continue
        mass = S[on].mean(axis=0)        # 켠 토큰들의 supersense 평균 귀속
        top = int(mass.argmax())
        raw = float(mass[top])
        adj = raw - float(base_rate[top])
        results.append({
            'feature': f, 'n_on': int(on.sum()),
            'top_ss': ss_list[top], 'raw_purity': round(raw,3),
            'adj_purity': round(adj,3)
        })
    n_absorbed_raw = sum(1 for r in results if r['raw_purity'] >= 0.5)
    n_absorbed_adj = sum(1 for r in results if r['adj_purity'] >= adj_thresh)
    summary = {
        'n_features': len(results),
        'absorption_raw_(purity>=0.5)': round(n_absorbed_raw/max(len(results),1), 3),
        'absorption_adj_(adj>=%.1f)'%adj_thresh: round(n_absorbed_adj/max(len(results),1), 3),
        'mean_raw_purity': round(np.mean([r['raw_purity'] for r in results]), 3),
        'mean_adj_purity': round(np.mean([r['adj_purity'] for r in results]), 3),
    }
    return results, summary

# ---- 실행 ----
print("\n[discover] running F1 discovery (eps=0.002)...")
selected, B_sel, rho_traj = discover_features(EMB, k_pool=200, z_thresh=1.5, eps=0.002)
print(f"[discover] 발견 고정폭 = {len(selected)} features (ρ 최종={rho_traj[-1]:.4f})")

print("\n[absorption] 발견 군집 ↔ supersense 흡수율...")
res, summ = absorption(B_sel, S, base_rate, SS, ss_idx, adj_thresh=0.2)
print(json.dumps(summ, ensure_ascii=False, indent=2))

print("\n[per-feature] 발견자질별 top supersense (raw vs 보정):")
for r in sorted(res, key=lambda x:-x['adj_purity']):
    print(f"  f{r['feature']:3d} n={r['n_on']:3d}  {r['top_ss']:20s} raw={r['raw_purity']:.3f} adj={r['adj_purity']:+.3f}")

# 저장
out = {'fixed_width': len(selected), 'rho_final': rho_traj[-1],
       'summary': summ, 'per_feature': res, 'base_rate': {s:round(float(base_rate[ss_idx[s]]),4) for s in SS}}
json.dump(out, open('/home/claude/g_absorption_508_result.json','w'), ensure_ascii=False, indent=2)
print("\n[saved] g_absorption_508_result.json")
