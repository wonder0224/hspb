"""
embed_other_models.py  —  [당신 PC에서 실행]
=============================================
모델 독립성 검증: 같은 oncology 토큰을 다른 임베딩 모델로 재임베딩.
text-embedding-3-large(OpenAI)에서 본 sublinear가 BERT/E5에서도 나오나.
같은 토큰·다른 모델 → 모델만의 효과 분리.

준비:
  pip install sentence-transformers numpy
  (GPU 있으면 자동 사용, 없어도 CPU로 됨 — 4629개라 CPU도 몇 분)

입력 : C:\CL_token_experiment\data\oncology\oncology_units.json (토큰 텍스트)
출력 : oncology_emb_e5.npy, oncology_emb_mpnet.npy  (각 N x dim, L2정규화)
       → Claude에 업로드(둘 다, 또는 하나씩).

두 모델:
  e5-large-v2     (1024d, 검색특화, 'query:' 프리픽스 권장하나 여기선 순수비교 위해 생략)
  all-mpnet-base-v2 (768d, BERT계열 SBERT, 범용 의미표현)
차원 달라도 sublinear 측정은 차원무관.
"""
import json, numpy as np
from sentence_transformers import SentenceTransformer

UNITS_PATH = r"C:\CL_token_experiment\data\oncology\oncology_units.json"
OUT_DIR    = r"C:\CL_token_experiment\data\oncology"

units = json.load(open(UNITS_PATH, encoding="utf-8"))
texts = [u["text"] for u in units]
print(f"[load] {len(texts)} oncology 토큰")

MODELS = {
    "e5":    "intfloat/e5-large-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}

for tag, name in MODELS.items():
    print(f"\n[{tag}] 로딩: {name}")
    model = SentenceTransformer(name)
    # e5는 원래 'query: '/'passage: ' 프리픽스 쓰지만,
    # 다른 모델과 동일 조건 비교 위해 순수 텍스트로 통일(편향 없음).
    emb = model.encode(texts, batch_size=64, show_progress_bar=True,
                       normalize_embeddings=True, convert_to_numpy=True)
    emb = emb.astype(np.float32)
    out = f"{OUT_DIR}\\oncology_emb_{tag}.npy"
    np.save(out, emb)
    print(f"[{tag}] saved {out}  shape={emb.shape}")

print("\n완료. oncology_emb_e5.npy, oncology_emb_mpnet.npy 를 Claude에 업로드하세요.")
print("(oncology_units.json은 Claude가 이미 갖고 있으면 재업로드 불필요)")
