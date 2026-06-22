"""
embed_only.py  —  [당신 PC에서 실행]
====================================
이미 만들어진 world_entities.json(개체 1658개)을 읽어 임베딩만 생성.
NER을 다시 안 돌린다.

준비 (PowerShell):
  $env:OPENAI_API_KEY="sk-proj-..."      # 따옴표 포함, set 아님
  python scripts\embed_only.py

먼저 키 확인:
  echo $env:OPENAI_API_KEY               # 값이 보여야 함

출력: world_entities_emb.npy  (json과 행 정렬, L2 정규화)
→ world_entities.json 과 이 npy 두 개를 Claude에 업로드.
"""
import os, json, time
import numpy as np

IN_JSON  = r"C:\CL_token_experiment\data\diverse\world_entities.json"
OUT_NPY  = r"C:\CL_token_experiment\data\diverse\world_entities_emb.npy"
EMB_MODEL = "text-embedding-3-large"

def main():
    # 키 사전 점검 — 없으면 즉시 멈춤(긴 작업 전에)
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("[stop] OPENAI_API_KEY 환경변수가 비어 있음. "
                         'PowerShell: $env:OPENAI_API_KEY="sk-..." 먼저 실행.')
    from openai import OpenAI
    client = OpenAI()

    entities = json.load(open(IN_JSON, encoding="utf-8"))
    texts = [e["text"] for e in entities]
    print(f"[embed] {len(texts)}개 개체 임베딩 (model={EMB_MODEL})")

    # 작은 테스트 호출 1건 — 키 유효성 먼저 확인(전량 돌리기 전)
    try:
        client.embeddings.create(model=EMB_MODEL, input=["test"])
        print("[embed] 키 유효 확인 OK")
    except Exception as e:
        raise SystemExit(f"[stop] 키 검증 실패: {str(e)[:200]}")

    vecs = []
    B = 256
    for i in range(0, len(texts), B):
        chunk = texts[i:i+B]
        r = client.embeddings.create(model=EMB_MODEL, input=chunk)
        vecs.extend([d.embedding for d in r.data])
        print(f"  embedded {min(i+B,len(texts))}/{len(texts)}")
        time.sleep(0.2)

    arr = np.asarray(vecs, dtype=np.float32)
    arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)   # 508과 동일 규약
    np.save(OUT_NPY, arr)
    print(f"[out] {OUT_NPY}  shape={arr.shape}")
    print("\nworld_entities.json + world_entities_emb.npy 두 파일을 Claude에 업로드하세요.")

if __name__ == "__main__":
    main()
