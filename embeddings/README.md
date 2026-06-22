# Embeddings

The `.npy` embedding matrices are archived at Zenodo (see the DOI in the main
README) and are git-ignored here because of their size (~140 MB total).

To regenerate them locally from the token inventories in `../data/`:

    python ../scripts/embed_only.py          # text-embedding-3-large (needs OPENAI_API_KEY)
    python ../scripts/embed_other_models.py  # e5-large-v2, all-mpnet-base-v2

Place the resulting files here. Expected filenames are listed in the main README.
