# Jabarti LLM (from scratch)
A bilingual Arabic + English GPT-style language model written in PyTorch to learn how each part works.

**Credit:** built while following a YouTube playlist on building an LLM from scratch. The training data is the public Hugging Face dataset `bakrianoo/jabarti-llm-dataset`.

**What is in it**
- `Jabarti_LLM _from_scratch/jabarti_llm/model/attention.py`: multi-head causal self-attention with an optional key/value cache.
- `.../model/embeddings.py`: embeddings.
- `.../config.py`: model settings (default 12 layers, 12 heads, 768 width, 1,024 context, 32,000 vocabulary, tied weights).
- `.../tokenizer.py`: loads the trained BPE tokenizer; special tokens for padding, language ([AR]/[EN]) and chat roles.
- `scripts/`: corpus download and cleaning (`clean_corpus.py`, `Cleaning.py`), BPE tokenizer training (`Tokenizer/build_tokenizer.py`, 32,000 vocabulary), and notebooks for dataset exploration, attention and embeddings.

**Status:** architecture and tokenizer pipeline. No training results are included.
