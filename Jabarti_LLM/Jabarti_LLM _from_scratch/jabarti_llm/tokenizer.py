"""
tokenizer.py -- loads the bilingual BPE tokenizer trained in ch03
"""

from pathlib import Path

from tokenizers import Tokenizer as BackingTokenizer

DEFAULT_TOKENIZER_PATH = Path(__file__).parent / "assets" / "tokenizer.json"
