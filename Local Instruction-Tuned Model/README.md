# Local Instruction-Tuned Model
Runs `TinyLlama/TinyLlama-1.1B-Chat-v1.0` locally with the Hugging Face Transformers `pipeline`. The script sends a system + user chat ("reply with code only", write a rectangle-area function), generates up to 100 new tokens with sampling off, and prints the reply.
Runs on CPU (`device="cpu"`).

**Run:** `pip install transformers torch`, then `python TinyLlama-1.1B-Chat.py`.
