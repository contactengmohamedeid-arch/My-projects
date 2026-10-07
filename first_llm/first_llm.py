from transformers import pipeline

# 1. استدعاء الموديل (الـ Decoder اللي اتكلمنا عنه)
generator = pipeline('text-generation', model='gpt2')

# 2. التلقين (Prompt)
prompt = "As an AI engineer, my next project will be"

# 3. توليد الكلمات بناءً على السياق (Attention Mechanism)
result = generator(prompt, max_length=50, num_return_sequences=1)

# 4. طباعة النتيجة
print(result[0]['generated_text'])