from transformers import pipeline, set_seed

# استدعاء الموديل
generator = pipeline('text-generation', model='gpt2')

# التلقين
prompt = "As an AI engineer, my next project will be"

# التوليد مع التحكم في الإبداع
result = generator(
    prompt,
    max_new_tokens=50,        # عدد الكلمات الجديدة اللي هيولدها فقط
    num_return_sequences=1,
    do_sample=True,           # تفعيل الاختيار العشوائي الذكي
    temperature=0.7,          # مؤشر الإبداع
    pad_token_id=50256        # لإخفاء التحذير المزعج
)

# طباعة النتيجة
print(result[0]['generated_text'])