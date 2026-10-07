from transformers import pipeline

# 1. استدعاء موديل المحادثة وتوجيهه للعمل على كارت الشاشة
print("Loading model on RTX 4050...")
generator = pipeline(
    "text-generation", 
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device="cpu"  # السر كله هنا لتشغيل الموديل على الـ GPU
)

# 2. هيكل المحادثة (الـ Chat Template)
# هنا إنت بتدي الموديل "دور" وبعدين بتديه "أمر"
messages = [
    {"role": "system", "content": "You are a smart Python developer. Reply with code only."},
    {"role": "user", "content": "Write a Python function to calculate the area of a rectangle."}
]

# 3. توليد الإجابة
print("Generating response...")
result = generator(
    messages, 
    max_new_tokens=100, 
    do_sample=False  # خليناها False عشان محتاجين كود دقيق مش إبداع عشوائي
)

# 4. طباعة الرد الخاص بالموديل فقط
print("\n--- AI Response ---")
print(result[0]['generated_text'][-1]['content'])