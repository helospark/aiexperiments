from kitten_tts_cpp import KittenTTS

model = KittenTTS("KittenML/kitten-tts-2", decoder="student_w4", threads=8)
model.generate_to_file("One day, a little girl named Lily found a needle in her room.", "hello.wav", voice="Bruno")
print(model.last_report)  # Generation timings and token IDs.
