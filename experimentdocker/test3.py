from kitten_tts_cpp import KittenTTS

model = KittenTTS("KittenML/kitten-tts-2", decoder="student_w4", threads=8)
model.generate_to_file("[joyful] We actually won the grant <laugh> I can (((hardly))) believe it!", "hello2.wav", voice="Kiki")
print(model.last_report)  # Generation timings and token IDs.
