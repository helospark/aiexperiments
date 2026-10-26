from kitten_tts_cpp import KittenTTS
import soundfile as sf

model = KittenTTS("KittenML/kitten-tts-2", decoder="student_w4", threads=4)

audio = model.generate("Hi! umm.. how are you, I guess?  .. uh.. nice to meet you??",
                   voice="WeatherPresenterCheerful")
sf.write("output.wav", audio, model.sample_rate)

print(model.last_report)  # Generation timings and token IDs.
