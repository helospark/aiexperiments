from kittenml import KittenTTS
import soundfile as sf

m = KittenTTS("KittenML/kitten-tts-2")

audio = m.generate("Hi! umm.. how are you, I guess?  .. uh.. nice to meet you??",
                   voice="WeatherPresenterCheerful")
sf.write("output.wav", audio, m.sample_rate)
