from kittenml import KittenTTS
import soundfile as sf

model = KittenTTS("KittenML/kitten-tts-2", decoder="student_w4")

audio = model.generate("The reference feeds the model by two independent routes -- a speaker embedding through the model's projection head, and the clip itself as codec tokens in the prompt -- so identity survives even when one route is weak", reference="toclone.wav")

sf.write("cloned_output.wav", audio, model.sample_rate)

#print(model.last_report)  # Generation timings and token IDs.
