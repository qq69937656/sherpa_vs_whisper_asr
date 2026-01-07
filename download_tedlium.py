from datasets import load_dataset

gs = load_dataset("speechcolab/gigaspeech", "xs", use_auth_token=True)

# see structure
print(gs)

# load audio sample on the fly
audio_input = gs["train"][0]["audio"]  # first decoded audio sample
transcription = gs["train"][0]["text"]  # first transcription
