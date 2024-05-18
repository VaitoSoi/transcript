from os import listdir
from os.path import isfile, join
from transformers import WhisperForConditionalGeneration, WhisperTokenizer, WhisperFeatureExtractor, AutomaticSpeechRecognitionPipeline
import re
from mutagen.mp3 import MP3, HeaderNotFoundError
import python_ms as ms
import time
import datetime
import requests
import yaml
import platform

wh_files = ""
wh_notification = ""
mode = -1

with open("config.yaml", "r") as f:
    webhook = yaml.safe_load(f)
    wh_notification = webhook["notification"]
    wh_files = webhook["files"]
    mode = list(map(int, webhook["mode"])) if type(webhook["mode"]) == list else webhook["mode"]

def log(msg: str) -> None:
    if wh_notification:
        requests.post(wh_notification, json = {
            "content": msg
        })
    print(msg)

log(f"Start session at {datetime.datetime.now()} on {platform.node()} -- {platform.system()} {platform.release()}")

model_path = "distil-whisper/distil-large-v3"
model = WhisperForConditionalGeneration.from_pretrained(model_path)
tokenizer = WhisperTokenizer.from_pretrained(model_path)
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
asr_pipeline = AutomaticSpeechRecognitionPipeline(model=model, tokenizer=tokenizer, feature_extractor=feature_extractor)
log(f"Loaded model: {model_path}")

directory = "./archive/"
files = [f for f in listdir(directory) if isfile(join(directory, f)) and f.endswith(".mp3")]
files.sort()
log(f"Found {len(files)} files")

ignore = []
with open("transcripted.txt", "r") as f:
    lines = f.readlines()
    newline = re.compile(r"\n")
    for line in lines:
        tmp = newline.sub("", line)
        if tmp.startswith("#"):
            continue
        ignore.append(tmp)
transcripted = open("transcripted.txt", "a")
skipped_file = open("skipped_files.txt", "w")
log("Loaded ignore files")


for index, file in enumerate(files):
    outfile = f"{file[:-4]}.txt"
    if outfile in ignore:
        log(f"Skipped {file} (in transcripted.txt)")
        continue

    audio_path = join(directory, file)
    try:
        audio_data = MP3(audio_path)
    except HeaderNotFoundError:
        log(f"Skipped {file} (mutagen HeaderNotFoundError)")
        skipped_file.write(f"{file}\n")
        continue
    length = ms(round(audio_data.info.length * 1000))
    if mode != -1 and not (mode[0] <= audio_data.info.length and audio_data.info.length <= mode[1]):
        log(f"Skipped {file} (out of range, {length})")
        skipped_file.write(f"{file}\n")
        continue

    log(f"Transcripting {file} ({length})")
    start = time.time()
    output = asr_pipeline(audio_path)
    total = round((time.time() - start) * 1000)

    outpath = join("./transcripts/", outfile)
    with open(outpath, "w") as f:
        f.write(output["text"])
    log(f"Done after {ms(total)}")
    transcripted.write(f"{outfile}\n")

    if wh_files:
        requests.post(wh_files, files = {
            "file": open(outpath, "rb")
        })
