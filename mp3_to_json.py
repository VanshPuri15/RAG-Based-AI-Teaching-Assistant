import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import whisper
import json

model = whisper.load_model("large-v2", device="cuda")
audios=os.listdir("audios")

# ADDED: Create jsons folder if it doesn't exist
os.makedirs("jsons", exist_ok=True)

for audio in audios:
    print(audio)
    if("_" in audio):

        # ADDED: Skip files that are already processed
        if(os.path.exists(f"jsons/{audio}.json")):
            print("Already processed, skipping...")
            continue

        number = audio.split("_")[0]
        title = audio.split("_")[1][:-4]

        result = model.transcribe(audio=f"audios/{audio}",
                                language="hi",
                                task="translate",
                                word_timestamps=False,
                                fp16=True)  # ADDED: Explicit FP16

        chunks=[]
        for segment in result["segments"]:
            chunks.append({"number":number,"title":title,"start":segment["start"],"end":segment["end"],"text":segment["text"]})

        chunks_with_metadata = {"chunks":chunks,"text":result["text"]}

        with open(f"jsons/{audio}.json","w", encoding="utf-8") as f:
            json.dump(chunks_with_metadata,f, ensure_ascii=False, indent=2)
