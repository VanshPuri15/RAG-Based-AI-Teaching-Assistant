import os
import subprocess

files = os.listdir("videos")

for file in files:
    print(file)

    tutorial_number = file.split("Tutorial #")[1].split(".")[0]
    file_name = file.split(" Sigma Web Development Course")[0]

    print("Tutorial Number:", tutorial_number)
    print("File Name:", file_name)
    subprocess.run(["ffmpeg","-i",f"videos/{file}",f"audios/{tutorial_number}_{file_name}.mp3"])