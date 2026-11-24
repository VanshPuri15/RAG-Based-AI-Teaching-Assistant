# RAG-Based AI Teaching Assistant

A Retrieval-Augmented Generation (RAG) based AI Teaching Assistant that allows users to ask questions about educational video content and receive context-aware answers with relevant video references and timestamps.

The project uses **BGE-M3** for generating text embeddings, **Cosine Similarity** for semantic retrieval, and **Llama 3.2** for generating natural-language responses using the retrieved video content.

## Features

- Converts educational videos into MP3 audio files.
- Transcribes audio into structured JSON files with timestamps.
- Combines smaller transcript chunks into larger, context-rich chunks.
- Generates semantic embeddings using Ollama's BGE-M3 model.
- Retrieves the most relevant transcript chunks using cosine similarity.
- Generates context-aware answers using Llama 3.2.
- Provides relevant video titles and timestamps to help users locate specific topics.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core development |
| Ollama | Running local AI models |
| BGE-M3 | Text embedding generation |
| Llama 3.2 | Answer generation |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Cosine similarity |
| Joblib | Saving and loading embeddings |
| FFmpeg | Video-to-audio conversion |

## Project Workflow

The project follows a five-stage RAG pipeline:

1. **Video Collection:** Collect educational videos and store them in the `videos` folder.

2. **Audio Extraction:** Use `video_to_mp3.py` to extract audio from video files and convert them into MP3 format.

3. **Transcription:** Use `mp3_to_json.py` to transcribe audio files into structured JSON files containing transcript text and timestamps.

4. **Chunk Processing and Embedding Generation:**
   - Process the original JSON files and combine every 5 consecutive transcript chunks into a larger chunk.
   - Save the processed files in the `newjsons` folder.
   - Generate embeddings for the processed chunks using BGE-M3.
   - Store the text, video metadata, timestamps, and embeddings in a Pandas DataFrame.
   - Save the dataframe as `embeddings.joblib` using Joblib.

5. **Retrieval and Response Generation:**
   - Load the saved embeddings dataframe.
   - Convert the user's query into an embedding.
   - Retrieve the top 5 most relevant chunks using cosine similarity.
   - Construct a context-aware prompt using the retrieved transcript content.
   - Send the prompt to Llama 3.2.
   - Generate an answer with relevant video references and timestamps.

## How to Use the Project with Your Own Data

### Prerequisites

- Python installed on your system.
- FFmpeg installed and configured.
- Ollama installed and running.
- Required Python libraries installed.

### Step 1: Collect Your Videos

Move your educational video files into the `videos` folder.

### Step 2: Convert Videos to MP3

Run the `video_to_mp3.py` script to extract audio from your video files.

### Step 3: Convert MP3 Files to JSON

Run the `mp3_to_json.py` script to transcribe the audio and generate JSON files containing the extracted text and timestamps.

### Step 4: Process Transcript Chunks

Run the chunk-processing script to:

- Read the JSON files from the `jsons` folder.
- Combine every 5 consecutive transcript chunks into one larger chunk.
- Preserve the video number, title, starting timestamp, and ending timestamp.
- Save the processed JSON files in the `newjsons` folder.

### Step 5: Generate Embeddings

Run the `preprocess_json.py` script to:

- Read the processed JSON files from `newjsons`.
- Generate embeddings using BGE-M3.
- Process the transcript chunks in batches.
- Store embeddings and associated metadata in a Pandas DataFrame.
- Save the dataframe as `embeddings.joblib`.

### Step 6: Ask Questions Using the RAG Assistant

Run `process_incoming.py`.

The system will:

- Load `embeddings.joblib`.
- Accept a question from the user.
- Generate an embedding for the question.
- Retrieve the top 5 matching transcript chunks.
- Construct a prompt using the retrieved context.
- Generate a response using Llama 3.2.
- Display the answer with relevant video references and timestamps.

## Ollama Model Setup

Install Ollama and download the required models:

```bash
ollama pull bge-m3
ollama pull llama3.2
```

Make sure the Ollama service is running before executing the scripts.

## Running the Project

After completing the preprocessing steps and generating `embeddings.joblib`, run:

```bash
python process_incoming.py
```

Enter your question when prompted and receive an AI-generated answer based on your educational video content.

## Future Improvements

- Build a web-based interface for easier interaction.
- Add direct video playback from recommended timestamps.
- Improve retrieval accuracy through better chunking and ranking strategies.
- Support additional document formats such as PDFs and text files.
- Implement conversation history for follow-up questions.

---

**Built with Python, Ollama, BGE-M3, and Llama 3.2.**