# RAG-Based AI Teaching Assistant

A Retrieval-Augmented Generation (RAG) based AI Teaching Assistant that allows users to ask questions about educational video content and receive context-aware answers with relevant video references and timestamps.

The project uses **BGE-M3** to generate text embeddings, **Cosine Similarity** for semantic search, and **Llama 3.2** to generate natural-language responses using the retrieved video transcripts.

## Features

- Converts educational videos into audio files.
- Transcribes audio into structured JSON data.
- Generates semantic embeddings using Ollama's BGE-M3 model.
- Retrieves the most relevant video transcript chunks using cosine similarity.
- Generates context-aware answers using Llama 3.2.
- Provides relevant video titles and timestamps to help users locate the content.

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

The project follows a five-step RAG pipeline:

1. **Video Collection:** Collect educational videos and store them in the `videos` folder.
2. **Audio Extraction:** Convert video files into MP3 audio files.
3. **Transcription:** Convert audio files into structured JSON transcripts containing text and timestamps.
4. **Embedding Generation:** Process JSON files, generate embeddings using BGE-M3, and save the resulting dataframe as a Joblib pickle file.
5. **Retrieval & Response Generation:** Convert the user's query into an embedding, retrieve the most relevant transcript chunks using cosine similarity, and pass the retrieved context to Llama 3.2 to generate an answer.

## How to Use This Project with Your Own Data

### Prerequisites

- Python installed on your system.
- FFmpeg installed and configured.
- Ollama installed and running.
- Required Python libraries installed.

### Step 1: Collect Your Videos

Move all your educational video files into the `videos` folder.

### Step 2: Convert Videos to MP3

Run the `video_to_mp3` script to extract audio from all video files and convert them into MP3 format.

### Step 3: Convert MP3 Files to JSON

Run the `mp3_to_json` script to transcribe the audio files and generate JSON files containing the extracted text and timestamps.

### Step 4: Generate Embeddings

Run the `preprocess_json` script to:

- Read and process the generated JSON files.
- Convert transcript text into embeddings using BGE-M3.
- Store the video metadata, timestamps, text, and embeddings in a Pandas DataFrame.
- Save the processed data as a Joblib pickle file.

### Step 5: Ask Questions Using the RAG Assistant

Run the `process_incoming.py` script.

The system will:

- Load the saved embeddings dataframe.
- Accept a question from the user.
- Generate an embedding for the question.
- Retrieve the top 5 most relevant transcript chunks using cosine similarity.
- Construct a context-aware prompt using the retrieved information.
- Send the prompt to Llama 3.2.
- Display the generated answer with relevant video references and timestamps.

## Ollama Model Setup

Make sure Ollama is installed and running. Pull the required models using:

```bash
ollama pull bge-m3
ollama pull llama3.2
```

## Running the Project

After completing the data preparation steps and generating the embeddings file, run:

```bash
python process_incoming.py
```

Enter your question when prompted and receive an AI-generated answer based on your educational video content.

## Future Improvements

- Build a web-based interface for easier interaction.
- Add direct video playback from the recommended timestamps.
- Improve retrieval accuracy through better chunking and ranking strategies.
- Support additional document formats such as PDFs and text files.
- Implement conversation history for follow-up questions.

---

**Built with Python, Ollama, BGE-M3, and Llama 3.2.**