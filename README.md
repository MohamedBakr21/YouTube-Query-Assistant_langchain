# YouTube-Query-Assistant_langchain

A Streamlit app that processes YouTube video transcripts and answers queries using LangChain, Chroma, and a local embedding model.

# Setup:

1.Install dependencies:
pip install -r requirements.txt

2.Run the app: 
streamlit run app.py

# Usage:

1.Enter a YouTube video URL and a query.

2.Click "Generate" to get an answer based on the video's transcript.

<img width="1904" height="923" alt="image" src="https://github.com/user-attachments/assets/425ed102-b1c2-4861-bf94-b99560f31df5" />


# YouTube Video Loading with LangChain

This project uses LangChain’s built-in YoutubeLoader to extract transcript text directly from YouTube videos that have available subtitles. Instead of relying on audio conversion or Whisper, this loader retrieves the text transcript (if subtitles are available), making it lightweight and fast. After loading the video, the transcript is split into manageable chunks using RecursiveCharacterTextSplitter. These chunks are then embedded using OpenRouter-compatible embeddings and stored in a Chroma vector database, which allows fast and efficient similarity search. When a user submits a query, the system retrieves the most relevant chunks from Chroma and sends them, along with the question, to a language model (Gemma via OpenRouter) for answering. This setup provides a smooth experience for interacting with educational or informative video content using natural language.

<img width="971" height="282" alt="image" src="https://github.com/user-attachments/assets/7130eaf4-675c-48ba-8263-8b55db9ebabb" />

1.Document Loading: We load the transcript from a YouTube video using `YoutubeLoader`.

2.Splitting: The text is split into smaller chunks using `RecursiveCharacterTextSplitter`.

3.Storage: The chunks are converted into embeddings and stored in a Chroma vector database.

4.Retrieval: When a user asks a question, we search the vector store for the most relevant chunks.

5.Output: The relevant chunks + question are sent to the LLM (Gemma), and the answer is returned.
