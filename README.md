# Drission-YT-RAG

A Python command-line interface (CLI) that lets you extract subtitles from YouTube videos, turn them into plain-text transcripts, optionally translate them, and build a Retrieval-Augmented Generation (RAG) pipeline using Google’s Gemini models. Get insights from any video without watching the whole thing.

---

## Features

- **YouTube Subtitle Extraction**  
  Fetches timed text (captions) from a given YouTube video URL.
- **Transcript Generation**  
  Converts JSON subtitle data into a clean `.txt` transcript.
- **Optional Translation**  
  Translates the transcript to English (or another target language) via `googletrans-py`, with a progress bar.
- **Gemini-Powered RAG**  
  Uses LangChain + Google Generative AI (Gemini chat & embeddings) to build an interactive Q&A over your transcript.
- **Context Window Optimization**  
  Employs a `ContextualCompressionRetriever` + `LLMChainExtractor` to pare down the context for more efficient LLM calls.
- **Interactive Q&A**  
  A simple REPL: ask questions about your transcript, get grounded answers.
- **Flexible Input**  
  Works from either a live YouTube URI or an existing `.txt` transcript file.

---

## Prerequisites

- Python 3.8+
- A Google Generative AI API key (see below)

---

## Installation

1. **Clone the repo** (or just save the `youtube_lang.py` script).  
2. **Install dependencies**:
   ```bash
   pip install requirements.txt
   ```
   - **DrissionPage**: automates YouTube subtitle scraping  
   - **langchain** & **langchain-google-genai**: RAG & Gemini integration  
   - **googletrans-py**: stable translation  
   - **tqdm**: progress bars during translation  

---

## Configuring Your Gemini API Key

This tool relies on Google’s Gemini models. You need an API key:

1. Go to [Google AI Studio](https://aistudio.google.com/apikey).  
2. Create an API key.
3. Provide it via CLI. Refer to [Examples](https://github.com/djmahe4/Drission-YT-RAG/blob/main/README.md#examples)

---

## Usage

```bash
usage: youtube_lang.py [-h] [--uri URI] [--file FILE]
                       [--api-key API_KEY] [--translate]

Options:
  -h, --help            show help message and exit
  --uri URI, -u URI     YouTube Video URI (e.g., `dQw4w9WgXcQ`)
  --file FILE, -f FILE  Path to an existing transcript `.txt` file
  --api-key API_KEY, -a GenAI API key
  --translate, -t       Translate the transcript to English
```

### Examples

1. **Extract + Translate + Query**  
   ```bash
   python youtube_lang.py -u <VIDEO_ID> -t -a YOUR_GEMINI_API_KEY
   ```
   - Opens a headless Chromium via DrissionPage
   - Workflow is not fully automated and reqires user interaction while enabling Closed Captions
   - Saves `VideoTitle.json` → `VideoTitle.txt` → `trans_VideoTitle.txt`  
   - Launches interactive Q&A  

2. **Existing Transcript + Translate + Query**  
   ```bash
   python youtube_lang.py -f my_transcript.txt -t -a YOUR_GEMINI_API_KEY
   ```

3. **Existing Transcript (No Translation)**  
   ```bash
   python youtube_lang.py -f my_transcript.txt -a YOUR_GEMINI_API_KEY
   ```

---

## Interactive Q&A

Once processing is done, you’ll see a prompt:

```text
Enter prompt (Press ENTER to exit): Can you summarize the video?
--- Response ---
[AI-generated summary]

Enter prompt (Press ENTER to exit): What are the main points about X?
--- Response ---
[Answer]

Enter prompt (Press ENTER to exit):
```

Hit **ENTER** on an empty line to quit.

---

## How It Works

1. **Input Handling**  
   - **YouTube URI**: DrissionPage opens a browser, captures the timed-text API call, and saves the JSON.
   - **Transcript File**: Reads your `.txt` straight away.
2. **Post-Processing**  
   - `postLoad` parses and concatenates subtitle JSON → `.txt`.
3. **Translation (optional)**  
   - Chunks text + translates via `googletrans-py` → `trans_<orig>.txt`.
   - By default translation is set from any language, but only to English
4. **RAG Pipeline**  
   - **Splitter**: `RecursiveCharacterTextSplitter` → overlapping chunks.  
   - **Embeddings**: `GoogleGenerativeAIEmbeddings` → vectors.  
   - **Vector Store**: In-memory FAISS for similarity search.  
   - **Compressor**: `ContextualCompressionRetriever` + `LLMChainExtractor` → slim context.  
   - **LLM**: `ChatGoogleGenerativeAI` ("gemini-2.5-flash-preview-04-17") with a `PromptTemplate`.  
   - **Chain**: A `RunnableParallel` orchestrates retrieval → extraction → chat.
5. **Interactive Loop**  
   Each prompt invokes the RAG chain and returns a context-grounded answer.

---

## Notes & Considerations

- **Manual CC Click**: Sometimes you must click the “CC” button in the browser for DrissionPage to capture subtitles.  
- **Translation Stability**: `googletrans-py` is more stable than the original but may still hiccup under heavy load.  
- **Model Versions**: `embedding-001` & `gemini-2.5-flash-preview-04-17` may change—update if deprecated.  
- **URI vs File**: For reliability, using a pre‐downloaded transcript file can be smoother than live scraping.

---

## Contributing

Contributions welcome! Feel free to open issues or PRs to:

- Improve error handling  
- Add support for newer Gemini versions  
- Enhance the UI/UX  
- Integrate other translation or embedding backends  

---

> _This project is maintained under the MIT License. See [LICENSE](./LICENSE) for details._
