import argparse
import os
from DrissionPage import ChromiumPage
import json
import time
from pathlib import Path
#from langchain_community.vectorstores import FAISS
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
# New imports for context window optimization
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from googletrans import Translator
import asyncio
from tqdm import tqdm

def create_main_chain(fpath):
    # 1. Load and Split Document
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Assuming 'fpath' txt file is in the same directory as this script
    # Or provide the full path if it's elsewhere
    try:
        file_path = fpath # Adjust path if necessary
        with open(file_path, "r", encoding="utf-8") as file:
            transcript = file.read()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        print(f"Please ensure '{fpath}' is in the correct directory.")
        exit() # Exit if the file isn't found

    chunks = splitter.create_documents([transcript])
    print(f"Number of chunks created: {len(chunks)}")

    # 2. Initialize Embeddings and Vector Store with Google Generative AI
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(chunks, embeddings)
    base_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # 3. Initialize Chat Model with Gemini 2.5 Flash
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=1.0)

    # --- Context Window Optimization Implementation ---
    # 4. Create a compressor for the retrieved documents
    # We'll use the same LLM for compression, but you could use a smaller, faster one if needed.
    compressor = LLMChainExtractor.from_llm(llm)

    # 5. Create a ContextualCompressionRetriever
    # This retriever will first get the documents using base_retriever,
    # then pass them through the compressor to extract relevant parts.
    compressed_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    # --- End Context Window Optimization Implementation ---

    # 6. Define Prompt Template
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        Context: {context}
        Question: {question}
        """,
        input_variables=['context', 'question']
    )

    # 7. Define Document Formatter
    def format_docs(retrieved_docs):
        """Formats the retrieved documents into a single string."""
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        return context_text

    # 8. Construct the RAG Chain using the compressed_retriever
    parallel_chain = RunnableParallel({
        'context': compressed_retriever | RunnableLambda(format_docs), # Use compressed_retriever here
        'question': RunnablePassthrough()
    })
    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | llm | parser
    return main_chain
async def new_translate_text_file(input_filepath, output_filepath="", dest_lang='en', src_lang='auto', chunk_size=4000):
    """
    Translates the content of a text file line by line using googletrans asynchronously.

    Args:
        input_filepath (str): The path to the input text file.
        output_filepath (str): The path to save the translated text file.
        dest_lang (str): The target language code (e.g., 'es' for Spanish, 'fr' for French).
        src_lang (str): The source language code (e.g., 'en' for English).
                        'auto' detects the source language automatically.
    """
    output_filepath=f"trans_{Path(input_filepath).stem}.txt"
    translator = Translator()
    #translated_lines = []

    try:
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            content=infile.read()

        print(f"Translating '{input_filepath}' from {src_lang.upper()} to {dest_lang.upper()}...")

        try:
            chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
            translated_chunks = []

            print(f"Translating: {os.path.basename(input_filepath)}")
            for chunk in tqdm(chunks, desc="  Chunks", leave=False):
                result = await translator.translate(chunk, src=src_lang, dest=dest_lang)
                translated_chunks.append(result.text)

            translated_content = ' '.join(translated_chunks)
        except Exception as e:
            with open("error_log.txt", "a", encoding="utf-8") as log:
                log.write(f"{input_filepath}: {e}\n")
            translated_content = content

        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            #for translated_line in translated_lines:
                #outfile.write(translated_line + '\n')
            outfile.write(translated_content)

        print(f"\nTranslation complete! Translated text saved to '{output_filepath}'")
        return output_filepath

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_filepath}'")
        return None
    except Exception as e:
        print(f"An unexpected error during translation occurred: {e}")
        return None

def postLoad(jfile="LearningtoHackA.json"):
    with open(jfile,"r") as JSON:
        jdata=json.load(JSON)
        for event in jdata['events'][1:]:
            stime=event['tStartMs']//1000
            if len(event['segs'])<=1:
                continue
            text=f""
            for seg in event['segs']:
                text+=seg['utf8']
            text+=" "
            with open(f'{Path(jfile).stem}.txt',"a") as file:
                file.write(text)
    return f'{Path(jfile).stem}.txt'
def loader(uri="tYxJMr8jAgo"):

    # Initialize ChromiumPage
    page = ChromiumPage()

    # Open the target YouTube video
    page.get(f"https://www.youtube.com/watch?v={uri}&cc_load_policy=1")
    #time.sleep(10)
    # Start listening to network logs
    page.listen.start()
    # Try to skip ads
    # try:
    #     page.wait.ele_displayed(".ytp-skip-ad-button", timeout=5)
    #     page(".ytp-skip-ad-button").click()
    #     print("Ad skipped.")
    # except:
    #     print("No ad skip button found.")
    # Wait for network requests to collect
    print("Waiting for network requests...")
    time.sleep(10)
    found=False
    # Filter and display subtitle API responses
    while not found:
        print("!! Click 'CC' button manually to extract subtitles")
        for step in page.listen.steps():
            if hasattr(step, 'response') and step.response:
                url = step.response.url
                try:
                    content_type = step.response.headers.get('Content-Type', '')
                except:
                    print(url)
                    continue
                if 'api' in url.lower() and 'timedtext' in url.lower() and 'json' in content_type:
                    print(f"API URL: {url}")
                    print(f"Status Code: {step.response.status_code}")
                    try:
                        #body = step.response.body.decode('utf-8')
                        #json_data = json.loads(body)
                        #print("JSON Response:", json.dumps(json_data, indent=2))
                        print("JSON Response:", step.response.body)
                    except Exception as e:
                        print("Failed to parse JSON:", e)
                    print("=" * 60)
                    time.sleep(5)
                    found=True
                    try:
                        # Decode and parse JSON
                        body = step.response.body
                        json_string = json.dumps(body, indent=2, ensure_ascii=False)
                        tit=page.title.replace(" ","")
                        # Dump to a file
                        filename = f"{tit}.json"  # You can generate dynamic names if needed
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write(json_string)

                        print(f"✅ JSON saved to {filename}")
                        page.quit()
                        return f"{tit}.json"
                    except Exception as e:
                        print("❌ Failed to save JSON:", e)
                    break
        if not found:
            print("No subtitle API response found yet, retrying...")
            time.sleep(5)
            page.refresh()
    # Optional:
    page.quit()


yt_lang=argparse.ArgumentParser(description="Youtube langchain Tool",epilog="""
Examples:
                                python youtube_lang.py -u uri -t -a GEMINI-API-KEY
                                python youtube_lang.py -f title.txt -t -a GEMINI-API-KEY
""")
yt_lang.add_argument("--uri","-u",type=str,help="Youtube Video URI",)
yt_lang.add_argument("--file","-f",type=str,default="",help=".txt file if available")
yt_lang.add_argument("--api-key","-a",help="Genai API key from https://aistudio.google.com/apikey")
yt_lang.add_argument("--translate","-t", action="store_true",help="To translate to EN")
# 3. Parse the arguments
args = yt_lang.parse_args()
if args.api_key:
        print(f"Received API Key: {args.api_key}")
        # In a real application, you would now use this key to authenticate
        # e.g., make an API call, set an environment variable for a library, etc.
        os.environ["GOOGLE_API_KEY"] = args.api_key
        if args.uri:
            json_name=loader(args.uri)
            if json_name:
                txt_name=postLoad(json_name)
                if args.translate:
                        translated_file = asyncio.run(new_translate_text_file(
                            input_filepath=txt_name
                        ))
                        if translated_file:
                            trans = translated_file # Use the translated file for RAG
                            bot=create_main_chain(trans)
                        else:
                            print("Translation failed. Exiting.")
                else:
                    bot=create_main_chain(txt_name)
            else:
                print("Error while creating json file")
                exit()
        elif args.file:
            txt_name=args.file
            # --- AWAIT THE ASYNC TRANSLATION FUNCTION HERE ---
            # The result of new_translate_text_file is the path string, not a coroutine
            if args.translate:
                translated_file = asyncio.run( new_translate_text_file(
                    input_filepath=txt_name
                ))
                if translated_file:
                    trans = translated_file # Use the translated file for RAG
                    bot=create_main_chain(trans)
                else:
                    print("Translation failed. Exiting.")
                    exit()
                    
            else:
                bot=create_main_chain(txt_name)
        else:
            print("Invalid entry")
            print("Usage example: python youtube_lang.py -u uri -t -a GEMINI-API-KEY")
        if bot:
            while True:
                # 9. Invoke the Chain with a Question
                question = input("Enter prompt (Press ENTER to exit): ")#'eg: Can you summarize the video'
                if not question:
                    break
                print(f"\nAsking the question: '{question}'")
                try:
                    response = bot.invoke(question)
                    print("\n--- Response ---")
                    print(response)
                except Exception as e:
                    print(f"Exception occured: {e}")

else:
    print("No API key provided via --api-key.")
    print("Usage example: python youtube_lang.py -u uri -t -a GEMINI-API-KEY")