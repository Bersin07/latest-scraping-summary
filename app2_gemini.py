from flask import Flask, render_template, request, jsonify
import asyncio
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
from duckduckgo_search import DDGS
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.models import CrawlResult
import chromadb
import tempfile
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama
import sys
import time
from duckduckgo_search.exceptions import DuckDuckGoSearchException
import requests
import google.generativeai as genai



# Replace these with your Google API Key and Custom Search Engine (CSE) ID
GOOGLE_API_KEY = "AIzaSyCywxFplOkVXs7kklrII66oSDNyp4kkgLg"
GOOGLE_CSE_ID = "64888ddbc66cd43b6"

# Configure Gemini API
GENAI_API_KEY = "AIzaSyBjOR_Ph4gU7sLkRbiBx191PJNvt16fit0"
genai.configure(api_key=GENAI_API_KEY)

# Set event loop policy for Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

app = Flask(__name__)

# Add custom Jinja2 filter for starts_with
@app.template_filter('starts_with')
def starts_with(string, prefixes):
    """
    Custom Jinja2 filter to check if a string starts with any of the given prefixes.
    """
    if not isinstance(string, str) or not prefixes:
        return False
    if not isinstance(prefixes, (list, tuple)):
        prefixes = [prefixes]
    return any(string.startswith(prefix) for prefix in prefixes)

# System prompt for the LLM
system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context.
Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

Context will be passed as "Context:"
User question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. When the context supports an answer, ensure your response is clear, concise, and directly addresses the question.
5. When there is no context, just say you have no context and stop immediately.
6. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.
7. Avoid explaining why you cannot answer or speculating about missing details. Simply state that you lack sufficient context when necessary.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.
6. Do not mention what you received in context, just focus on answering based on the context.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""

# Query and response storage in ChromaDB
def get_query_response_collection():
    """
    Initializes or retrieves the ChromaDB collection for storing user queries and responses.
    """
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings", model_name="mxbai-embed-large"
    )
    chroma_client = chromadb.PersistentClient(
        path="./query-response-db", settings=Settings(anonymized_telemetry=False)
    )
    collection_name = "query_responses"

    try:
        collection = chroma_client.get_collection(collection_name)
    except:
        collection = chroma_client.create_collection(
            name=collection_name, embedding_function=ollama_ef, metadata={"hnsw:space": "cosine"}
        )

    return collection, chroma_client



def store_query_response(query, response):
    """
    Stores the user query and generated response in ChromaDB.
    """
    collection, _ = get_query_response_collection()
    timestamp = str(int(time.time()))  # Unique identifier for the query-response pair

    # Normalize whitespace before storing
    query = " ".join(query.strip().split())
    response = " ".join(response.strip().split())

    collection.upsert(
        documents=[query, response],
        metadatas=[{"type": "query"}, {"type": "response"}],
        ids=[f"query_{timestamp}", f"response_{timestamp}"]
    )

async def fetch_similar_past_query(query):
    """
    Checks if a similar query has been answered before and retrieves its response.
    """
    collection, _ = get_query_response_collection()
    results = collection.query(query_texts=[query], n_results=1)

    if results.get("documents") and len(results["documents"]) > 0:
        stored_data = results["documents"][0]  # Get first query-response pair
        if len(stored_data) >= 2:  # Ensure both query and response exist
            stored_query = stored_data[0]
            stored_response = stored_data[1]
            return stored_response if stored_query.lower() == query.lower() else None

    return None  # Return None if no past query is found

# Vector database for web search context
def get_vector_collection() -> tuple[chromadb.Collection, chromadb.Client]:
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings", model_name="mxbai-embed-large"
    )
    chroma_client = chromadb.PersistentClient(
        path="./web-search-llm-db", settings=Settings(anonymized_telemetry=False)
    )
    collection_name = "web_llm"
    
    if collection_name in chroma_client.list_collections():
        print(f"✅ Using existing collection: {collection_name}")
        collection = chroma_client.get_collection(collection_name)
    else:
        print(f"🛠 Creating new ChromaDB collection: {collection_name}")
        collection = chroma_client.create_collection(
            name=collection_name, embedding_function=ollama_ef, metadata={"hnsw:space": "cosine"}
        )
    return collection, chroma_client

def normalize_url(url):
    return (
        url.replace("https://", "").replace("www.", "").replace("/", "_")
        .replace("-", "_").replace(".", "_")
    )


def google_search(query, num_results=10, retries=3):
    """
    Perform a Google search using the official Google Custom Search API.
    """
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "num": min(num_results, 10),  # Google limits to 10 results per request
    }

    for attempt in range(retries):
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            results = response.json().get("items", [])
            return [result["link"] for result in results]
        except requests.exceptions.RequestException as e:
            print(f"Google Search Error: {e}. Retrying ({attempt + 1}/{retries})...")
            time.sleep(2**attempt)  # Exponential backoff

    return []


async def get_web_urls(search_term: str, num_results: int = 15):
    """
    Fetch web URLs using Google Search instead of DuckDuckGo.
    """
    web_urls = google_search(search_term, num_results=num_results)
    return web_urls

async def check_robots_txt_async(urls: list[str]) -> list[str]:
    allowed_urls = []
    blocked_sites = [
        "presidentsusa.net",  # Site causing too many redirects
    ]

    for url in urls:
        try:
            if any(blocked in url for blocked in blocked_sites):
                print(f"Skipping blocked site: {url}")
                continue  # Skip known problematic URLs
            
            robots_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}/robots.txt"
            rp = RobotFileParser(robots_url)
            rp.read()
            
            if rp.can_fetch("*", url):
                allowed_urls.append(url)
        except Exception:
            allowed_urls.append(url)  # Assume allowed if there's an error

    return allowed_urls

async def crawl_webpages(urls: list[str], prompt: str) -> list[CrawlResult]:
    bm25_filter = BM25ContentFilter(user_query=prompt, bm25_threshold=0.8)
    md_generator = DefaultMarkdownGenerator(content_filter=bm25_filter)
    crawler_config = CrawlerRunConfig(
        markdown_generator=md_generator, excluded_tags=["nav", "footer", "header", "form", "img", "a"],
        only_text=True, exclude_social_media_links=True,
        remove_overlay_elements=True, user_agent="Mozilla/5.0",
        cache_mode=CacheMode.BYPASS,  # Prevent re-fetching recent pages
        page_timeout=40000,  # Increase to 40 seconds
    )
    browser_config = BrowserConfig(headless=True, text_mode=True, light_mode=True)
    
    results = []
    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            results = await crawler.arun_many(urls, config=crawler_config)
    except Exception as e:
        print(f"⚠ Web Crawler Error: {e}")
    return results

def add_to_vector_database(results: list[CrawlResult]):
    collection, _ = get_vector_collection()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150, separators=["\n\n", "\n", ".", "?", "!", ""])
    
    for result in results:
        if not result.markdown_v2:
            continue
        markdown_result = result.markdown_v2.fit_markdown
        temp_file = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8")
        temp_file.write(markdown_result)
        temp_file.flush()
        loader = UnstructuredMarkdownLoader(temp_file.name, mode="single")
        docs = loader.load()
        all_splits = text_splitter.split_documents(docs)
        normalized_url = normalize_url(result.url)
        documents, metadatas, ids = [], [], []
        
        for idx, split in enumerate(all_splits):
            documents.append(split.page_content)
            metadatas.append({"source": result.url})
            ids.append(f"{normalized_url}_{idx}")
        
        if documents:
            collection.upsert(documents=documents, metadatas=metadatas, ids=ids)


def format_llm_response(response_text: str) -> str:
    """
    Formats the LLM response into HTML for better readability.
    """
    if not response_text or "no context" in response_text.lower():
        return "<p>No response available.</p>"

    lines = [line.strip() for line in response_text.split('\n') if line.strip()]
    if not lines:
        return f"<p>{response_text}</p>"

    formatted_html = []
    in_list = False

    for line in lines:
        # Handle list items
        if line.startswith(('- ', '* ')):
            if not in_list:
                formatted_html.append("<ul>")
                in_list = True
            formatted_html.append(f"<li>{line[2:].strip()}</li>")
        else:
            if in_list:
                formatted_html.append("</ul>")
                in_list = False
            formatted_html.append(f"<p>{line}</p>")

    if in_list:
        formatted_html.append("</ul>")

    return ''.join(formatted_html)


# def call_llm(prompt: str, with_context: bool = True, context: str | None = None):
#     """
#     Calls the Ollama LLM with the given prompt and context, returning a formatted response.
#     """
#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": f"Context: {context}, Question: {prompt}"},
#     ]
#     if not with_context:
#         messages.pop(0)
#         messages[0]["content"] = prompt

#     response = ollama.chat(model="gemma3", stream=True, messages=messages)
#     accumulated_response = ""
#     for chunk in response:
#         if chunk["done"] is False:
#             accumulated_response += chunk["message"]["content"] + " "
#     # Normalize whitespace: remove extra spaces and ensure single spaces between words
#     print(f"Debug - Raw LLM Response: {accumulated_response}")  # Add debug print
#     cleaned_response = " ".join(accumulated_response.strip().split())
#     return cleaned_response.strip()


def call_llm(prompt: str, with_context: bool = True, context: str | None = None):
    """
    Calls the Gemini 2.0 API with the given prompt and optional context.
    Returns the formatted response or an error message.
    """
    try:
        model = genai.GenerativeModel("gemini-2.0-flash-001")

        # Construct message payload properly
        if with_context and context:
            full_prompt = f"Context: {context}\n\nQuestion: {prompt}"
        else:
            full_prompt = prompt

        response = model.generate_content([{"text": full_prompt}])  # Correct structure

        # Extract response text safely
        if response and hasattr(response, "text"):
            return response.text.strip()
        else:
            return "Error: No response generated."

    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        return "Error: Unable to process the request."


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form.get('prompt')
        is_web_search = request.form.get('web_search') == 'on'

        if not prompt:
            return render_template('index.html', response="Please enter a prompt.", prompt=prompt)

        # Initialize async event loop for safe execution
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Step 1: Check if a similar query exists in ChromaDB
        stored_response = loop.run_until_complete(fetch_similar_past_query(prompt))
        if stored_response:
            formatted_response = format_llm_response(stored_response)
            print(f"Retrieved stored response: {formatted_response}")  # Debug print
            return render_template('index.html', response=formatted_response, prompt=prompt)

        # Step 2: Handle Web Search and Crawling (ONLY IF Web Search is Enabled)
        if is_web_search:
            web_urls = loop.run_until_complete(get_web_urls(prompt))
            results = loop.run_until_complete(crawl_webpages(web_urls, prompt)) if web_urls else []
            add_to_vector_database(results)

            # Retrieve stored context from ChromaDB
            collection, _ = get_vector_collection()
            qresults = collection.query(query_texts=[prompt], n_results=15)
            context = qresults.get("documents", [[]])
            context = context[0] if context else ""
        else:
            context = None

        # Step 3: Generate response using LLM
        raw_response = call_llm(prompt=prompt, with_context=is_web_search, context=context)
        print(f"Raw LLM response: {raw_response}")  # Debug print

        # Step 4: Format the response for HTML display
        formatted_response = format_llm_response(raw_response)
        print(f"Formatted response: {formatted_response}")  # Debug print

        # Step 5: Store query-response in ChromaDB (store raw text for consistency)
        store_query_response(prompt, raw_response)

        # Step 6: Render response to frontend
        return render_template('index.html', response=formatted_response, prompt=prompt)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5050)