from flask import Flask, render_template, request
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
import httpx
from tenacity import retry, stop_after_attempt, wait_fixed

# Set event loop policy for Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

app = Flask(__name__)

# Add custom Jinja2 filter for starts_with
@app.template_filter('starts_with')
def starts_with(string, prefixes):
    if not isinstance(string, str) or not prefixes:
        return False
    if not isinstance(prefixes, (list, tuple)):
        prefixes = [prefixes]
    return any(string.startswith(prefix) for prefix in prefixes)

# System prompt for the LLM
system_prompt = """
You are an AI assistant with a persistent memory brain stored in a database. Use the provided context as your memory to deliver precise, detailed, and consistent answers. Prioritize relevant past queries, responses, and web content to enhance your response. If the memory lacks sufficient detail, state this clearly and avoid speculation.
Context: {context}
Question: {question}
"""

# Unified Memory Brain Collection
def get_memory_brain_collection():
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="mxbai-embed-large"
    )
    chroma_client = chromadb.PersistentClient(
        path="./memory_brain_db", settings=Settings(anonymized_telemetry=False)
    )
    collection_name = "memory_brain"
    try:
        collection = chroma_client.get_collection(collection_name)
    except:
        collection = chroma_client.create_collection(
            name=collection_name, embedding_function=ollama_ef, metadata={"hnsw:space": "cosine"}
        )
    return collection, chroma_client
def store_memory(content, metadata):
    collection, _ = get_memory_brain_collection()
    timestamp = str(int(time.time()))
    content = " ".join(content.strip().split())  # Normalize whitespace
    collection.upsert(
        documents=[content],
        metadatas=[metadata],
        ids=[f"{metadata['type']}_{timestamp}"]
    )

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def fetch_relevant_memory(query, n_results=10, similarity_threshold=0.9):
    collection, _ = get_memory_brain_collection()
    try:
        results = collection.query(query_texts=[query], n_results=n_results)
        relevant_memory = []
        for doc, metadata, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            if distance < (1 - similarity_threshold):  # Convert cosine distance to similarity
                relevant_memory.append({"content": doc, "metadata": metadata})
        return relevant_memory
    except Exception as e:
        print(f"Error fetching memory: {e}")
        return []  # Fallback to empty list

# Web Crawling Functions
async def get_web_urls(search_term: str, num_results: int = 15):
    discard_urls = ["youtube.com", "britannica.com", "vimeo.com"]
    for url in discard_urls:
        search_term += f" -site: {url}"
    try:
        time.sleep(1)  # Respect rate limits
        results = DDGS().text(search_term, max_results=num_results)
        return [result["href"] for result in results]
    except DuckDuckGoSearchException as e:
        if "Ratelimit" in str(e):
            print("Rate limit hit. Waiting before retrying...")
            time.sleep(5)
            try:
                results = DDGS().text(search_term, max_results=num_results)
                return [result["href"] for result in results]
            except Exception as e:
                print(f"Retry failed: {e}")
                return []
        else:
            print(f"Search error: {e}")
            return []

async def check_robots_txt_async(urls: list[str]) -> list[str]:
    allowed_urls = []
    blocked_sites = ["presidentsusa.net"]
    for url in urls:
        try:
            if any(blocked in url for blocked in blocked_sites):
                print(f"Skipping blocked site: {url}")
                continue
            robots_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}/robots.txt"
            rp = RobotFileParser(robots_url)
            rp.read()
            if rp.can_fetch("*", url):
                allowed_urls.append(url)
        except Exception:
            allowed_urls.append(url)  # Assume allowed if error
    return allowed_urls

async def crawl_webpages(urls: list[str], prompt: str) -> list[CrawlResult]:
    bm25_filter = BM25ContentFilter(user_query=prompt, bm25_threshold=0.8)
    md_generator = DefaultMarkdownGenerator(content_filter=bm25_filter)
    crawler_config = CrawlerRunConfig(
        markdown_generator=md_generator, excluded_tags=["nav", "footer", "header", "form", "img", "a"],
        only_text=True, exclude_social_media_links=True,
        remove_overlay_elements=True, user_agent="Mozilla/5.0",
        cache_mode=CacheMode.BYPASS, page_timeout=40000
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
        for idx, split in enumerate(all_splits):
            store_memory(split.page_content, {"type": "web_content", "timestamp": str(int(time.time())), "source": result.url})

# Response Formatting
def format_llm_response(response_text: str) -> str:
    if not response_text or "no context" in response_text.lower():
        return "<p>No response available.</p>"
    lines = [line.strip() for line in response_text.split('\n') if line.strip()]
    if not lines:
        return f"<p>{response_text}</p>"
    formatted_html = []
    in_list = False
    for line in lines:
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

# LLM Call
def call_llm(prompt, context):
    messages = [
        {"role": "system", "content": system_prompt.format(context=context, question=prompt)},
        {"role": "user", "content": prompt}
    ]
    response = ollama.chat(model="mistral:instruct", stream=True, messages=messages)
    accumulated_response = ""
    for chunk in response:
        if chunk["done"] is False:
            accumulated_response += chunk["message"]["content"]
    return " ".join(accumulated_response.strip().split())

# Main Route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form.get('prompt')
        is_web_search = request.form.get('web_search') == 'on'
        if not prompt:
            return render_template('index.html', response="Please enter a prompt.", prompt=prompt)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Fetch relevant memory
        memory_chunks = loop.run_until_complete(fetch_relevant_memory(prompt))
        memory_context = "\n".join([chunk["content"] for chunk in memory_chunks])

        # Web search if enabled
        if is_web_search:
            web_urls = loop.run_until_complete(get_web_urls(prompt))
            allowed_urls = loop.run_until_complete(check_robots_txt_async(web_urls))
            results = loop.run_until_complete(crawl_webpages(allowed_urls, prompt)) if allowed_urls else []
            add_to_vector_database(results)
            # Refresh memory with new web data
            memory_chunks = loop.run_until_complete(fetch_relevant_memory(prompt))
            memory_context = "\n".join([chunk["content"] for chunk in memory_chunks])

        # Generate response
        raw_response = call_llm(prompt, memory_context)
        formatted_response = format_llm_response(raw_response)

        # Store query and response
        store_memory(prompt, {"type": "query", "timestamp": str(int(time.time())), "source": "user"})
        store_memory(raw_response, {"type": "response", "timestamp": str(int(time.time())), "source": "llm"})

        return render_template('index.html', response=formatted_response, prompt=prompt)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5050)