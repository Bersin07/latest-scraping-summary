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
import aiohttp

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

app = Flask(__name__)

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

# Query storing
def get_query_response_collection():
    ollama_ef = OllamaEmbeddingFunction(url="http://localhost:11434/api/embeddings", model_name="mxbai-embed-large")
    chroma_client = chromadb.PersistentClient(path="./query-response-db", settings=Settings(anonymized_telemetry=False))
    collection_name = "query_responses"
    try:
        collection = chroma_client.get_collection(collection_name)
    except:
        collection = chroma_client.create_collection(name=collection_name, embedding_function=ollama_ef, metadata={"hnsw:space": "cosine"})
    return collection, chroma_client

def store_query_response(query, response):
    collection, _ = get_query_response_collection()
    timestamp = str(int(time.time()))
    collection.upsert(
        documents=[query, response],
        metadatas=[{"type": "query"}, {"type": "response"}],
        ids=[f"query_{timestamp}", f"response_{timestamp}"]
    )

async def fetch_similar_past_query(query):
    collection, _ = get_query_response_collection()
    results = collection.query(query_texts=[query], n_results=1)
    if results.get("documents") and len(results["documents"]) > 0:
        stored_data = results["documents"][0]
        if len(stored_data) >= 2:
            stored_query = stored_data[0]
            stored_response = stored_data[1]
            return stored_response if stored_query.lower() == query.lower() else None
    return None

# LLM and vector database
def call_llm(prompt: str, with_context: bool = True, context: str | None = None):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context: {context}, Question: {prompt}"},
    ]
    if not with_context:
        messages.pop(0)
        messages[0]["content"] = prompt

    try:
        response = ollama.chat(model="mistral:instruct", stream=True, messages=messages)
        accumulated_response = ""
        for chunk in response:
            if chunk["done"] is False:
                accumulated_response += chunk["message"]["content"] + " "
        return accumulated_response.strip()
    except Exception as e:
        return f"Error generating response from LLM: {str(e)}"

def get_vector_collection():
    ollama_ef = OllamaEmbeddingFunction(url="http://localhost:11434/api/embeddings", model_name="mxbai-embed-large")
    chroma_client = chromadb.PersistentClient(path="./web-search-llm-db", settings=Settings(anonymized_telemetry=False))
    collection_name = "web_llm"
    existing_collections = [c.name for c in chroma_client.list_collections()]
    if collection_name in existing_collections:
        print(f"✅ Using existing collection: {collection_name}")
        collection = chroma_client.get_collection(collection_name)
    else:
        print(f"🛠 Creating new ChromaDB collection: {collection_name}")
        collection = chroma_client.create_collection(name=collection_name, embedding_function=ollama_ef, metadata={"hnsw:space": "cosine"})
    return collection, chroma_client

def normalize_url(url):
    return url.replace("https://", "").replace("www.", "").replace("/", "_").replace("-", "_").replace(".", "_")

async def get_web_urls(search_term: str, num_results: int = 15):
    discard_urls = ["youtube.com", "britannica.com", "vimeo.com"]
    for url in discard_urls:
        search_term += f" -site: {url}"
    try:
        results = DDGS().text(search_term, max_results=num_results)
        return [result["href"] for result in results]
    except Exception as e:
        return f"Error during web search: {str(e)}"

async def check_robots_txt_async(urls: list[str]) -> list[str]:
    allowed_urls = []
    blocked_sites = ["presidentsusa.net"]
    async with aiohttp.ClientSession() as session:
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

async def crawl_webpages(urls: list[str], prompt: str) -> CrawlResult:
    bm25_filter = BM25ContentFilter(user_query=prompt, bm25_threshold=0.8)
    md_generator = DefaultMarkdownGenerator(content_filter=bm25_filter)
    crawler_config = CrawlerRunConfig(
        markdown_generator=md_generator, excluded_tags=["nav", "footer", "header", "form", "img", "a"],
        only_text=True, exclude_social_media_links=True, remove_overlay_elements=True,
        user_agent="Mozilla/5.0", cache_mode=CacheMode.BYPASS, page_timeout=40000, max_concurrent=3  # Limit to 3 concurrent crawls
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150, separators=["\n\n", "\n", ".", "?", "!", " "])
    for result in results:
        if not result.markdown_v2:
            continue
        markdown_result = result.markdown_v2.fit_markdown
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as temp_file:
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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form.get('prompt')
        is_web_search = request.form.get('web_search') == 'on'

        if not prompt:
            return render_template('x_index.html', error="Please enter a prompt.", prompt=prompt)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            stored_response = loop.run_until_complete(fetch_similar_past_query(prompt))
            if stored_response:
                return render_template('index.html', response=stored_response, prompt=prompt)

            if is_web_search:
                web_urls = loop.run_until_complete(get_web_urls(prompt))
                if isinstance(web_urls, str):  # Error occurred
                    return render_template('index.html', error=web_urls, prompt=prompt)
                allowed_urls = loop.run_until_complete(check_robots_txt_async(web_urls))
                results = loop.run_until_complete(crawl_webpages(allowed_urls, prompt)) if allowed_urls else []
                add_to_vector_database(results)
                collection, _ = get_vector_collection()
                qresults = collection.query(query_texts=[prompt], n_results=15)
                context = qresults.get("documents", [[]])[0] if qresults.get("documents") else ""
            else:
                context = None

            response = call_llm(prompt=prompt, with_context=is_web_search, context=context)
            if response.startswith("Error"):
                return render_template('index.html', error=response, prompt=prompt)

            store_query_response(prompt, response)
            return render_template('index.html', response=response, prompt=prompt)
        except Exception as e:
            return render_template('index.html', error=f"Unexpected error: {str(e)}", prompt=prompt)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5050)