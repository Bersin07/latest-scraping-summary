import asyncio
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
import streamlit as st
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
import sys
import ollama
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy()) 

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

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present
"""

def call_llm(prompt: str, with_context: bool = True, context: str | None = None):
    messages = [
        {
            "role": "system",
            "content":system_prompt,
        },
        {
            "role": "user",
            "content": f"Context: {context}, Question: {prompt}",
        },
    ]

    if not with_context:   
        messages.pop(0)
        messages[0]["content"] = prompt
    
    response = ollama.chat(model="llama3.2:3b", stream=True, messages=messages)
    
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"] ["content"]
        else:
            break



def get_vector_collection() -> tuple[chromadb.Collection, chromadb.Client]:
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="mxbai-embed-large",
    )
    chroma_client = chromadb.PersistentClient(
        path="./web-search-llm-db", settings=Settings(anonymized_telemetry=False)
    )

    collection_name = "web_llm"
    
    # **🔥 FIX: List collections correctly**
    existing_collection_names = [col for col in chroma_client.list_collections()]  

    if collection_name in existing_collection_names:
        print(f"✅ Using existing collection: {collection_name}")
        collection = chroma_client.get_collection(name=collection_name)
    else:
        print(f"🛠 Creating new ChromaDB collection: {collection_name}")
        collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=ollama_ef,
            metadata={"hnsw:space": "cosine"},
        )

    return collection, chroma_client





def normalize_url(url):
    normalized_url = (
    url.replace("https://", "")
    .replace("www.", "")
    .replace("/", "_")
    .replace("-", "_")
    .replace(".", "_")
    )
    print("Normalized URL", normalized_url)
    return normalized_url



def add_to_vector_database(results: list[CrawlResult]):
    collection, _ = get_vector_collection()
    
    for result in results:
        documents, metadatas, ids = [], [], []
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "?", "!", "" ""]
        )

        if result.markdown_v2:
            markdown_result = result.markdown_v2.fit_markdown
        else:
            continue

        temp_file = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8")
        temp_file.write(markdown_result)
        temp_file.flush()

        loader = UnstructuredMarkdownLoader(temp_file.name, mode="single", encoding="utf-8")
        docs = loader.load()
        all_splits = text_splitter.split_documents(docs)
        normalized_url = normalize_url(result.url)

        if all_splits:
            for idx, split in enumerate(all_splits):
                documents.append(split.page_content)
                metadatas.append({"source": result.url})
                ids.append(f"{normalized_url}_{idx}")

            print("Upserting documents into ChromaDB...")
            print(f"Documents count: {len(documents)}")
            print(f"Metadata count: {len(metadatas)}")
            print(f"IDs count: {len(ids)}")

            # **🛠 Fix: Ensure documents are not empty before upserting**
            if not documents:
                print("⚠ Warning: No documents found after splitting. Skipping upsert.")
                continue  # Skip empty embeddings

            collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )




async def crawl_webpages(urls: list[str], prompt: str) -> CrawlResult:
    bm25_filter = BM25ContentFilter(user_query=prompt, bm25_threshold=1.2)
    md_generator = DefaultMarkdownGenerator (content_filter=bm25_filter)
    crawler_config = CrawlerRunConfig(
        markdown_generator=md_generator,
        excluded_tags=["nav", "footer", "header", "form", "img", "a"],
        only_text=True,
        exclude_social_media_links=True,
        keep_data_attributes=False,
        cache_mode=CacheMode. BYPASS,
        remove_overlay_elements=True,
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
        page_timeout=20000, # in ms: 20 seconds
    )
    browser_config = BrowserConfig(headless=True, text_mode=True, light_mode=True)

    async with AsyncWebCrawler (config=browser_config) as crawler:
        results = await crawler.arun_many (urls, config=crawler_config)
        return results



def check_robots_txt(urls: list[str]) -> list[str]:
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



def get_web_urls (search_term: str, num_results: int = 10) -> list[str]:
    try:
        discard_urls = ["youtube.com", "britannica.com", "vimeo.com"]
        for url in discard_urls:
            search_term += f" -site: {url}"

        results = DDGS().text(search_term, max_results=num_results)
        results = [result["href"] for result in results]

        return check_robots_txt(results)

    except Exception as e:
        error_msg = ("X Failed to fetch results from the web", str(e))
        print(error_msg)
        st.write(error_msg)
        st.stop()



async def run():
    st.set_page_config(page_title="LLM Web search")
    st.header("LLM Web search")
    prompt=st.text_area("Enter your prompt here",
        placeholder="Add your Query",
        label_visibility="hidden",
    )
    is_web_search=st.toggle("Enable Web search", value=False, key="Web Search")
    go = st.button("Go")

    collection, chroma_client = get_vector_collection()

    if prompt and go:
        if is_web_search:
            web_urls = get_web_urls(search_term=prompt)
            if not web_urls:
                st.write("No results found.")
                st.stop()

            results = await crawl_webpages(urls=web_urls, prompt=prompt)
            add_to_vector_database(results)

            # **🛠 Fix: Ensure that there is data before querying**
            qresults = collection.query(query_texts=[prompt], n_results=10)
            if not qresults or not qresults.get("documents") or len(qresults.get("documents")) == 0:
                st.write("⚠ No relevant documents found in the database.")
                st.stop()

            context = qresults.get("documents")[0]

            llm_response = call_llm(
                context=context, prompt=prompt, with_context=is_web_search
            )
            st.write_stream(llm_response)
        else:
            llm_response = call_llm(prompt=prompt, with_context=is_web_search)
            st.write_stream(llm_response)



if __name__ == "__main__":
    asyncio.run(run())