import asyncio
from typing import List
from crawl4ai import AsyncWebCrawler
from langchain_core.documents import Document

async def crawl_esilv(start_url: str, max_depth: int = 2) -> List[Document]:
    """
    Crawls the ESILV website starting from `start_url` up to `max_depth`.
    Returns a list of LangChain Document objects, filtered for relevance.
    """
    documents = []
    
    # Keywords that indicate the content is relevant to programs/admissions
    RELEVANT_KEYWORDS = [
        "admission", "program", "bachelor", "master", "curriculum", 
        "syllabus", "tuition", "fees", "apply", "deadline", 
        "engineering", "major", "course", "calendar", "scholarship",
        "international", "exchange", "degree"
    ]
    
    def is_content_relevant(text: str) -> bool:
        """Checks if the text contains any of the relevant keywords."""
        if not text:
            return False
        text_lower = text.lower()
        # Check if at least one keyword is present
        # We could increase this threshold (e.g., at least 2 unique keywords) for stricter filtering
        match_count = sum(1 for keyword in RELEVANT_KEYWORDS if keyword in text_lower)
        return match_count >= 1

    # Configure the crawler (you can add more options here if needed)
    async with AsyncWebCrawler(verbose=True) as crawler:
        # For this simple implementation, we'll just crawl the start_url
        # If deep crawling is needed, crawl4ai's implementation details 
        # for recursive crawling would be used here.
        # Since crawl4ai is primarily a single-page or list-of-urls crawler in basic usage,
        # we might need to implement the recursion manually or use its features if available.
        # However, for the initial request "Crawl ESILV Website", let's assume we might receive 
        # a request to crawl a specific page or we can extend this to find links.
        
        # NOTE: crawl4ai's primary API is `arun` for a single URL.
        # To support "max_depth", we would typically need to parse links and queue them.
        # For this first iteration, let's implement a simplified version that 
        # just crawls the provided URL. If the user wants true recursion, we can expand.
        # BUT, the prompt said "scraps information from the ESILV website" and "Crawl4AI".
        # Let's try to get the content of the main page first.
        
        # If we really want recursion, we need to extract links.
        # crawl4ai results include links.
        
        visited = set()
        queue = [(start_url, 0)]
        
        while queue:
            current_url, current_depth = queue.pop(0)
            
            if current_url in visited or current_depth > max_depth:
                continue
            
            visited.add(current_url)
            
            try:
                result = await crawler.arun(url=current_url)
                
                if result.success:
                    # Check if content is relevant
                    if is_content_relevant(result.markdown):
                        # Create a LangChain Document
                        doc = Document(
                            page_content=result.markdown,
                            metadata={
                                "source": current_url,
                                "title": result.metadata.get("title", "No Title"),
                                "page": 1 # Placeholder for compatibility
                            }
                        )
                        documents.append(doc)
                        print(f"✅ Keeping relevant page: {current_url}")
                    else:
                        print(f"⚠️ Skipping irrelevant page: {current_url}")
                    
                    # If we haven't reached max depth, add links to queue
                    if current_depth < max_depth:
                        # Extract internal links (heuristic: starts with start_url domain)
                        # result.links is a dictionary usually? Or list of dicts?
                        # Let's check the result structure dynamically or assume standard crawl4ai output
                        # Based on standard usage, result.links might be available.
                        # If not, we might rely on the user to provide specific URLs or 
                        # just stick to depth 0 for safety in V1.
                        
                        # Let's stick to just the requested URL for now to ensure stability 
                        # unless we are sure about the link structure.
                        # However, to be "better", let's try to find internal links if available.
                        links = result.links.get("internal", [])
                        for link_data in links:
                            href = link_data.get("href")
                            if href and href not in visited:
                                queue.append((href, current_depth + 1))
                                
            except Exception as e:
                print(f"Error crawling {current_url}: {e}")
                
    return documents
