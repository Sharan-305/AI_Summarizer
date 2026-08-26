import io
import re
from typing import Dict, Any, Union
import pypdf
import requests
from bs4 import BeautifulSoup
from src.preprocessing import clean_text


def extract_text_from_pdf(pdf_source: Union[str, bytes, io.BytesIO]) -> Dict[str, Any]:
    """
    Extracts text content page-by-page from a PDF file.
    Accepts file path string, raw bytes, or BytesIO stream.
    
    Returns a dictionary containing:
      - 'text': Extracted and cleaned full text string.
      - 'pages': List of extracted text per page.
      - 'num_pages': Total page count.
      - 'word_count': Total word count.
    """
    if isinstance(pdf_source, str):
        reader = pypdf.PdfReader(pdf_source)
    elif isinstance(pdf_source, bytes):
        reader = pypdf.PdfReader(io.BytesIO(pdf_source))
    elif isinstance(pdf_source, io.BytesIO):
        reader = pypdf.PdfReader(pdf_source)
    else:
        raise ValueError("Unsupported PDF input type. Must be file path, bytes, or BytesIO.")
        
    num_pages = len(reader.pages)
    page_texts = []
    
    for i, page in enumerate(reader.pages):
        try:
            raw_page_text = page.extract_text() or ""
            cleaned_page = clean_text(raw_page_text)
            if cleaned_page:
                page_texts.append(cleaned_page)
        except Exception as e:
            print(f"Warning: Failed to extract text from page {i+1}: {e}")
            
    full_text = "\n\n".join(page_texts)
    word_count = len(full_text.split())
    
    return {
        "text": full_text,
        "pages": page_texts,
        "num_pages": num_pages,
        "word_count": word_count
    }


def extract_text_from_url(url: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Scrapes main body article text from a public web page URL using BeautifulSoup.
    
    Returns a dictionary containing:
      - 'text': Extracted article text.
      - 'title': Page title.
      - 'word_count': Word count.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch content from URL: {e}")
        
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract page title
    title = soup.title.string.strip() if soup.title and soup.title.string else "Web Article"
    
    # Remove script, style, nav, footer, header tags
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
        
    # Gather text paragraphs
    paragraphs = [p.get_text() for p in soup.find_all('p')]
    raw_article = " ".join(paragraphs)
    
    if not raw_article.strip():
        # Fallback to body text if no <p> tags found
        raw_article = soup.get_text()
        
    cleaned_article = clean_text(raw_article)
    word_count = len(cleaned_article.split())
    
    return {
        "text": cleaned_article,
        "title": title,
        "word_count": word_count
    }
