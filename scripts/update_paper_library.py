import feedparser
import os
from datetime import datetime

# Configuration
CATEGORIES = ["cs.LG", "cs.AI", "cs.CL", "stat.ML"]
KEYWORDS = [
    "Mixture of Experts", 
    "MoE", 
    "Manifold", 
    "Zhang Invariant", 
    "Sinkhorn", 
    "DeepSeek", 
    "MPI"
]
OUTPUT_FILE = os.path.join("docs", "research", "latest_papers.md")

def fetch_arxiv_rss(category):
    """Fetches entries from arXiv RSS feed for a given category."""
    url = f"https://rss.arxiv.org/rss/{category}"
    print(f"Fetching {url}...")
    feed = feedparser.parse(url)
    return feed.entries

def filter_entries(entries, keywords):
    """Filters entries based on keywords in title or summary."""
    results = []
    seen_links = set()
    
    for e in entries:
        link = e.link
        if link in seen_links:
            continue
            
        announce_type = getattr(e, "arxiv_announce_type", None)
        if announce_type and "replace" in announce_type.lower():
            continue  # Skip replacements
            
        text = (e.title + " " + e.summary).lower()
        matched_keywords = [k for k in keywords if k.lower() in text]
        
        if matched_keywords:
            results.append({
                "title": e.title,
                "authors": [a.name for a in e.get("authors", [])] if "authors" in e else [],
                "summary": e.summary,
                "link": e.link,
                "published": e.published if hasattr(e, "published") else "Unknown",
                "matched_keywords": matched_keywords,
                "category": e.get("arxiv_primary_category", {}).get("term") or (e.tags[0].term if hasattr(e, 'tags') and e.tags else "Unknown")
            })
            seen_links.add(link)
            
    return results

def update_markdown(papers):
    """Updates the markdown file with new papers."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    header = f"# Latest Relevant arXiv Papers\n\n*Last Updated: {timestamp}*\n\n"
    header += "This list is automatically generated using logic from `arXiv-mcp`.\n\n"
    
    content = ""
    if not papers:
        content = "No new relevant papers found in the latest RSS feeds.\n"
    else:
        for p in papers:
            content += f"## {p['title']}\n"
            content += f"- **Authors**: {', '.join(p['authors'])}\n"
            content += f"- **Date**: {p['published']}\n"
            content += f"- **Category**: {p['category']}\n"
            content += f"- **Link**: [{p['link']}]({p['link']})\n"
            content += f"- **Matched Keywords**: {', '.join(p['matched_keywords'])}\n\n"
            content += f"**Abstract**:\n{p['summary']}\n\n"
            content += "---\n\n"
            
    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(header + content)
    
    print(f"Updated {OUTPUT_FILE} with {len(papers)} papers.")

def main():
    all_entries = []
    for category in CATEGORIES:
        entries = fetch_arxiv_rss(category)
        all_entries.extend(entries)
    
    # Filter unique entries (RSS feeds might overlap or duplicate)
    unique_entries = {e.link: e for e in all_entries}.values()
    
    filtered_papers = filter_entries(unique_entries, KEYWORDS)
    update_markdown(filtered_papers)

if __name__ == "__main__":
    main()
