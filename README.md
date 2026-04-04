# Instagram Face Matcher

Upload photos with faces, point at an Instagram account, find matching profiles in their followers/following network.

## Setup

```bash
# Install dependencies (requires cmake for dlib)
brew install cmake
pip install -r requirements.txt

# Copy and fill in your Instagram credentials
cp .env.example .env
```

## Pre-scrape an account (do this before the demo)

```python
from scraper import scrape_account
stats = scrape_account("target_username", max_followers=500)
print(stats)
```

Scraping uses 15-30s delays between requests. Expect 1-2 hours for 1,000 followers.

## Run the app

```bash
streamlit run app.py
```

## Run tests

```bash
pytest tests/ -v
```

## How it works

1. **Scraper** downloads profile photos from an Instagram account's network
2. **Encoder** uses DeepFace (ArcFace model) to create 512-dim face embeddings
3. **Matcher** compares your uploaded photo's faces against stored embeddings using cosine similarity
4. **UI** shows detected faces with bounding boxes and matching profiles with confidence scores
