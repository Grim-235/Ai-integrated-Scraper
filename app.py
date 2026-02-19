import os
import time
import pickle
import json
import random
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = Flask(__name__)
client = genai.Client()

with open("model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

# =========================
# ADVANCED SCRAPER LOGIC
# =========================
def get_advanced_headers():
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0"
    ]
    return {
        "User-Agent": random.choice(user_agents),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    }

def universal_scrape(url):
    try:
        response = requests.get(url, headers=get_advanced_headers(), timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')

        for script in soup(["script", "style", "noscript", "meta"]):
            script.extract()

        title = soup.title.string.strip() if soup.title else "No Title Found"
        raw_text = soup.get_text(separator=" | ", strip=True)
        
        images_data = []
        for img in soup.find_all('img'):
            src = img.get('src')
            alt = img.get('alt', 'No Description')
            if src and src.startswith('http'):
                images_data.append(f"[IMG: alt='{alt}', src='{src}']")

        combined_data = raw_text[:3500] + "\n\nIMAGES FOUND:\n" + "\n".join(images_data[:20])

        if not raw_text:
            combined_data = "No readable content found."

        X_input = vectorizer.transform([raw_text[:1000]])
        prediction = model.predict(X_input)[0]

        return title, combined_data, prediction
    except Exception as e:
        return "Extraction Error", f"Scraping failed: {str(e)}", "Unknown"

# =========================
# AI JSON STRUCTURER
# =========================
def ai_extract_to_json(url, title, raw_data):
    prompt = f"""
    You are an advanced data extraction engine.
    Target URL: {url}
    Page Title: {title}
    Raw Web Data & Images: {raw_data}
    
    TASK: Analyze the text and the "IMAGES FOUND" list. Extract the main items (up to 4).
    If price or rating is missing, YOU MUST estimate a realistic numerical value based on the item name.
    Correlate the most relevant image URL to the item based on the alt text. If none matches, leave it blank.
    
    STRICT RULES:
    You MUST output ONLY a valid JSON array. Do not include markdown formatting like ```json. 
    Format:
    [
      {{
        "title": "Item Name",
        "price": "$0.00",
        "rating": "0.0/5",
        "summary": "2-sentence summary.",
        "image_url": "https://..."
      }}
    ]
    """
    try:
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        cleaned_json = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned_json)
        return data
    except Exception as e:
        print(f"JSON Parsing Error: {e}")
        return [{"title": "AI Parsing Failed", "price": "N/A", "rating": "N/A", "summary": "The AI failed to format the data into JSON.", "image_url": ""}]

# =========================
# APPLICATION ROUTES
# =========================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/scrape', methods=['POST'])
def scrape():
    start_time = time.time()
    url = request.form['url']
    
    title, raw_data, category = universal_scrape(url)
    json_content = ai_extract_to_json(url, title, raw_data)
    
    enrichment = "Notice: Extracted using rotating headers. Missing metrics estimated. UI generated dynamically via JSON payload."
    elapsed = round(time.time() - start_time, 2)
    
    category_color = {"Technology": "text-info", "Health": "text-success", "Sports": "text-warning", "Politics": "text-danger", "Entertainment": "text-primary"}.get(category, "text-secondary")

    return render_template('result.html', title=title, content_list=json_content, raw_content=raw_data[:2000], category=category, category_color=category_color, enrichment=enrichment, elapsed=elapsed, original_url=url)

@app.route('/compare', methods=['POST'])
def compare():
    start_time = time.time()

    # STATE MANAGEMENT: Fetch Source 1 data and decode the JSON string back into a Python list
    url1 = request.form.get('url1')
    title1 = request.form.get('title1')
    category1 = request.form.get('category1')
    raw_content1 = request.form.get('raw_content1', '')
    
    try:
        content_list1 = json.loads(request.form.get('content_list1', '[]'))
    except Exception:
        content_list1 = []

    # Process Source 2 natively
    url2 = request.form['url2']
    title2, raw_content2, category2 = universal_scrape(url2)
    content_list2 = ai_extract_to_json(url2, title2, raw_content2)
    
    disclaimer = "UI generated dynamically via cross-reference JSON payload."

    if raw_content1 and raw_content2:
        vectors = vectorizer.transform([raw_content1, raw_content2])
        similarity = round(cosine_similarity(vectors[0], vectors[1])[0][0] * 100, 2)
    else:
        similarity = 0.0

    return render_template(
        'compare.html', 
        title1=title1, content_list1=content_list1, category1=category1, 
        title2=title2, content_list2=content_list2, category2=category2, 
        similarity=similarity, enrichment1=disclaimer, enrichment2=disclaimer, 
        raw_content1=raw_content1, raw_content2=raw_content2,
        elapsed=round(time.time() - start_time, 2)
    )

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message')
    scraped_context = data.get('context', '')[:4000]
    
    if not user_message:
        return jsonify({"reply": "Please enter a message."})

    prompt = f"Website Data:\n{scraped_context}\n\nUser Question:\n{user_message}"
    
    try:
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return jsonify({"reply": response.text})
    except Exception as e:
        print(f"Chat Route Error: {e}")
        return jsonify({"reply": "AI connection error. Check backend logs."})

if __name__ == '__main__':
    app.run(debug=True)