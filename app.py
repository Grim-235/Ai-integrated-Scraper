import os
import time
import pickle
import json
import random
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, jsonify, send_from_directory
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GEMINI_MODEL = "gemini-2.5-flash"

with open("model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

def load_gemini_clients():
    clients = []
    key_names = ["GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"]

    for key_name in key_names:
        api_key = os.getenv(key_name)
        if api_key:
            clients.append((key_name, genai.Client(api_key=api_key)))

    if not clients:
        clients.append(("DEFAULT_ENV", genai.Client()))

    return clients

GEMINI_CLIENTS = load_gemini_clients()

def generate_content_with_fallback(contents):
    last_error = None
    quota_errors = []

    for key_name, gemini_client in GEMINI_CLIENTS:
        try:
            response = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=contents)
            return response, key_name, None
        except Exception as e:
            error_text = str(e)
            last_error = e
            print(f"Gemini request failed on {key_name}: {error_text}")

            if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text or "quota" in error_text.lower():
                quota_errors.append(key_name)
                continue

            return None, key_name, e

    return None, None, last_error or Exception("No Gemini clients available.")

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
    
    TASK: Analyze the text and the "IMAGES FOUND" list. Extract the main items (up to 15).
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
        response, used_key, error = generate_content_with_fallback(prompt)
        if error:
            raise error
        cleaned_json = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned_json)
        return data, f"Structured extraction completed using {used_key}."
    except Exception as e:
        error_text = str(e)
        print(f"JSON Parsing Error: {error_text}")

        if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text or "quota" in error_text.lower():
            return (
                [{
                    "title": "AI quota reached",
                    "price": "N/A",
                    "rating": "N/A",
                    "summary": "The page was scraped, but all configured Gemini API keys are currently quota-limited. Try again after the retry window or switch to a billed plan/model.",
                    "image_url": ""
                }],
                "All configured Gemini API keys have reached quota for the current model, so structured AI extraction is temporarily unavailable."
            )

        return (
            [{
                "title": "AI Parsing Failed",
                "price": "N/A",
                "rating": "N/A",
                "summary": "The page was scraped, but the AI response could not be converted into structured JSON.",
                "image_url": ""
            }],
            "The scrape completed, but the AI response could not be parsed into the expected JSON structure."
        )

# =========================
# APPLICATION ROUTES
# =========================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/college-logo')
def college_logo():
    return send_from_directory(BASE_DIR, 'WhatsApp_Image_2026-02-17_at_13.29.51-removebg-preview.png')

@app.route('/project-report')
def project_report():
    return send_from_directory(BASE_DIR, 'AI_Web_Scraper_Report_Final.docx.pdf')

@app.route('/scrape', methods=['POST'])
def scrape():
    start_time = time.time()
    url = request.form['url']
    
    title, raw_data, category = universal_scrape(url)
    json_content, ai_notice = ai_extract_to_json(url, title, raw_data)
    
    enrichment = "Notice: Extracted using rotating headers. Missing metrics estimated. UI generated dynamically via JSON payload."
    if ai_notice:
        enrichment = f"{enrichment} {ai_notice}"
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
    content_list2, ai_notice2 = ai_extract_to_json(url2, title2, raw_content2)
    
    disclaimer = "UI generated dynamically via cross-reference JSON payload."
    if ai_notice2:
        disclaimer = f"{disclaimer} {ai_notice2}"

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
        response, used_key, error = generate_content_with_fallback(prompt)
        if error:
            raise error
        return jsonify({"reply": response.text})
    except Exception as e:
        print(f"Chat Route Error: {e}")
        return jsonify({"reply": "AI connection error across configured Gemini keys. Check backend logs."})

if __name__ == '__main__':
    app.run(debug=True)
