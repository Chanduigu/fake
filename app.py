from flask import Flask, render_template, request
import joblib
import requests
import os
from bs4 import BeautifulSoup
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Load ML model and vectorizer
model = joblib.load(os.path.join('model', 'news_classifier.pkl'))
vectorizer = joblib.load(os.path.join('model', 'tfidf_vectorizer.pkl'))

# Load API keys securely
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

def fetch_news_and_predict():
    try:
        url = f"https://newsapi.org/v2/top-headlines?country=in&pageSize=10&apiKey={NEWS_API_KEY}"
        response = requests.get(url, timeout=5)
        articles = response.json().get("articles", [])
    except Exception:
        return []

    predictions = []
    for article in articles:
        title = article.get("title", "")
        desc = article.get("description", "")
        content = f"{title}. {desc}"

        if content.strip():
            try:
                vec = vectorizer.transform([content])
                pred = model.predict(vec)[0]
                label = "REAL" if pred == 1 else "FAKE"

                gpt_check = gpt_fact_check(content)

                predictions.append({
                    'title': title,
                    'description': desc,
                    'prediction': label,
                    'gpt_verification': gpt_check
                })
            except Exception:
                continue

    return predictions

def gpt_fact_check(text):
    try:
        prompt = (
            f"Fact check the following claim and respond ONLY with 'REAL' or 'FAKE':\n\n{text}\n"
        )
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5
        )
        reply = response['choices'][0]['message']['content'].strip().upper()
        return "REAL" if "REAL" in reply else "FAKE"
    except Exception as e:
        return f"ERROR: {e}"

@app.route('/', methods=['GET'])
def home():
    results = fetch_news_and_predict()
    return render_template('autodetect.html', results=results)

@app.route('/check_url', methods=['POST'])
def check_url():
    url = request.form['url']
    results = fetch_news_and_predict()

    try:
        page = requests.get(url, timeout=10)
        page.encoding = 'utf-8'
        soup = BeautifulSoup(page.text, 'html.parser')

        title = soup.title.string.strip() if soup.title else "No title found"
        paragraphs = soup.find_all('p')
        content = ' '.join(p.get_text().strip() for p in paragraphs[:10])

        if not content.strip():
            raise Exception("No readable content found.")

        vec = vectorizer.transform([content])
        pred = model.predict(vec)[0]
        label = "REAL" if pred == 1 else "FAKE"

        gpt_check = gpt_fact_check(content)

        return render_template('autodetect.html',
                               results=results,
                               url_result=label,
                               url_title=title,
                               url_content=content,
                               gpt_url_result=gpt_check)

    except Exception as e:
        return render_template('autodetect.html',
                               results=results,
                               url_result="ERROR",
                               url_error=str(e))

@app.route('/check_sentence', methods=['POST'])
def check_sentence():
    sentence = request.form['sentence']
    results = fetch_news_and_predict()

    try:
        vec = vectorizer.transform([sentence])
        pred = model.predict(vec)[0]
        label = "REAL" if pred == 1 else "FAKE"

        gpt_check = gpt_fact_check(sentence)

        return render_template('autodetect.html',
                               results=results,
                               sentence_result=label,
                               sentence_text=sentence,
                               gpt_sentence_result=gpt_check)
    except Exception as e:
        return render_template('autodetect.html',
                               results=results,
                               sentence_result="ERROR",
                               sentence_text=sentence,
                               url_error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
