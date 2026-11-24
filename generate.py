from openai import OpenAI

groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError("Missing GROQ_API_KEY environment variable.")

client = OpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1",
)

prompt = f"""
You are a legal news anchor. Turn this news into a punchy 45-second YouTube Short script (approx 100-115 words).
Keep it objective, accurate, and engaging. Do not give legal advice.
Return ONLY valid JSON in this exact format:
{{
  "banner_text": "SUPREME COURT UPDATE",
  "script": "Full spoken narration text...",
  "scenes": [
    {{"keyword": "courtroom gavel", "ai_prompt": "editorial illustration of supreme court pillars dramatic lighting"}},
    {{"keyword": "legal contract", "ai_prompt": "scales of justice on dark wooden desk moody lighting"}},
    {{"keyword": "law books", "ai_prompt": "modern high court building exterior cinematic"}}
  ]
}}
News: {news_context}
"""

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"},
)

data = json.loads(response.choices[0].message.content)
