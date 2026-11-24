import os, json, asyncio, requests, feedparser
import edge_tts
from google import genai
from faster_whisper import WhisperModel
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (
    VideoFileClip, ImageClip, AudioFileClip, TextClip, 
    CompositeVideoClip, concatenate_videoclips
)

# --- 1. FETCH UNPOSTED LEGAL NEWS ---
RSS_URL = "https://livelaw.in" # Or "https://barandbench.com"
feed = feedparser.parse(RSS_URL)

posted_links = set()
if os.path.exists("posted.txt"):
    with open("posted.txt", "r") as f:
        posted_links = set(line.strip() for line in f)

article = None
for entry in feed.entries:
    if entry.link not in posted_links:
        article = entry
        break

if not article:
    print("No new articles found.")
    exit(0)

news_context = f"Title: {article.title}\nSummary: {article.summary}"

# Extract real featured image from RSS feed if available
rss_image_url = None
if "media_content" in article and article.media_content:
    rss_image_url = article.media_content[0].get("url")
elif "enclosures" in article and article.enclosures:
    rss_image_url = article.enclosures[0].get("href")

# --- 2. GENERATE SCRIPT & VISUAL PROMPTS VIA GEMINI ---
client = genai.Client(api_key=os.environ["AIzaSyBSX3mjT0TZUJNSk6s68RKveVCiykitbHQ"])
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
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config={"response_mime_type": "application/json"}
)
data = json.loads(response.text)

# --- 3. GENERATE VOICEOVER (EDGE-TTS - FREE & UNLIMITED) ---
async def make_audio(text, filename="voice.mp3"):
    # en-IN-PrabhatNeural (Indian English) or en-US-AndrewNeural (US English)
    communicate = edge_tts.Communicate(text, "en-IN-PrabhatNeural", rate="+5%")
    await communicate.save(filename)

asyncio.run(make_audio(data["script"]))
audio = AudioFileClip("voice.mp3")
total_duration = audio.duration
scene_duration = total_duration / len(data["scenes"])

# Helper: Download & fit any image to 1080x1920 with Ken Burns zoom
def make_ken_burns_clip(img_url, filename, duration):
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(img_url, headers=headers, timeout=20)
    with open(filename, "wb") as f:
        f.write(resp.content)
    # Resize and center-crop via Pillow so aspect ratios never distort
    img = Image.open(filename).convert("RGB")
    target_ratio = 1080 / 1920
    img_ratio = img.width / img.height
    if img_ratio > target_ratio:
        new_width = int(img.height * target_ratio)
        offset = (img.width - new_width) // 2
        img = img.crop((offset, 0, offset + new_width, img.height))
    else:
        new_height = int(img.width / target_ratio)
        offset = (img.height - new_height) // 2
        img = img.crop((0, offset, img.width, offset + new_height))
    img = img.resize((1080, 1920), Image.Resampling.LANCZOS)
    img.save(filename)

    return (ImageClip(filename)
            .set_duration(duration)
            .resize(lambda t: 1 + 0.04 * t)
            .set_position("center"))

# --- 4. BUILD HYBRID VISUALS (RSS + POLLINATIONS + PIXABAY) ---
clips = []
pixabay_key = os.environ.get("58812422-c0efb3d3a7f26cf017bdaa407", "")

for i, scene in enumerate(data["scenes"]):
    ai_fallback_url = f"https://pollinations.ai{requests.utils.quote(scene['ai_prompt'])}?width=1080&height=1920&nologo=true"
    
    # Scene 0: Use real RSS news photo if available, else Pollinations AI
    if i == 0 and rss_image_url:
        try:
            clips.append(make_ken_burns_clip(rss_image_url, f"img_{i}.jpg", scene_duration))
            continue
        except Exception:
            pass

    # Scene 2: Use Pixabay HD stock video if key is provided, else Pollinations AI
    if i == 2 and pixabay_key:
        try:
            r = requests.get(
                f"https://pixabay.com{pixabay_key}&q={requests.utils.quote(scene['keyword'])}&per_page=3",
                timeout=15
            ).json()
            video_url = r["hits"][0]["videos"]["medium"]["url"]
            with open(f"clip_{i}.mp4", "wb") as f:
                f.write(requests.get(video_url).content)
            v_clip = VideoFileClip(f"clip_{i}.mp4")
            # Loop if stock clip is shorter than scene_duration
            if v_clip.duration < scene_duration:
                v_clip = v_clip.loop(duration=scene_duration)
            else:
                v_clip = v_clip.subclip(0, scene_duration)
            v_clip = v_clip.resize(height=1920)
            v_clip = v_clip.crop(x_center=v_clip.w / 2, width=1080)
            clips.append(v_clip)
            continue
        except Exception:
            pass

    # Default / Fallback: Pollinations.ai vertical artwork
    clips.append(make_ken_burns_clip(ai_fallback_url, f"img_{i}.jpg", scene_duration))

base_video = concatenate_videoclips(clips, method="compose").set_audio(audio)

# --- 5. CREATE TOP NEWS BANNER OVERLAY ---
banner_img = Image.new("RGBA", (1080, 1920), (0, 0, 0, 0))
draw = ImageDraw.Draw(banner_img)
draw.rectangle([(140, 180), (940, 270)], fill=(185, 28, 28, 235)) # Crimson news bar
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
except Exception:
    font = ImageFont.load_default()
banner_label = data.get("banner_text", "LEGAL NEWS UPDATE").upper()[:28]
draw.text((540, 225), banner_label, fill="white", font=font, anchor="mm")
banner_img.save("banner.png")
banner_clip = ImageClip("banner.png").set_duration(total_duration)

# --- 6. GENERATE WORD-LEVEL CAPTIONS VIA FASTER-WHISPER ---
model = WhisperModel("tiny", device="cpu", compute_type="int8")
segments, _ = model.transcribe("voice.mp3", word_timestamps=True)

subtitle_clips = []
for seg in segments:
    for word in seg.words:
        txt = (TextClip(word.word.strip().upper(), fontsize=75, color="yellow",
                        font="DejaVu-Sans-Bold", stroke_color="black", stroke_width=3)
               .set_position(("center", 1250))
               .set_start(word.start)
               .set_end(word.end))
        subtitle_clips.append(txt)

# --- 7. RENDER FINAL VIDEO & LOG ARTICLE ---
final = CompositeVideoClip([base_video, banner_clip] + subtitle_clips)
final.write_videofile("final_legal_short.mp4", fps=24, codec="libx264", audio_codec="aac")

with open("posted.txt", "a") as f:
    f.write(article.link + "\n")
