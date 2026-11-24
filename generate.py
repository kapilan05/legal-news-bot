import os
import json
import asyncio
import requests
import feedparser
import edge_tts
from google import genai
from faster_whisper import WhisperModel
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (
    VideoFileClip,
    ImageClip,
    AudioFileClip,
    TextClip,
    CompositeVideoClip,
    concatenate_videoclips,
)

# --- 1. FETCH UNPOSTED LEGAL NEWS ---
RSS_URLS = [
    "https://www.livelaw.in/google_feeds.xml",
    "https://www.livelaw.in/feed",
]

feed = None
for url in RSS_URLS:
    parsed = feedparser.parse(url)
    if getattr(parsed, "entries", None):
        feed = parsed
        break

if not feed or not feed.entries:
    print("No RSS entries found.")
    raise SystemExit(0)

posted_links = set()
if os.path.exists("posted.txt"):
    with open("posted.txt", "r", encoding="utf-8") as f:
        posted_links = {line.strip() for line in f if line.strip()}

article = None
for entry in feed.entries:
    link = getattr(entry, "link", "").strip()
    if link and link not in posted_links:
        article = entry
        break

if not article:
    print("No new unposted articles found.")
    raise SystemExit(0)

title = getattr(article, "title", "Legal News Update")
summary = getattr(article, "summary", "")
news_context = f"Title: {title}\nSummary: {summary}"

# Extract real featured image from RSS feed if available
rss_image_url = None
if getattr(article, "media_content", None):
    rss_image_url = article.media_content[0].get("url")
elif getattr(article, "enclosures", None):
    rss_image_url = article.enclosures[0].get("href")
elif getattr(article, "media_thumbnail", None):
    rss_image_url = article.media_thumbnail[0].get("url")

# --- 2. GENERATE SCRIPT & VISUAL PROMPTS VIA GEMINI ---
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not gemini_api_key:
    raise RuntimeError("Missing GEMINI_API_KEY environment variable.")

client = genai.Client(api_key=gemini_api_key)
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
    model="gemini-3.8-flash",
    contents=prompt,
    config={"response_mime_type": "application/json"},
)
data = json.loads(response.text)

scenes = data.get("scenes") or [
    {
        "keyword": "courtroom gavel",
        "ai_prompt": "editorial illustration of supreme court pillars dramatic lighting",
    }
]

# --- 3. GENERATE VOICEOVER (EDGE-TTS) ---
async def make_audio(text: str, filename: str = "voice.mp3") -> None:
    communicate = edge_tts.Communicate(text, "en-IN-PrabhatNeural", rate="+5%")
    await communicate.save(filename)


asyncio.run(make_audio(data["script"], "voice.mp3"))
audio = AudioFileClip("voice.mp3")
total_duration = max(float(audio.duration), 5.0)
scene_duration = total_duration / len(scenes)


def download_file(url: str, dest_path: str, timeout: int = 25) -> None:
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(resp.content)


def make_ken_burns_clip(img_url: str, filename: str, duration: float):
    download_file(img_url, filename)
    img = Image.open(filename).convert("RGB")

    target_w, target_h = 1080, 1920
    target_ratio = target_w / target_h
    img_ratio = img.width / img.height

    if img_ratio > target_ratio:
        new_width = int(img.height * target_ratio)
        offset = (img.width - new_width) // 2
        img = img.crop((offset, 0, offset + new_width, img.height))
    else:
        new_height = int(img.width / target_ratio)
        offset = (img.height - new_height) // 2
        img = img.crop((0, offset, img.width, offset + new_height))

    img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    img.save(filename)

    zoomed = (
        ImageClip(filename)
        .set_duration(duration)
        .resize(lambda t: 1.0 + 0.03 * min(t, duration))
        .set_position(("center", "center"))
    )
    return CompositeVideoClip([zoomed], size=(target_w, target_h)).set_duration(duration)


# --- 4. BUILD HYBRID VISUALS ---
clips = []
pixabay_key = os.environ.get("PIXABAY_API_KEY", "").strip()

for i, scene in enumerate(scenes):
    encoded_prompt = requests.utils.quote(scene["ai_prompt"])
    ai_fallback_url = (
        f"https://image.pollinations.ai/prompt/{encoded_prompt}"
        f"?width=1080&height=1920&nologo=true"
    )

    if i == 0 and rss_image_url:
        try:
            clips.append(make_ken_burns_clip(rss_image_url, f"img_{i}.jpg", scene_duration))
            continue
        except Exception as exc:
            print(f"RSS image failed, falling back to AI image: {exc}")

    if i == 2 and pixabay_key:
        try:
            q = requests.utils.quote(scene["keyword"])
            api_url = f"https://pixabay.com/api/videos/?key={pixabay_key}&q={q}&per_page=3"
            r = requests.get(api_url, timeout=15)
            r.raise_for_status()
            hits = r.json().get("hits", [])
            if hits:
                video_url = hits[0]["videos"]["medium"]["url"]
                raw_clip_path = f"clip_{i}.mp4"
                download_file(video_url, raw_clip_path, timeout=30)

                v_clip = VideoFileClip(raw_clip_path).without_audio()
                if v_clip.duration < scene_duration:
                    v_clip = v_clip.loop(duration=scene_duration)
                else:
                    v_clip = v_clip.subclip(0, scene_duration)

                v_clip = v_clip.resize(height=1920)
                if v_clip.w < 1080:
                    v_clip = v_clip.resize(width=1080)
                v_clip = v_clip.crop(x_center=v_clip.w / 2, width=1080)
                v_clip = v_clip.set_duration(scene_duration)
                clips.append(v_clip)
                continue
        except Exception as exc:
            print(f"Pixabay clip failed, falling back to AI image: {exc}")

    clips.append(make_ken_burns_clip(ai_fallback_url, f"img_{i}.jpg", scene_duration))

base_video = concatenate_videoclips(clips, method="compose").set_audio(audio)

# --- 5. CREATE TOP NEWS BANNER OVERLAY ---
banner_img = Image.new("RGBA", (1080, 1920), (0, 0, 0, 0))
draw = ImageDraw.Draw(banner_img)
draw.rectangle([(120, 180), (960, 275)], fill=(185, 28, 28, 235))

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 38)
except Exception:
    font = ImageFont.load_default()

banner_label = str(data.get("banner_text", "LEGAL NEWS UPDATE")).upper()[:28]
draw.text((540, 227), banner_label, fill="white", font=font, anchor="mm")
banner_img.save("banner.png")
banner_clip = ImageClip("banner.png").set_duration(total_duration)

# --- 6. GENERATE WORD-LEVEL CAPTIONS VIA FASTER-WHISPER ---
subtitle_clips = []
try:
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    segments, _ = model.transcribe("voice.mp3", word_timestamps=True)

    for seg in segments:
        if not seg.words:
            continue
        for word in seg.words:
            clean_word = word.word.strip().upper()
            if not clean_word or word.end <= word.start:
                continue
            txt = (
                TextClip(
                    clean_word,
                    fontsize=72,
                    color="yellow",
                    font="DejaVu-Sans-Bold",
                    stroke_color="black",
                    stroke_width=3,
                )
                .set_position(("center", 1250))
                .set_start(word.start)
                .set_end(word.end)
            )
            subtitle_clips.append(txt)
except Exception as exc:
    print(f"Caption generation skipped due to error: {exc}")

# --- 7. RENDER FINAL VIDEO & LOG ARTICLE ---
output_file = "final_legal_short.mp4"
final = CompositeVideoClip([base_video, banner_clip] + subtitle_clips, size=(1080, 1920))
final.write_videofile(
    output_file,
    fps=24,
    codec="libx264",
    audio_codec="aac",
    threads=2,
    logger=None,
)

if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
    with open("posted.txt", "a", encoding="utf-8") as f:
        f.write(article.link.strip() + "\n")
    print(f"Successfully generated {output_file}")
else:
    raise RuntimeError("Final video file was not created or is empty.")
