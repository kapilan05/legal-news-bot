import os, json, random, asyncio, requests, feedparser
import edge_tts
from google import genai
from faster_whisper import WhisperModel
from moviepy.editor import (
    VideoFileClip, ImageClip, AudioFileClip, TextClip, 
    CompositeVideoClip, concatenate_videoclips
)

# 1. FETCH LATEST LEGAL NEWS
RSS_URL = "https://livelaw.in" # Swap with SCOTUSblog/Reuters if preferred
feed = feedparser.parse(RSS_URL)
article = feed.entries[0]
news_context = f"Title: {article.title}\nSummary: {article.summary}"

# 2. GENERATE SCRIPT & SCENE PROMPTS VIA GEMINI FREE TIER
client = genai.Client(api_key=os.environ["AIzaSyBSX3mjT0TZUJNSk6s68RKveVCiykitbHQ"])
prompt = f"""
You are a legal news anchor. Turn this news into a punchy 45-second YouTube Short script (approx 100-115 words).
Keep it objective, accurate, and engaging. Do not give legal advice.
Return ONLY valid JSON in this exact format:
{{
  "title": "Catchy 5-word headline",
  "script": "Full spoken narration text...",
  "scenes": [
    {{"keyword": "courtroom gavel", "ai_prompt": "editorial illustration of supreme court pillars dramatic lighting"}},
    {{"keyword": "legal contract signing", "ai_prompt": "scales of justice on dark wooden desk moody lighting"}},
    {{"keyword": "law books library", "ai_prompt": "modern high court building exterior cinematic"}}
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

# 3. GENERATE VOICEOVER (EDGE-TTS - FREE & UNLIMITED)
async def make_audio(text, filename="voice.mp3"):
    # en-IN-PrabhatNeural (Indian English) or en-US-AndrewNeural (US English)
    communicate = edge_tts.Communicate(text, "en-IN-PrabhatNeural", rate="+5%")
    await communicate.save(filename)

asyncio.run(make_audio(data["script"]))
audio = AudioFileClip("voice.mp3")
total_duration = audio.duration

# 4. FETCH VISUALS (HYBRID: PEXELS B-ROLL + POLLINATIONS AI ART)
pexels_headers = {"Authorization": os.environ["58812422-c0efb3d3a7f26cf017bdaa407"]}
scene_duration = total_duration / len(data["scenes"])
clips = []

for i, scene in enumerate(data["scenes"]):
    # Alternate between Pexels stock video and Pollinations AI editorial art
    if i % 2 == 0:
        r = requests.get(
            f"https://pixabay.com{scene['keyword']}&orientation=portrait&per_page=3",
            headers=pexels_headers
        ).json()
        video_url = r["videos"][0]["video_files"][0]["link"]
        with open(f"clip_{i}.mp4", "wb") as f:
            f.write(requests.get(video_url).content)
        clip = VideoFileClip(f"clip_{i}.mp4").subclip(0, scene_duration).resize((1080, 1920))
    else:
        img_url = f"https://pollinations.ai{requests.utils.quote(scene['ai_prompt'])}?width=1080&height=1920&nologo=true"
        with open(f"img_{i}.jpg", "wb") as f:
            f.write(requests.get(img_url).content)
        # Apply subtle Ken Burns zoom
        clip = (ImageClip(f"img_{i}.jpg")
                .set_duration(scene_duration)
                .resize(lambda t: 1 + 0.04 * t)
                .set_position("center"))
    clips.append(clip)

base_video = concatenate_videoclips(clips, method="compose").set_audio(audio)

# 5. GENERATE WORD-LEVEL CAPTIONS VIA FASTER-WHISPER
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

# 6. RENDER FINAL VIDEO
final = CompositeVideoClip([base_video] + subtitle_clips)
final.write_videofile("final_legal_short.mp4", fps=24, codec="libx264", audio_codec="aac")
