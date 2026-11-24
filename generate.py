import os
import re
import json
import time
import asyncio
import textwrap
from urllib.parse import urlparse
import requests
import feedparser
import edge_tts
from ddgs import DDGS
from google import genai
from google.genai.errors import APIError, ServerError
from faster_whisper import WhisperModel
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (
    ImageClip,
    AudioFileClip,
    TextClip,
    CompositeVideoClip,
    concatenate_videoclips,
)

if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

W, H = 1080, 1920
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126.0.0.0 Safari/537.36"
}

RSS_URLS = [
    "https://www.livelaw.in/google_feeds.xml",
    "https://www.livelaw.in/feed",
]


def clean_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"&nbsp;|&#160;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


# --- 1. FETCH UNPOSTED LEGAL ARTICLE ---
posted_links = set()
if os.path.exists("posted.txt"):
    with open("posted.txt", "r", encoding="utf-8") as f:
        posted_links = {line.strip() for line in f if line.strip()}

article_title = None
article_summary = None
article_link = None
rss_image_url = None

for url in RSS_URLS:
    parsed = feedparser.parse(url)
    for entry in getattr(parsed, "entries", []):
        link = getattr(entry, "link", "").strip()
        if link and link not in posted_links:
            article_title = clean_html(getattr(entry, "title", ""))
            article_summary = clean_html(getattr(entry, "summary", ""))
            article_link = link
            if getattr(entry, "media_content", None):
                rss_image_url = entry.media_content[0].get("url")
            elif getattr(entry, "enclosures", None):
                rss_image_url = entry.enclosures[0].get("href")
            elif getattr(entry, "media_thumbnail", None):
                rss_image_url = entry.media_thumbnail[0].get("url")
            break
    if article_title:
        break

if not article_title:
    article_title = "Supreme Court Reviews Constitutional Principles on Judicial Timelines"
    article_summary = (
        "The Supreme Court of India reiterated the importance of timely adjudication in matters "
        "affecting fundamental rights, statutory interpretation, and administrative accountability. "
        "Bench observations emphasized clear legal reasoning, procedural fairness, and accessible "
        "justice delivery across courts."
    )


# --- 2. SCRIPT GENERATION: GEMINI FIRST, DETERMINISTIC FALLBACK SECOND ---
def build_fallback_package(title: str, summary: str) -> dict:
    sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", summary) if len(s.strip()) > 15
    ]
    legal_terms = (
        "court", "bench", "justice", "act", "section", "petition",
        "order", "law", "right", "high court", "supreme", "tribunal", "appeal",
    )
    ranked = sorted(
        sentences,
        key=lambda s: sum(1 for kw in legal_terms if kw in s.lower()),
        reverse=True,
    )

    s1_narration = (
        f"Legal news brief. {title}. Let's break down what happened, why it matters, "
        f"and what the court or legal authorities focused on."
    )
    s2_narration = (
        " ".join(ranked[:2])
        if ranked
        else f"The key development centers on {title} and its immediate procedural impact."
    )
    s3_narration = (
        " ".join(ranked[2:4])
        if len(ranked) > 2
        else (
            "From a legal standpoint, the reasoning touches on statutory interpretation, "
            "procedural fairness, and how similar petitions are evaluated going forward."
        )
    )
    s4_narration = (
        "In summary, this update highlights key legal principles shaping ongoing litigation "
        "and regulatory compliance. Follow for more structured Indian legal news briefs."
    )

    return {
        "banner_tag": "COURT & POLICY BRIEF",
        "scenes": [
            {
                "label": "HEADLINE",
                "card_text": title,
                "narration": s1_narration,
                "search_query": f"{' '.join(title.split()[:5])} India law court",
            },
            {
                "label": "KEY FACTS",
                "card_text": s2_narration,
                "narration": s2_narration,
                "search_query": "Supreme Court of India courtroom gavel justice",
            },
            {
                "label": "LEGAL CONTEXT",
                "card_text": s3_narration,
                "narration": s3_narration,
                "search_query": "scales of justice law books legal library",
            },
            {
                "label": "TAKEAWAY",
                "card_text": s4_narration,
                "narration": s4_narration,
                "search_query": "Indian high court parliament constitution building",
            },
        ],
    }


def generate_with_gemini(title: str, summary: str) -> dict | None:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("GEMINI_API_KEY not set; using fallback script generator.")
        return None

    client = genai.Client(api_key=api_key)
    prompt = f"""
You are a clear, objective legal news editor creating a detailed 60-to-75 second vertical video brief
(around 160-195 words total across 4 scenes) from an Indian legal news report.
Do not give legal advice. Stay accurate to the provided news context.

Return ONLY valid JSON matching this schema:
{{
  "banner_tag": "SHORT UPPERCASE HEADER (MAX 28 CHARS)",
  "scenes": [
    {{
      "label": "HEADLINE",
      "card_text": "Concise on-screen summary (max 140 chars)",
      "narration": "Spoken narration for scene 1 (~40-50 words)",
      "search_query": "3 to 6 word web image search query"
    }},
    {{
      "label": "KEY FACTS",
      "card_text": "Concise on-screen summary (max 160 chars)",
      "narration": "Spoken narration for scene 2 (~40-50 words)",
      "search_query": "3 to 6 word web image search query"
    }},
    {{
      "label": "LEGAL ANGLE",
      "card_text": "Concise on-screen summary (max 160 chars)",
      "narration": "Spoken narration for scene 3 (~40-50 words)",
      "search_query": "3 to 6 word web image search query"
    }},
    {{
      "label": "TAKEAWAY",
      "card_text": "Concise on-screen summary (max 140 chars)",
      "narration": "Spoken narration for scene 4 (~30-45 words)",
      "search_query": "3 to 6 word web image search query"
    }}
  ]
}}

Article Title: {title}
Article Summary: {summary}
"""

    models = ["gemini-3.8-flash", "gemini-3.5-flash-lite"]
    for model_name in models:
        for attempt in range(3):
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={"response_mime_type": "application/json"},
                )
                raw = (resp.text or "").strip()
                if not raw:
                    continue
                parsed = json.loads(raw)
                scenes = parsed.get("scenes")
                if isinstance(scenes, list) and len(scenes) >= 2:
                    valid_scenes = []
                    for sc in scenes[:4]:
                        narr = str(sc.get("narration", "")).strip()
                        if narr:
                            valid_scenes.append(
                                {
                                    "label": str(sc.get("label", "LEGAL UPDATE")).upper()[:20],
                                    "card_text": str(sc.get("card_text", narr))[:220],
                                    "narration": narr,
                                    "search_query": str(sc.get("search_query", "courtroom law legal"))[:80],
                                }
                            )
                    if valid_scenes:
                        return {
                            "banner_tag": str(parsed.get("banner_tag", "COURT & POLICY BRIEF")).upper()[:28],
                            "scenes": valid_scenes,
                        }
            except (ServerError, APIError, json.JSONDecodeError) as exc:
                print(f"Gemini ({model_name}) attempt {attempt + 1} issue: {exc}")
                if attempt < 2:
                    time.sleep(3 * (2 ** attempt))
            except Exception as exc:
                print(f"Unexpected Gemini error on {model_name}: {exc}")
                break

    return None


package = generate_with_gemini(article_title, article_summary) or build_fallback_package(
    article_title, article_summary
)
banner_tag = package["banner_tag"]
scenes = package["scenes"]
full_narration = " ".join(sc["narration"] for sc in scenes)


# --- 3. WEB IMAGE SOURCING + CREDITS (NO PEXELS) ---
def verify_and_save_image(content: bytes, out_path: str, min_w: int = 250, min_h: int = 180) -> bool:
    if not content:
        return False
    with open(out_path, "wb") as f:
        f.write(content)
    try:
        with Image.open(out_path) as im:
            im.verify()
        with Image.open(out_path) as im:
            return im.width >= min_w and im.height >= min_h
    except Exception:
        return False


def fetch_web_search_image(query: str, out_path: str) -> str | None:
    try:
        results = list(DDGS().images(query=query, safesearch="moderate", max_results=8))
    except Exception as exc:
        print(f"Web image search failed for '{query}': {exc}")
        return None

    for item in results:
        img_url = (item.get("image") or "").strip()
        source_page = (item.get("url") or "").strip()
        result_title = clean_html(item.get("title") or "")
        if not img_url.startswith(("http://", "https://")):
            continue
        if any(img_url.lower().endswith(ext) for ext in (".svg", ".gif", ".webp")):
            continue
        try:
            resp = requests.get(
                img_url,
                headers={**HEADERS, "Referer": source_page or "https://duckduckgo.com/"},
                timeout=15,
            )
            resp.raise_for_status()
            if verify_and_save_image(resp.content, out_path):
                domain = urlparse(source_page or img_url).netloc.replace("www.", "")
                title_part = f"{result_title[:34]} — " if result_title else ""
                return f"Web: {title_part}{domain} | {source_page or img_url}"
        except Exception:
            continue
    return None


def fetch_wikimedia_image(query: str, out_path: str) -> str | None:
    try:
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": 6,
            "gsrlimit": 5,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata",
            "iiurlwidth": 1200,
            "format": "json",
        }
        r = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params=params,
            headers={"User-Agent": "LegalNewsBot/1.0"},
            timeout=15,
        )
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        for page in pages.values():
            info_list = page.get("imageinfo") or []
            if not info_list:
                continue
            info = info_list[0]
            img_url = info.get("thumburl") or info.get("url") or ""
            if not img_url or img_url.lower().endswith((".svg", ".pdf", ".tif", ".tiff", ".webp")):
                continue
            img_resp = requests.get(img_url, headers={"User-Agent": "LegalNewsBot/1.0"}, timeout=20)
            img_resp.raise_for_status()
            if verify_and_save_image(img_resp.content, out_path):
                meta = info.get("extmetadata", {})
                artist = clean_html((meta.get("Artist") or {}).get("value", ""))
                lic = clean_html((meta.get("LicenseShortName") or {}).get("value", "CC/Public Domain"))
                who = f"{artist[:28]} — " if artist else ""
                return f"Wikimedia: {who}{lic} | {img_url}"
    except Exception as exc:
        print(f"Wikimedia fallback failed for '{query}': {exc}")
    return None


def make_editorial_backdrop(out_path: str, accent_rgb=(185, 28, 28)) -> str:
    img = Image.new("RGB", (W, H), (10, 15, 28))
    draw = ImageDraw.Draw(img)
    for y in range(0, H, 4):
        v = int(10 + (y / H) * 20)
        draw.line([(0, y), (W, y)], fill=(v, v + 6, v + 18), width=4)
    draw.ellipse([(640, 240), (1280, 880)], outline=(30, 58, 138), width=4)
    draw.ellipse([(720, 320), (1200, 800)], outline=accent_rgb, width=3)
    img.save(out_path)
    return "Editorial Frame | Local"


# --- 4. GENERATE VOICEOVER ---
async def make_audio(text: str, filename: str = "voice.mp3") -> None:
    communicate = edge_tts.Communicate(text, "en-IN-PrabhatNeural", rate="-3%")
    await communicate.save(filename)


asyncio.run(make_audio(full_narration, "voice.mp3"))
audio = AudioFileClip("voice.mp3")
total_duration = max(float(audio.duration), 8.0)
scene_duration = total_duration / len(scenes)


# --- 5. RENDER PROFESSIONAL VERTICAL SLIDES ---
f_top_pill = load_font(30, bold=True)
f_sec_tag = load_font(26, bold=True)
f_headline = load_font(46, bold=True)
f_body = load_font(38, bold=False)
f_credit = load_font(22, bold=False)

accent_palette = [
    (220, 38, 38),    # Crimson
    (37, 99, 235),    # Royal Blue
    (180, 83, 9),     # Amber
    (15, 118, 110),   # Teal
]

credits_log = []
clips = []

for idx, sc in enumerate(scenes):
    raw_path = f"raw_{idx}.jpg"
    credit_full = None
    accent = accent_palette[idx % len(accent_palette)]

    if idx == 0 and rss_image_url:
        try:
            resp = requests.get(rss_image_url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            if verify_and_save_image(resp.content, raw_path):
                domain = urlparse(rss_image_url).netloc.replace("www.", "")
                credit_full = f"RSS ({domain}) | {rss_image_url}"
        except Exception:
            credit_full = None

    if not credit_full:
        credit_full = fetch_web_search_image(sc["search_query"], raw_path)

    if not credit_full:
        credit_full = fetch_wikimedia_image(sc["search_query"], raw_path)

    if not credit_full:
        credit_full = make_editorial_backdrop(raw_path, accent)

    credits_log.append(f"Scene {idx + 1} [{sc['label']}]: {credit_full}")

    with Image.open(raw_path) as im:
        photo = im.convert("RGB")

    target_ratio = W / H
    src_ratio = photo.width / photo.height
    if src_ratio > target_ratio:
        new_w = int(photo.height * target_ratio)
        left = (photo.width - new_w) // 2
        photo = photo.crop((left, 0, left + new_w, photo.height))
    else:
        new_h = int(photo.width / target_ratio)
        top = (photo.height - new_h) // 2
        photo = photo.crop((0, top, photo.width, top + new_h))
    photo = photo.resize((W, H), Image.Resampling.LANCZOS)

    # Dark cinematic gradient overlay
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay, "RGBA")
    for y in range(0, H, 4):
        alpha = int(135 + 65 * (y / H))
        ov_draw.line([(0, y), (W, y)], fill=(8, 12, 24, min(alpha, 215)), width=4)

    frame = Image.alpha_composite(photo.convert("RGBA"), overlay)
    draw = ImageDraw.Draw(frame, "RGBA")

    # Top header bar
    draw.rectangle([(0, 96), (W, 186)], fill=(15, 23, 42, 235))
    draw.rectangle([(0, 182), (W, 188)], fill=accent + (255,))
    header_label = f"{banner_tag}  •  {idx + 1}/{len(scenes)}"
    draw.text((W // 2, 141), header_label, fill=(248, 250, 252, 255), font=f_top_pill, anchor="mm")

    # Main editorial card
    draw.rounded_rectangle(
        [(56, 260), (W - 56, 1170)],
        radius=28,
        fill=(15, 23, 42, 225),
        outline=(148, 163, 184, 120),
        width=2,
    )
    draw.rounded_rectangle([(96, 302), (380, 352)], radius=10, fill=accent + (245,))
    draw.text((238, 327), sc["label"], fill="white", font=f_sec_tag, anchor="mm")

    active_font = f_headline if idx == 0 else f_body
    wrap_cols = 28 if idx == 0 else 36
    wrapped_text = textwrap.fill(sc["card_text"], width=wrap_cols)
    draw.multiline_text(
        (96, 388),
        wrapped_text,
        fill=(248, 250, 252, 255),
        font=active_font,
        spacing=16,
    )

    # Credit attribution line inside card footer
    short_credit = credit_full.split(" | ")[0]
    credit_line = textwrap.shorten(f"Credit: {short_credit}", width=78, placeholder="...")
    draw.text((96, 1118), credit_line, fill=(148, 163, 184, 240), font=f_credit)

    slide_file = f"slide_{idx}.png"
    frame.convert("RGB").save(slide_file)

    zoom_clip = (
        ImageClip(slide_file)
        .set_duration(scene_duration)
        .resize(lambda t: 1.0 + 0.02 * min(t, scene_duration))
        .set_position(("center", "center"))
    )
    clips.append(CompositeVideoClip([zoom_clip], size=(W, H)).set_duration(scene_duration))

base_video = concatenate_videoclips(clips, method="compose").set_audio(audio)

# --- 6. SMOOTH PROGRESS BAR & WORD-LEVEL CAPTIONS ---
progress_clips = []
steps = 30
step_dur = total_duration / steps
for s in range(steps):
    fill_w = int(((s + 1) / steps) * W)
    p_img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    p_draw = ImageDraw.Draw(p_img)
    p_draw.rectangle([(0, H - 22), (W, H)], fill=(30, 41, 59, 210))
    p_draw.rectangle([(0, H - 22), (fill_w, H)], fill=(239, 68, 68, 255))
    p_path = f"prog_{s}.png"
    p_img.save(p_path)
    progress_clips.append(ImageClip(p_path).set_start(s * step_dur).set_duration(step_dur))

subtitle_clips = []
try:
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    segments, _ = model.transcribe("voice.mp3", word_timestamps=True)
    for seg in segments:
        if not seg.words:
            continue
        for w_item in seg.words:
            token = re.sub(r"\s+", " ", (w_item.word or "").strip().upper())
            if not token or w_item.end <= w_item.start:
                continue
            txt = (
                TextClip(
                    token,
                    fontsize=72,
                    color="#FACC15",
                    font="DejaVu-Sans-Bold",
                    stroke_color="black",
                    stroke_width=3,
                )
                .set_position(("center", 1295))
                .set_start(float(w_item.start))
                .set_end(float(w_item.end))
            )
            subtitle_clips.append(txt)
except Exception as exc:
    print(f"Caption generation skipped: {exc}")

# --- 7. EXPORT FINAL VIDEO & LOGS ---
output_file = "final_legal_short.mp4"
final = CompositeVideoClip([base_video] + progress_clips + subtitle_clips, size=(W, H))
final.write_videofile(
    output_file,
    fps=24,
    codec="libx264",
    audio_codec="aac",
    threads=2,
    logger=None,
)

with open("credits.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(credits_log) + "\n")

if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
    if article_link:
        with open("posted.txt", "a", encoding="utf-8") as f:
            f.write(article_link + "\n")
    print(f"Generated {output_file} ({total_duration:.1f}s)")
    print("Credits:\n" + "\n".join(credits_log))
else:
    raise RuntimeError("Failed to generate final_legal_short.mp4")
