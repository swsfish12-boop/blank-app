#!/usr/bin/env python3
"""실화 괴담 쇼츠 생성기.

stories/<id>.json (대본) -> output/<id>.mp4 (1080x1920, 30fps, 한국어 TTS + 자막 + 효과음)

사용법:
    python3 make_short.py stories/robert_the_doll.json
"""
import asyncio
import json
import math
import os
import random
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


W, H, FPS = 1080, 1920, 30
SR = 24000
LEAD_IN, GAP, TAIL = 0.3, 0.28, 1.2

ROOT = Path(__file__).resolve().parent
FONT_SANS = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT_SERIF = "/usr/share/fonts/truetype/nanum/NanumMyeongjoBold.ttf"
RED = (220, 38, 38)


def font(size, serif=False):
    if serif:
        return ImageFont.truetype(FONT_SERIF, size)
    return ImageFont.truetype(FONT_SANS, size, index=1)  # index 1 = KR


# ---------------------------------------------------------------- TTS
def tts_backend(story):
    """'azure'(정식 Azure Speech, 상업 이용 OK) 또는 'edge'(무료 비공식). 환경변수가 우선."""
    return os.environ.get("TTS_BACKEND") or story.get("tts", "edge")


def azure_tts(text, voice, rate, pitch, mp3_path):
    """Azure Speech REST API. 키는 네트워크 프록시가 주입하므로 AZURE_SPEECH_KEY는 비워 둬도 된다."""
    from xml.sax.saxutils import escape
    region = os.environ.get("AZURE_SPEECH_REGION", "koreacentral")
    key = os.environ.get("AZURE_SPEECH_KEY", "injected-by-proxy")
    ssml = (f"<speak version='1.0' xml:lang='ko-KR'><voice name='{voice}'>"
            f"<prosody rate='{rate}' pitch='{pitch}'>{escape(text)}</prosody></voice></speak>")
    url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
    r = subprocess.run(["curl", "-sS", "-m", "60", "-o", str(mp3_path), "-w", "%{http_code}", "-X", "POST", url,
                        "-H", f"Ocp-Apim-Subscription-Key: {key}",
                        "-H", "Content-Type: application/ssml+xml",
                        "-H", "X-Microsoft-OutputFormat: audio-24khz-96kbitrate-mono-mp3",
                        "-H", "User-Agent: creepy-shorts",
                        "--data-binary", ssml.encode("utf-8")], capture_output=True)
    code = r.stdout.decode().strip()
    if code != "200":
        raise RuntimeError(f"Azure TTS 실패 (HTTP {code}): {r.stderr.decode()[:200]} "
                           f"{Path(mp3_path).read_text(errors='ignore')[:200] if Path(mp3_path).exists() else ''}")


def estimate_words(text, audio):
    """단어 타이밍이 없는 TTS용: 실제 발화 구간을 글자 수 + 쉼표/마침표 쉼 비율로 나눈다."""
    env = np.abs(audio)
    on = np.where(env > 0.02)[0]
    t0, t1 = (on[0] / SR, on[-1] / SR) if len(on) else (0.0, len(audio) / SR)
    words = text.split()
    weights = []
    for w_ in words:
        wt = len(w_.strip(",.?!")) + 0.5
        if w_.endswith(","):
            wt += 2.5
        elif w_.endswith((".", "?", "!")):
            wt += 4.0
        weights.append(wt)
    total = sum(weights) or 1
    out, cur = [], t0
    for w_, wt in zip(words, weights):
        dur = (t1 - t0) * wt / total
        out.append((cur, dur, w_.strip(",.?!")))
        cur += dur
    return out


async def _tts(text, voice, rate, pitch, mp3_path):
    import edge_tts
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    com = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch,
                               boundary="WordBoundary", proxy=proxy)
    words = []
    with open(mp3_path, "wb") as f:
        async for chunk in com.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                words.append((chunk["offset"] / 1e7, chunk["duration"] / 1e7, chunk["text"]))
    return words


def decode(mp3_path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp3_path), "-f", "s16le",
                          "-ac", "1", "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def write_wav(path, x):
    x = np.clip(x, -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((x * 32767).astype(np.int16).tobytes())


# ---------------------------------------------------------------- captions
def chunk_words(line_text, words, max_chars=9):
    """어절 단위 단어 경계를 화면용 자막 묶음으로 합친다. 반환: [(start, text, char_offset)]"""
    pos, located = 0, []
    for off, dur, t in words:
        i = line_text.find(t, pos)
        if i < 0:
            i = pos
        located.append((off, dur, t, i))
        pos = i + len(t)
    chunks, cur = [], None
    for off, dur, t, i in located:
        if cur is None:
            cur = [off, i, i + len(t)]
            continue
        cand_len = (i + len(t)) - cur[1]
        punct_after = line_text[cur[2]:cur[2] + 1] in (".", ",", "?", "!")
        if cand_len <= max_chars and not punct_after:
            cur[2] = i + len(t)
        else:
            chunks.append(tuple(cur))
            cur = [off, i, i + len(t)]
    if cur:
        chunks.append(tuple(cur))
    out = []
    for start, a, b in chunks:
        # 문장부호까지 포함해서 보여주되 끝의 쉼표/마침표는 떼어낸다
        txt = line_text[a:b].strip().rstrip(",.")
        out.append((start, txt, a))
    return out


def highlight_mask(line_text, hls):
    mask = [False] * len(line_text)
    for h in hls:
        s = line_text.find(h)
        while s >= 0:
            for k in range(s, s + len(h)):
                mask[k] = True
            s = line_text.find(h, s + 1)
    return mask


def draw_caption(img, text, offset, mask, pop, hl_color=(255, 214, 64), shake=(0, 0)):
    """화면 중하단에 큼직한 자막. pop: 0..1 등장 애니메이션, shake: 흔들림 오프셋"""
    f = font(92)
    d = ImageDraw.Draw(img)
    words = text.split(" ")
    lines, cur = [], ""
    for w_ in words:
        test = (cur + " " + w_).strip()
        if d.textlength(test, font=f) > W - 140 and cur:
            lines.append(cur)
            cur = w_
        else:
            cur = test
    lines.append(cur)
    layer = Image.new("RGBA", (W, 420), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    y = 210 - len(lines) * 62
    char_i = offset
    for ln in lines:
        x = (W - ld.textlength(ln, font=f)) / 2
        for ch in ln:
            hl = char_i < len(mask) and mask[char_i]
            fill = tuple(hl_color) if hl else (255, 255, 255)
            ld.text((x, y), ch, font=f, fill=fill, stroke_width=9, stroke_fill=(0, 0, 0))
            x += ld.textlength(ch, font=f)
            char_i += 1
        char_i += 1
        y += 124
    s = 1.0 + 0.14 * (1 - pop) ** 2
    if s != 1.0:
        nw, nh = int(W * s), int(420 * s)
        layer = layer.resize((nw, nh), Image.BILINEAR).crop(((nw - W) // 2, (nh - 420) // 2,
                                                             (nw - W) // 2 + W, (nh - 420) // 2 + 420))
    img.alpha_composite(layer, (int(shake[0]), 1330 + int(shake[1])))


# ---------------------------------------------------------------- scene art
def radial(cx, cy, r, inner, outer):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
    t = np.clip(dist, 0, 1)[..., None]
    arr = np.array(inner, np.float32) * (1 - t) + np.array(outer, np.float32) * t
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def fabric(img, strength=10, seed=1):
    rng = np.random.default_rng(seed)
    a = np.asarray(img).astype(np.int16)
    n = rng.integers(-strength, strength, (H, W, 1), dtype=np.int16)
    return Image.fromarray(np.clip(a + n, 0, 255).astype(np.uint8))


def draw_doll(img, cx=540, top=430, scale=1.0, eye_glint=True, dark=False):
    d = ImageDraw.Draw(img)
    s = scale
    body = (18, 16, 15) if dark else (34, 30, 27)
    shirt = (52, 50, 48) if not dark else (28, 27, 26)

    def P(x, y):
        return (cx + x * s, top + y * s)
    # 몸통 (세일러복)
    d.polygon([P(-150, 430), P(150, 430), P(230, 980), P(-230, 980)], fill=shirt)
    d.polygon([P(-150, 430), P(150, 430), P(0, 640)], fill=(70, 66, 62) if not dark else (36, 34, 33))  # 세일러 칼라
    d.polygon([P(-40, 600), P(40, 600), P(0, 690)], fill=(90, 20, 20))  # 넥타이
    # 팔
    d.rounded_rectangle([P(-245, 450), P(-165, 860)], radius=40 * s, fill=shirt)
    d.rounded_rectangle([P(165, 450), P(245, 860)], radius=40 * s, fill=shirt)
    # 머리
    d.ellipse([P(-175, 70), P(175, 430)], fill=body)
    # 모자
    d.ellipse([P(-230, 40), P(230, 140)], fill=(60, 58, 56) if not dark else (30, 29, 28))
    d.rounded_rectangle([P(-150, -30), P(150, 95)], radius=30 * s, fill=(70, 68, 66) if not dark else (34, 33, 32))
    # 눈 (검은 단추)
    for ex in (-70, 70):
        d.ellipse([P(ex - 34, 205), P(ex + 34, 273)], fill=(4, 4, 4))
        if eye_glint:
            d.ellipse([P(ex - 14, 215), P(ex + 2, 231)], fill=(200, 200, 190))
    # 입 (꿰맨 선)
    d.line([P(-55, 345), P(55, 345)], fill=(8, 8, 8), width=int(5 * s))
    for k in range(-50, 60, 20):
        d.line([P(k, 333), P(k, 357)], fill=(8, 8, 8), width=int(3 * s))


def draw_case(img):
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([190, 330, 890, 1500], outline=(150, 150, 155, 200), width=6)
    d.rectangle([150, 1500, 930, 1600], fill=(48, 30, 20, 255))
    d.rectangle([150, 1500, 930, 1512], fill=(80, 55, 35, 255))
    for x0 in (260, 330, 700):
        d.polygon([(x0, 330), (x0 + 60, 330), (x0 - 140, 1500), (x0 - 200, 1500)], fill=(255, 255, 255, 14))
    # 명판
    d.rectangle([420, 1530, 660, 1575], fill=(150, 125, 70, 255))
    d.text((540, 1552), "ROBERT", font=font(30, serif=True), fill=(40, 30, 10), anchor="mm")


def scene_image(scene, args, rnd):
    if scene == "envelope":
        img = radial(540, 820, 1100, (46, 40, 34), (5, 4, 4))
        paper = Image.new("RGBA", (820, 540), (214, 200, 170, 255))
        pd = ImageDraw.Draw(paper)
        pd.line([(0, 0), (410, 300), (820, 0)], fill=(160, 145, 115), width=4)
        pd.text((70, 360), "To: Robert", font=font(54, serif=True), fill=(50, 40, 30))
        pd.text((70, 430), "Fort East Martello Museum, Key West", font=font(30, serif=True), fill=(70, 60, 45))
        pd.ellipse([370, 250, 450, 330], fill=(140, 20, 20))
        paper = paper.rotate(-6, expand=True, resample=Image.BICUBIC)
        img = img.convert("RGBA")
        img.alpha_composite(paper, (110, 520))
        return fabric(img.convert("RGB"), 8)
    if scene == "location":
        img = radial(540, 800, 1200, (18, 26, 30), (2, 3, 4))
        d = ImageDraw.Draw(img)
        for x in range(0, W, 90):
            d.line([(x, 0), (x, H)], fill=(24, 38, 44), width=1)
        for y in range(0, H, 90):
            d.line([(0, y), (W, y)], fill=(24, 38, 44), width=1)
        d.ellipse([500, 760, 580, 840], outline=RED, width=6)
        d.line([(540, 700), (540, 900)], fill=RED, width=2)
        d.line([(440, 800), (640, 800)], fill=RED, width=2)
        return img
    if scene == "year":
        img = radial(540, 900, 1100, (40, 30, 26), (4, 3, 3))
        d = ImageDraw.Draw(img)
        d.text((540, 880), args["year"], font=font(330, serif=True), fill=(92, 30, 28), anchor="mm")
        return fabric(img, 9)
    if scene in ("doll", "doll_dark"):
        dark = scene == "doll_dark"
        img = radial(540, 500, 1300, (70, 58, 46) if not dark else (30, 26, 24), (3, 3, 3))
        draw_doll(img, dark=dark)
        img = img.filter(ImageFilter.GaussianBlur(1.6))
        return fabric(img, 10, seed=rnd.randint(0, 99))
    if scene == "door":
        img = radial(540, 900, 1200, (24, 20, 18), (2, 2, 2))
        d = ImageDraw.Draw(img)
        d.rectangle([280, 360, 800, 1280], fill=(42, 30, 24))
        for y0 in (410, 850):
            d.rectangle([330, y0, 750, y0 + 380], outline=(28, 20, 16), width=10)
        d.ellipse([680, 800, 716, 836], fill=(120, 100, 60))
        d.rectangle([280, 1276, 800, 1288], fill=(230, 190, 120))  # 문틈 빛
        if "year" in args:
            d.text((540, 268), args["year"], font=font(96, serif=True), fill=(150, 32, 30), anchor="mm")
        return fabric(img.filter(ImageFilter.GaussianBlur(1.2)), 9)
    if scene in ("case", "camera", "glitch"):
        img = radial(540, 500, 1300, (64, 56, 48), (3, 3, 3))
        draw_doll(img, top=520, scale=0.82)
        draw_case(img)
        d = ImageDraw.Draw(img)
        if "year" in args:
            d.text((540, 268), args["year"], font=font(96, serif=True), fill=(150, 32, 30), anchor="mm")
        return fabric(img.filter(ImageFilter.GaussianBlur(1.0)), 10)
    if scene in ("letter", "typed"):
        img = radial(540, 800, 1100, (40, 34, 30), (4, 4, 4))
        paper = Image.new("RGBA", (860, 1000), (222, 212, 186, 255))
        pd = ImageDraw.Draw(paper)
        for y in range(160, 1000, 90):
            pd.line([(60, y), (800, y)], fill=(170, 160, 140), width=2)
        paper = paper.rotate(3, expand=True, resample=Image.BICUBIC)
        img = img.convert("RGBA")
        img.alpha_composite(paper, (90, 330))
        return fabric(img.convert("RGB"), 7)
    if scene == "wall":
        img = radial(540, 800, 1200, (52, 44, 36), (4, 4, 4))
        img = img.convert("RGBA")
        notes = ["Dear Robert,", "I'm sorry", "미안해", "Forgive me", "용서해 줘", "SORRY",
                 "Please...", "죄송합니다", "No more", "I didn't ask"]
        r = random.Random(7)
        for k in range(26):
            w_, h_ = r.randint(230, 340), r.randint(170, 260)
            tone = r.randint(175, 225)
            p = Image.new("RGBA", (w_, h_), (tone, tone - 10, tone - 35, 255))
            pd = ImageDraw.Draw(p)
            pd.text((22, 26), r.choice(notes), font=font(36, serif=True), fill=(60, 40, 30))
            for y in range(90, h_ - 20, 34):
                pd.line([(22, y), (w_ - r.randint(30, 90), y)], fill=(110, 95, 80), width=3)
            p = p.rotate(r.uniform(-14, 14), expand=True, resample=Image.BICUBIC)
            img.alpha_composite(p, (r.randint(-60, W - 200), r.randint(200, 1500)))
        return fabric(img.convert("RGB").filter(ImageFilter.GaussianBlur(0.8)), 8)
    if scene == "eyes":
        img = radial(540, 800, 1200, (30, 26, 24), (2, 2, 2))
        draw_doll(img, cx=540, top=-650, scale=3.4)
        return fabric(img.filter(ImageFilter.GaussianBlur(2.2)), 11)
    raise ValueError(scene)



# ---------------------------------------------------------------- AI images (Pollinations / Flux)
STYLES = {
    "cinema": (", cinematic horror film still, dark moody low-key lighting, desaturated, film grain, "
               "35mm photo, highly detailed, vertical composition, no text, no watermark"),
    "raw": "",
    "ukiyo": (", Edo period Japanese ukiyo-e woodblock print style, aged paper texture, muted colors, "
              "vertical composition, no text, no letters, no watermark"),
    "oldphoto": (", 1920s black and white antique photograph, grainy, scratched, faded, low contrast, "
                 "vertical composition, no text, no watermark"),
    "vhs": (", grainy 1990s VHS camcorder home video still, low resolution, washed out colors, "
            "overcast, slightly blurry, vertical composition, no text, no watermark"),
}
STYLE = STYLES["cinema"]


def fetch_image(prompt, seed, path, tries=4, style="cinema"):
    """Pollinations(무료, 키 불필요)로 세로 이미지를 받아 path에 저장. 이미 있으면 재사용."""
    import time
    import urllib.parse
    if path.exists() and path.stat().st_size > 10000:
        return path
    url = ("https://image.pollinations.ai/prompt/" + urllib.parse.quote(prompt + STYLES.get(style, STYLE))
           + f"?width=1080&height=1920&nologo=true&private=true&seed={seed}&model=flux")
    r = None
    for k in range(tries):
        r = subprocess.run(["curl", "-sS", "-m", "180", "-o", str(path), "-w", "%{http_code}", url],
                           capture_output=True, text=True)
        if r.stdout.strip() == "200" and path.exists() and path.stat().st_size > 10000:
            return path
        if path.exists():
            path.unlink()
        time.sleep(5 * (k + 1))
    raise RuntimeError(f"이미지 생성 실패: {prompt[:40]}... ({r.stdout} {r.stderr[:200]})")


def image_scene(path, args):
    """생성 이미지 -> 1080x1920 배경. 워터마크 제거 + 공포 톤 색보정.
    args.grade: 'dark' | 'red_pop'(빨간색만 남기는 흑백) | 'vhs' 를 리스트나 문자열로."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    im = im.crop((0, 0, w, int(h * 0.93)))  # 하단 워터마크 제거
    w, h = im.size
    sc = max(W / w, H / h)
    im = im.resize((int(w * sc + 1), int(h * sc + 1)), Image.LANCZOS)
    w, h = im.size
    im = im.crop(((w - W) // 2, (h - H) // 2, (w - W) // 2 + W, (h - H) // 2 + H))
    grade = args.get("grade", [])
    grade = [grade] if isinstance(grade, str) else list(grade)
    if "vhs" in grade:
        im = im.filter(ImageFilter.GaussianBlur(1.3))
    a = np.asarray(im).astype(np.float32)
    gray = a.mean(axis=2, keepdims=True)
    if "red_pop" in grade:
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        m = np.clip((r - np.maximum(g, b) - 20) / 50, 0, 1)[..., None]
        red = a * np.array([1.18, 0.75, 0.75], np.float32)
        a = gray * (1 - m) + red * m                              # 빨간색만 남기고 흑백
    else:
        a = gray + (a - gray) * 0.7
        a = a * np.array([0.92, 0.95, 1.0], np.float32)
    a *= 0.82
    if "old" in grade:
        g2 = a.mean(axis=2, keepdims=True)
        a = g2 * np.array([1.06, 0.97, 0.82], np.float32)        # 바랜 세피아 흑백
        a = a * 0.9 + 10
    if "vhs" in grade:
        a = a * 0.88 + 14
        a *= np.array([1.0, 1.02, 0.94], np.float32)             # 바랜 테이프 색
    if "dark" in grade or "red_pop" in grade:
        # 너무 어두운 원본은 먼저 밝혀서 디테일이 남게 한다
        mean = float(a.mean())
        if mean < 70:
            a = np.clip(a * min(2.2, 70 / max(mean, 1)), 0, 255)
        a = np.clip((a - 128) * 1.15 + 118, 0, 255) * 0.9
        yy2, xx2 = np.mgrid[0:H, 0:W].astype(np.float32)
        rr = np.sqrt(((xx2 - W / 2) / (W * 0.6)) ** 2 + ((yy2 - H * 0.42) / (H * 0.45)) ** 2)
        a *= (1 - 0.7 * np.clip(rr - 0.4, 0, 1))[..., None]
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    a *= 1 - 0.55 * np.clip((yy - 0.62) / 0.3, 0, 1)            # 자막 영역 어둡게
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    if "year" in args:
        d.text((540, 300), args["year"], font=font(150, serif=True), fill=(205, 40, 36),
               anchor="mm", stroke_width=4, stroke_fill=(0, 0, 0))
    return img


# ---------------------------------------------------------------- per-frame overlays
def scene_overlay(img, scene, args, lt, ldur, line_text):
    d = ImageDraw.Draw(img)
    if scene == "location":
        en = args["en"]
        n = int(min(1, lt / 1.0) * len(en))
        d.text((540, 600), en[:n], font=font(72, serif=True), fill=(235, 235, 230), anchor="mm",
                   stroke_width=4, stroke_fill=(0, 0, 0))
        if lt > 0.6:
            d.text((540, 700), args["coord"], font=font(42), fill=(200, 205, 205), anchor="mm",
                       stroke_width=3, stroke_fill=(0, 0, 0))
    elif scene == "camera":
        c, L, t = (235, 235, 235), 90, 6
        for (x, y, sx, sy) in ((70, 260, 1, 1), (1010, 260, -1, 1), (70, 1700, 1, -1), (1010, 1700, -1, -1)):
            d.line([(x, y), (x + sx * L, y)], fill=c, width=t)
            d.line([(x, y), (x, y + sy * L)], fill=c, width=t)
        if int(lt * 2) % 2 == 0:
            d.ellipse([110, 300, 150, 340], fill=(230, 30, 30))
        d.text((165, 320), "REC", font=font(44), fill=c, anchor="lm")
        secs = int(lt)
        d.text((970, 320), f"00:00:{secs:02d}", font=font(40), fill=c, anchor="rm")
    elif scene == "typed":
        rows = args.get("lines", [])
        total = sum(len(b) for b in rows) or 1
        n = int(min(1, lt / max(0.6, ldur * 0.75)) * total)
        fm = ImageFont.truetype(FONT_SANS.replace("Bold", "Regular"), 54, index=6)   # Mono KR
        y = 560
        for b in rows:
            k = min(len(b), max(0, n))
            d.text((170, y), b[:k], font=fm, fill=(35, 28, 24))
            if 0 < k < len(b) and int(lt * 4) % 2 == 0:
                d.text((170 + d.textlength(b[:k], font=fm), y), "▌", font=fm, fill=(60, 50, 40))
            n -= len(b)
            y += 96
    elif scene == "letter":
        body = ["로버트에게,", "", "허락 없이 사진을 찍어서", "정말 미안해.", "", "제발 용서해 줘."]
        total = sum(len(b) for b in body)
        n = int(min(1, lt / max(0.5, ldur * 0.8)) * total)
        y = 470
        for b in body:
            k = min(len(b), max(0, n))
            d.text((200, y), b[:k], font=font(58, serif=True), fill=(40, 30, 25))
            n -= len(b)
            y += 90


def apply_glitch(arr, rng, amount):
    out = arr.copy()
    for _ in range(int(6 * amount)):
        y = rng.integers(0, H - 60)
        h = rng.integers(8, 70)
        out[y:y + h] = np.roll(out[y:y + h], rng.integers(-80, 80), axis=1)
    sh = int(14 * amount)
    if sh:
        out[..., 0] = np.roll(out[..., 0], sh, axis=1)
        out[..., 2] = np.roll(out[..., 2], -sh, axis=1)
    return out


# ---------------------------------------------------------------- audio design (horror score)
def read_wav(path):
    with wave.open(str(path)) as w:
        ch, n = w.getnchannels(), w.getnframes()
        x = np.frombuffer(w.readframes(n), np.int16).astype(np.float32) / 32768.0
    return x.reshape(-1, ch).mean(axis=1) if ch > 1 else x


def ffmpeg_fx(src, dst, af):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-af", af, "-ar", str(SR), "-ac", "1", str(dst)],
                   check=True)
    return read_wav(dst)


def lowpass(x, k):
    return np.convolve(x, np.ones(k, np.float32) / k, mode="same")


def fft_conv(x, ir):
    n = len(x) + len(ir) - 1
    N = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, N) * np.fft.rfft(ir, N), N)[:len(x)].astype(np.float32)


def reverb(x, secs=2.5, wet=0.35, seed=0, damp=4):
    rng = np.random.default_rng(seed)
    L = int(secs * SR)
    t = np.arange(L) / SR
    ir = rng.normal(0, 1, L) * np.exp(-t * (6.9 / secs))
    ir = lowpass(ir, damp)
    ir /= np.sqrt(np.sum(ir ** 2)) + 1e-9
    return (1 - wet) * x + wet * fft_conv(x, ir)


def add_at(buf, t, sig, gain=1.0):
    i = int(t * SR)
    if i >= len(buf) or i + len(sig) <= 0:
        return
    if i < 0:
        sig, i = sig[-i:], 0
    seg = buf[i:i + len(sig)]
    seg += sig[:len(seg)] * gain


def music_box_track(n, t0, t1, rng):
    """원곡 자장가(오르골). t1 직전 태엽이 풀리듯 느려지고 음이 처지다가 멈춘다."""
    out = np.zeros(n, np.float32)
    mel = [0, 3, 7, 5, 3, 2, 0, None, -1, 2, 5, 3, 2, 0, -1, None,
           0, 3, 7, 8, 7, 5, 3, None, 2, 3, 2, 0, -1, 0, None, None]
    L = int(2.4 * SR)
    tt = np.arange(L) / SR
    att = np.clip(tt / 0.004, 0, 1)
    t, k, dt, pf = t0, 0, 0.42, 1.0
    while t < t1:
        s = mel[k % len(mel)]
        if t > t1 - 2.4:                       # 태엽 풀림
            dt *= 1.17
            pf *= 0.975
        if s is not None:
            f = 880 * 2 ** (s / 12) * pf * (1 + rng.normal(0, 0.002))
            tone = (np.sin(2 * np.pi * f * tt) + 0.5 * np.sin(2 * np.pi * 2.01 * f * tt)
                    + 0.22 * np.sin(2 * np.pi * 3.98 * f * tt) + 0.1 * np.sin(2 * np.pi * 6.2 * f * tt))
            tone *= np.exp(-tt * 3.2) * att * 0.09
            add_at(out, t, tone)
        t += dt
        k += 1
    # 멈추는 '딸깍'
    click = rng.normal(0, 1, int(0.03 * SR)) * np.exp(-np.arange(int(0.03 * SR)) / SR * 120) * 0.25
    add_at(out, t1, click)
    return reverb(out, secs=2.8, wet=0.45, seed=4)


def heartbeat(buf, ta, tb, gain=1.0):
    t = ta
    while t < tb:
        p = (t - ta) / max(tb - ta, 0.1)
        L = int(0.4 * SR)
        tt = np.arange(L) / SR
        th = np.sin(2 * np.pi * (46 + 34 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 13)
        add_at(buf, t, th, 0.55 * gain * (0.7 + 0.5 * p))
        add_at(buf, t + 0.17, th, 0.38 * gain * (0.7 + 0.5 * p))
        t += 1.05 - 0.5 * p


def riser(buf, ta, tb, rng):
    L = int((tb - ta) * SR)
    if L <= 0:
        return
    tt = np.arange(L) / SR
    env = (tt / tt[-1]) ** 2.2
    f = 160 * (6.0 ** (tt / tt[-1]))
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.6 * np.sin(2 * np.pi * np.cumsum(f * 1.06) / SR)
    noise = rng.normal(0, 1, L)
    noise -= lowpass(noise, 6)
    add_at(buf, ta, (tone * 0.10 + noise * 0.09) * env)


def stinger(buf, t, rng, amp=1.0, bright=False):
    L = int(2.8 * SR)
    tt = np.arange(L) / SR
    boom = np.sin(2 * np.pi * (55 - 22 * tt) * tt) * np.exp(-tt * 2.0) * 0.6
    att = np.clip(tt / 0.008, 0, 1)
    cluster = np.zeros(L, np.float32)
    for f0 in ([1046, 1108, 1174, 1661] if not bright else [1661, 1760, 1864, 2489]):
        for h in range(1, 5):
            cluster += np.sin(2 * np.pi * f0 * h * tt + rng.uniform(0, 6)) / h
    cluster *= att * np.exp(-tt * (2.4 if not bright else 3.5)) * 0.05
    noise = rng.normal(0, 1, L)
    noise -= lowpass(noise, 4)
    noise *= np.exp(-tt * 7) * 0.22
    sig = reverb(boom + cluster + noise, secs=2.2, wet=0.3, seed=9)
    add_at(buf, t, sig, amp)


def footsteps(buf, ta, count, rng, gain=1.0, gap=0.62):
    """다락 나무 바닥 위 발소리: 둔탁한 쿵 + 가끔 삐걱."""
    t = ta
    for k in range(count):
        L = int(0.35 * SR)
        tt = np.arange(L) / SR
        thud = np.sin(2 * np.pi * (70 + 40 * np.exp(-tt * 40)) * tt) * np.exp(-tt * 22)
        nz = lowpass(rng.normal(0, 1, L), 12) * np.exp(-tt * 30) * 1.5
        add_at(buf, t, reverb(thud + nz, secs=0.9, wet=0.35, seed=k), 0.45 * gain)
        if rng.random() < 0.45:
            Lc = int(0.5 * SR)
            tc = np.arange(Lc) / SR
            f = 380 + 120 * np.sin(2 * np.pi * 1.3 * tc) + rng.uniform(-40, 40)
            creak = np.sin(2 * np.pi * np.cumsum(f) / SR) * (0.5 + 0.5 * np.sin(2 * np.pi * 37 * tc))
            creak *= np.sin(np.pi * tc / tc[-1]) * 0.05
            add_at(buf, t + 0.05, reverb(creak, secs=1.2, wet=0.4, seed=k + 50), gain)
        t += gap * rng.uniform(0.85, 1.2)


def temple_bell(buf, t, gain=1.0):
    """일본 사찰 종(린) 소리: 비조화 배음 + 긴 여운."""
    L = int(6 * SR)
    tt = np.arange(L) / SR
    f0 = 523.0
    sig = np.zeros(L, np.float32)
    for ratio, amp, dec in [(1, 1.0, 0.55), (2.76, 0.55, 0.9), (5.40, 0.3, 1.6), (8.93, 0.15, 2.6), (0.5, 0.25, 0.4)]:
        sig += amp * np.sin(2 * np.pi * f0 * ratio * tt) * np.exp(-tt * dec)
    sig *= (1 + 0.15 * np.sin(2 * np.pi * 3.1 * tt)) * np.clip(tt / 0.003, 0, 1)
    add_at(buf, t, reverb(sig * 0.06, secs=3.5, wet=0.4, seed=77), gain)


def horror_bed(total, lines, story, js, rng):
    n = int(total * SR)
    t = np.arange(n) / SR
    mus = story.get("music", {})
    starts = [ln["_start"] for ln in lines]
    ends = [ln["_start"] + ln["_dur"] for ln in lines]
    last_end = ends[-1]
    mb_end = ends[mus["box_until"]] + 0.25 if "box_until" in mus else 0
    climax = starts[mus.get("climax", len(lines) - 1)]

    # 불협화음 드론 (단2도 맥놀이) + 바람
    drone = sum(a * np.sin(2 * np.pi * f * t + ph) for f, a, ph in
                [(55, 0.5, 0), (58.27, 0.42, 1), (82.4, 0.25, 2), (87.31, 0.2, 3), (164.8, 0.06, 4)])
    drone *= 0.65 + 0.35 * np.sin(2 * np.pi * 0.09 * t)
    wind = np.cumsum(rng.normal(0, 1, n)).astype(np.float32)
    wind -= lowpass(wind, 4000)
    wind /= np.max(np.abs(wind)) + 1e-9
    wind *= 0.5 + 0.5 * np.sin(2 * np.pi * 0.11 * t + 2 * np.sin(2 * np.pi * 0.03 * t))
    keys_t = [0, mb_end, mb_end + 0.6, climax, last_end + 0.05]
    keys_v = [0.35, 0.35, 1.0, 1.35, 1.35]
    lvl = np.interp(t, keys_t, keys_v).astype(np.float32)
    bed = (0.085 * drone + 0.05 * wind) * lvl

    if mb_end:
        bed += music_box_track(n, 0.15, mb_end, rng)
    hb = mus.get("heartbeat")
    if hb:
        heartbeat(bed, starts[hb[0]], ends[hb[1]])
    for k, ln in enumerate(lines):
        if ln.get("riser"):
            riser(bed, max(0, starts[k] - ln["riser"]), starts[k], rng)
        if ln.get("sting"):
            stinger(bed, starts[k] - 0.02, rng, amp=1.0)
        elif ln.get("boom"):
            stinger(bed, starts[k] - 0.02, rng, amp=0.6)
        if ln.get("flash"):
            stinger(bed, ln["_flash_t"], rng, amp=0.55, bright=True)
        if ln.get("scene") == "typed":
            tt0 = starts[k]
            while tt0 < ends[k]:
                Lk = int(0.012 * SR)
                click = rng.normal(0, 1, Lk) * np.exp(-np.arange(Lk) / SR * 600)
                add_at(bed, tt0, bandpass(click, 900, 6000) * 0.5)
                add_at(bed, tt0 + 0.004, np.sin(2 * np.pi * 180 * np.arange(Lk) / SR) * np.exp(-np.arange(Lk) / SR * 400) * 0.25)
                tt0 += rng.uniform(0.07, 0.16)
        if ln.get("bell"):
            temple_bell(bed, starts[k] - 0.3)
        if ln.get("steps"):
            footsteps(bed, starts[k] + 0.2, ln["steps"], rng)
        if ln.get("vhs"):
            i, j = int(starts[k] * SR), int(min(n, (ends[k] + 0.3) * SR))
            hiss = rng.normal(0, 1, j - i)
            bed[i:j] += (hiss - lowpass(hiss, 3)) * 0.012

    # 마지막 줄 뒤 정적 -> 점프 스케어
    if js:
        a, b = int((last_end + 0.1) * SR), int(js["t"] * SR)
        fade = np.linspace(1, 0, max(1, min(int(0.25 * SR), b - a)))
        bed[a:a + len(fade)] *= fade
        bed[a + len(fade):b] = 0
        if js.get("steps"):
            footsteps(bed, last_end + 0.35, js["steps"], rng, gain=1.2, gap=0.5)
        stinger(bed, js["t"], rng, amp=1.7, bright=True)
        stinger(bed, js["t"], rng, amp=0.9)
    f0 = int(0.4 * SR)
    bed[:f0] *= np.linspace(0, 1, f0)
    return bed


NARR_FX = ("rubberband=pitch=0.89,"
           "acompressor=threshold=0.1:ratio=4:attack=5:release=120:makeup=2,"
           "equalizer=f=160:t=q:w=1.2:g=4,highpass=f=60,"
           "aecho=0.8:0.6:70|170|310:0.28|0.18|0.1")
NARR_FX_LIGHT = ("acompressor=threshold=0.12:ratio=3:attack=5:release=120:makeup=1.5,"
                 "equalizer=f=140:t=q:w=1.2:g=2,highpass=f=60,"
                 "aecho=0.8:0.45:60|150|280:0.18|0.12|0.07")
NARR_FX_HUMAN = ("afftdn=nr=8:nf=-50,highpass=f=70,"
                 "equalizer=f=120:t=q:w=1:g=2.5,deesser=i=0.4,"
                 "equalizer=f=7500:t=q:w=2:g=-3,lowpass=f=10500,"
                 "acompressor=threshold=0.15:ratio=2.5:attack=10:release=180:makeup=1.4,"
                 "asoftclip=type=tanh")
WHISPER_FX = ("rubberband=pitch=0.97,highpass=f=500,"
              "acompressor=threshold=0.05:ratio=6:attack=3:release=80:makeup=4,"
              "aecho=0.8:0.85:45|120|240:0.45|0.3|0.2")
GHOST_FX = ("rubberband=pitch=1.1,highpass=f=350,lowpass=f=3200,"
            "aecho=0.8:0.9:260|540|900:0.5|0.35|0.2")


_QWEN = {}


def qwen_model():
    if "m" not in _QWEN:
        import torch
        from qwen_tts import Qwen3TTSModel
        torch.set_num_threads(os.cpu_count() or 4)
        _QWEN["m"] = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cpu",
                                                   dtype=torch.float32)
    return _QWEN["m"]


def qwen_clone(story, text, wav_path):
    """Qwen3-TTS(Apache-2.0) 목소리 복제: story['voice_ref'] 목소리로 text를 읽는다. 결과는 캐시."""
    import hashlib
    import soundfile as sf
    ref = ROOT / story["voice_ref"]
    ref_text = (ROOT / story["voice_ref"]).with_suffix(".txt").read_text(encoding="utf-8").strip()
    key = hashlib.md5((text + "|" + str(ref) + "|" + str(story.get("seed_voice", 0))).encode()).hexdigest()[:10]
    cache = wav_path.with_name(wav_path.stem + f"_{key}.wav")
    if not cache.exists():
        import torch
        torch.manual_seed(story.get("seed_voice", 0))
        w, sr = qwen_model().generate_voice_clone(text=text, language="Korean", ref_audio=str(ref),
                                                  ref_text=ref_text)
        sf.write(cache, w[0], sr)
    a, sr = sf.read(cache, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != SR:
        idx = np.linspace(0, len(a) - 1, int(len(a) * SR / sr))
        a = np.interp(idx, np.arange(len(a)), a).astype(np.float32)
    on = np.where(np.abs(a) > 0.02)[0]                    # 앞뒤 무음 정리
    if len(on):
        a = a[max(0, on[0] - int(0.05 * SR)): on[-1] + int(0.25 * SR)]
    return a


def bandpass(x, lo, hi, order=2):
    from scipy.signal import butter, lfilter
    b, a = butter(order, [lo / (SR / 2), hi / (SR / 2)], btype="band")
    return lfilter(b, a, x).astype(np.float32)


def breath_sound(rng, dur=None, gain=0.035):
    """사람이 문장 앞에서 들이쉬는 숨소리 (필터링한 노이즈 + 자연스러운 곡선)."""
    dur = dur or rng.uniform(0.32, 0.5)
    L = int(dur * SR)
    t = np.arange(L) / SR
    n = bandpass(rng.normal(0, 1, L), 350, 2600)
    n += 0.4 * bandpass(rng.normal(0, 1, L), 2600, 5200)
    env = np.sin(np.pi * t / dur) ** 1.6 * (0.55 + 0.45 * t / dur)
    return (n * env / (np.max(np.abs(n)) + 1e-9) * gain).astype(np.float32)


def room_tone(n, rng, level=0.0012):
    """완전 무음 대신 아주 작은 방 소음(핑크 노이즈)."""
    from scipy.signal import lfilter
    w = rng.normal(0, 1, n)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    pink = lfilter(b, a, w)
    pink = bandpass(pink, 60, 6000)
    return (pink / (np.std(pink) + 1e-9) * level).astype(np.float32)


def vary_tempo(audio, factor, work, tag):
    """줄마다 말 빠르기를 살짝 다르게 (음높이는 유지)."""
    if abs(factor - 1) < 0.005:
        return audio
    src, dst = work / f"_tmp_{tag}.wav", work / f"_tmp_{tag}_t.wav"
    write_wav(src, audio)
    return ffmpeg_fx(src, dst, f"rubberband=tempo={factor:.3f}")


def tts_line(story, text, voice, rate, pitch, mp3):
    if story.get("tts") == "qwen" and voice == story["voice"]:
        audio = qwen_clone(story, text, Path(mp3).with_suffix(".wav"))
        return audio, estimate_words(text, audio)
    if tts_backend(story) == "azure":
        azure_tts(text, voice, rate, pitch, mp3)
        audio = decode(mp3)
        return audio, estimate_words(text, audio)
    words = asyncio.run(_tts(text, voice, rate, pitch, mp3))
    return decode(mp3), words


# ---------------------------------------------------------------- main
def main(story_path):
    import concurrent.futures as cf
    story = json.loads(Path(story_path).read_text(encoding="utf-8"))
    sid = story["id"]
    work = ROOT / "build" / sid
    out_dir = ROOT / "output"
    work.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(11)

    # 1) TTS (줄마다 개별 생성 -> 정확한 자막 타이밍)
    lines = story["lines"]
    human = story.get("humanize", False)
    hrng = np.random.default_rng(story.get("seed_voice", 0) + 101)
    breaths = []
    t = LEAD_IN
    for i, ln in enumerate(lines):
        audio, words = tts_line(story, ln["text"], ln.get("voice", story["voice"]),
                                ln.get("rate", story["rate"]), ln.get("pitch", story["pitch"]),
                                work / f"line{i:02d}.mp3")
        if human and ln.get("fx") != "whisper":
            audio = vary_tempo(audio, hrng.uniform(0.95, 1.02), work, f"l{i}")
            words = estimate_words(ln["text"], audio)
            if i > 0 and hrng.random() < 0.8:
                b = breath_sound(hrng)
                breaths.append((t - len(b) / SR - 0.04, b))
        dur = len(audio) / SR
        ln["_start"], ln["_dur"], ln["_audio"] = t, dur, audio
        ln["_chunks"] = [(t + s, txt, off) for s, txt, off in chunk_words(ln["text"], words)]
        ln["_mask"] = highlight_mask(ln["text"], ln.get("hl", []))
        if ln.get("flash"):
            ln["_flash_t"] = t + dur * ln["flash"].get("at", 0.6)
        gap = hrng.uniform(0.34, 0.6) if human else GAP
        t += dur + gap + ln.get("pause", 0)
        print(f"  TTS {i + 1}/{len(lines)} {dur:.1f}s  {ln['text']}")
    last_end = lines[-1]["_start"] + lines[-1]["_dur"]
    js = None
    if story.get("jumpscare"):
        js = dict(story["jumpscare"])
        js["t"] = last_end + js.get("delay", 1.0)
        total = js["t"] + js.get("dur", 0.7) + 0.35
    else:
        total = last_end + TAIL
    print(f"총 길이 {total:.1f}s")

    # 2) 이미지 (병렬 생성)
    jobs = [(i, ln["img"], ln.get("seed", 1000 + i), ln.get("style", story.get("style", "cinema")),
             work / f"img{i:02d}.jpg") for i, ln in enumerate(lines) if ln.get("img") and not ln.get("photo")]
    if js and not js.get("photo"):
        jobs.append(("js", js["img"], js.get("seed", 666), js.get("style", "cinema"), work / "img_js.jpg"))
    with cf.ThreadPoolExecutor(3) as ex:
        futs = {ex.submit(fetch_image, p, s, path, 4, st): i for i, p, s, st, path in jobs}
        for fu in cf.as_completed(futs):
            fu.result()
            print(f"  이미지 {futs[fu]} OK")
    for i, ln in enumerate(lines):
        if ln.get("photo"):
            ln["_img"] = ROOT / ln["photo"]                      # 실제 사진 (assets/photos/CREDITS.md)
        elif ln.get("img"):
            ln["_img"] = work / f"img{i:02d}.jpg"

    # 3) 오디오: 내레이션 / 속삭임 / 유령 소리 / 배경음
    n = int(total * SR)
    narr = np.zeros(n, np.float32)
    wsp = np.zeros(n, np.float32)
    for ln in lines:
        add_at(wsp if ln.get("fx") == "whisper" else narr, ln["_start"], ln["_audio"])
    for bt, b in breaths:
        add_at(narr, bt, b)
    write_wav(work / "narr_raw.wav", narr)
    write_wav(work / "wsp_raw.wav", wsp)
    fx = {"light": NARR_FX_LIGHT, "human": NARR_FX_HUMAN}.get(story.get("narr_fx"), NARR_FX)
    narr = ffmpeg_fx(work / "narr_raw.wav", work / "narr.wav", fx)[:n]
    if story.get("narr_fx") == "human":
        narr = reverb(narr, secs=0.4, wet=0.08, seed=21, damp=6)    # 작은 방 울림
    wsp = ffmpeg_fx(work / "wsp_raw.wav", work / "wsp.wav", WHISPER_FX)[:n]

    ghost = np.zeros(n, np.float32)
    for k, g in enumerate(story.get("ghost_calls", [])):
        mp3 = work / f"ghost{k}.mp3"
        a, _ = tts_line(story, g["text"], g.get("voice", "ko-KR-SunHiNeural"), g.get("rate", "-15%"),
                        g.get("pitch", "+20Hz"), mp3)
        write_wav(work / f"ghost{k}_raw.wav", a)
        a = ffmpeg_fx(work / f"ghost{k}_raw.wav", work / f"ghost{k}.wav", GHOST_FX)
        ln = lines[g["line"]]
        add_at(ghost, ln["_start"] + ln["_dur"] + g.get("after", 0.3), a, g.get("gain", 0.8))
    wtexts = story.get("whispers", [])
    if wtexts:
        clips = []
        for k, w_ in enumerate(wtexts):
            a, _ = tts_line(story, w_, "ko-KR-SunHiNeural", "-10%", "+0Hz", work / f"wsp{k}.mp3")
            a = a[::-1].copy()                                   # 거꾸로 재생한 속삭임
            a = a - lowpass(a, 10)
            clips.append(reverb(a, secs=2.5, wet=0.6, seed=k))
        for ln in lines:
            if ln.get("whisper"):
                tt0, k = ln["_start"] + 0.1, 0
                while tt0 < ln["_start"] + ln["_dur"]:
                    c = clips[k % len(clips)]
                    add_at(ghost, tt0, c, 0.5)
                    tt0 += len(c) / SR * 0.8
                    k += 1

    bed = horror_bed(total, lines, story, js, rng)[:n]
    vo = np.abs(narr) + np.abs(wsp)
    duck = 1 - 0.3 * np.clip(lowpass(vo, 2400) * 8, 0, 1)
    mix = narr * 1.0 + wsp * 1.1 + ghost + bed * duck
    if human:
        mix += room_tone(n, hrng)
    mix /= np.max(np.abs(mix)) + 1e-9
    write_wav(work / "mix.wav", mix * 0.95)

    # 4) 영상 프레임
    rnd = random.Random(5)
    grains = [rng.integers(-11, 11, (H, W, 1), dtype=np.int16) for _ in range(6)]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    vign = (1 - 0.75 * np.clip(((xx - W / 2) / (W * 0.75)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2, 0, 1))[..., None]
    cache = {}
    js_img = flash_img = None
    if js:
        js_img = image_scene(ROOT / js["photo"] if js.get("photo") else work / "img_js.jpg",
                             {"grade": js.get("grade", ["red_pop", "dark"])})
        fa = np.asarray(js_img).astype(np.float32) * np.array([1.35, 0.55, 0.55], np.float32)
        flash_img = np.clip(fa, 0, 255).astype(np.uint8)
    hl_color = tuple(story.get("hl_color", [235, 40, 40]))
    tag = story.get("tag", "실화 괴담")

    nframes = int(total * FPS)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-maxrate", "8M",
                            "-bufsize", "16M", "-pix_fmt", "yuv420p", str(work / "video.mp4")],
                           stdin=subprocess.PIPE)
    f_small, f_tag, f_vhs = font(34), font(40), font(50)
    for fi in range(nframes):
        ft = fi / FPS
        # 점프 스케어 구간
        if js and ft >= js["t"]:
            if ft < js["t"] + js.get("dur", 0.7):
                p = (ft - js["t"]) / js.get("dur", 0.7)
                z = 1.0 + 0.35 * p
                cw, ch = W / z, H / z
                sx, sy = rng.integers(-30, 30), rng.integers(-30, 30)
                x0 = min(max(0, (W - cw) / 2 + sx), W - cw)
                y0 = min(max(0, (H - ch) / 2 + sy), H - ch)
                fr = js_img.resize((W, H), Image.BILINEAR, box=(x0, y0, x0 + cw, y0 + ch))
                arr = np.asarray(fr).astype(np.float32) * (1.25 if fi % 2 else 0.8)
                arr = apply_glitch(np.clip(arr, 0, 255).astype(np.uint8), rng, 0.8)
            else:
                arr = np.zeros((H, W, 3), np.uint8)
            enc.stdin.write(arr.tobytes())
            continue

        li = max(k for k, ln in enumerate(lines) if k == 0 or ln["_start"] - GAP / 2 <= ft)
        ln = lines[li]
        seg_start = 0 if li == 0 else ln["_start"] - GAP / 2
        seg_end = lines[li + 1]["_start"] - GAP / 2 if li + 1 < len(lines) else (js["t"] if js else total)
        lt, ldur = ft - seg_start, seg_end - seg_start

        # 순간 플래시(1~3프레임)
        if flash_img is not None and ln.get("flash") and 0 <= ft - ln["_flash_t"] < ln["flash"].get("frames", 3) / FPS:
            enc.stdin.write(apply_glitch(flash_img, rng, 1.0).tobytes())
            continue

        if ln.get("_img"):
            key = ("img", str(ln["_img"]), json.dumps(ln.get("args", {}), sort_keys=True))
        else:
            key = (ln["scene"], json.dumps(ln.get("args", {}), sort_keys=True))
        if key not in cache:
            cache[key] = (image_scene(ln["_img"], ln.get("args", {})) if ln.get("_img")
                          else scene_image(ln["scene"], ln.get("args", {}), rnd))
        base = cache[key]

        p = min(1.0, lt / max(ldur, 0.1))
        z = 1.0 + 0.08 * p if li % 2 == 0 else 1.08 - 0.08 * p
        if ln.get("zoom"):
            z = 1.0 + ln["zoom"] * p
        shake = ln.get("shake")
        sx = rng.uniform(-9, 9) if shake else 0
        sy = rng.uniform(-9, 9) if shake else 0
        cw, ch = W / z, H / z
        dx = (li % 3 - 1) * 20 * p + sx
        x0 = min(max(0.0, (W - cw) / 2 + dx), W - cw)
        y0 = min(max(0.0, (H - ch) / 2 + sy), H - ch)
        frame = base.resize((W, H), Image.BILINEAR, box=(x0, y0, x0 + cw, y0 + ch)).convert("RGBA")
        scene_overlay(frame, ln.get("scene", ""), ln.get("args", {}), lt, ldur, ln["text"])

        d = ImageDraw.Draw(frame)
        d.rounded_rectangle([60, 110, 60 + 28 + d.textlength(tag, font=f_tag), 178], radius=14,
                            fill=(150, 20, 20, 230))
        d.text((74, 144), tag, font=f_tag, fill=(255, 255, 255), anchor="lm")
        d.text((540, 1780), story["disclaimer"], font=f_small, fill=(170, 170, 170, 200), anchor="mm")
        if ln.get("vhs"):
            if int(ft * 1.6) % 2 == 0:
                d.text((70, 250), "▶ PLAY", font=f_vhs, fill=(240, 240, 240), stroke_width=3, stroke_fill=(0, 0, 0))
            cnt = int(ft + 767)
            d.text((1010, 250), f"0:{cnt // 60:02d}:{cnt % 60:02d}", font=f_vhs, fill=(240, 240, 240),
                   anchor="ra", stroke_width=3, stroke_fill=(0, 0, 0))
            d.text((1010, 320), story.get("vhs_date", ""), font=font(40), fill=(240, 240, 240),
                   anchor="ra", stroke_width=3, stroke_fill=(0, 0, 0))

        cur = None
        for cstart, txt, off in ln["_chunks"]:
            if ft >= cstart - 0.03:
                cur = (cstart, txt, off)
        if cur and ft <= ln["_start"] + ln["_dur"] + 0.25:
            pop = min(1.0, (ft - cur[0]) / 0.13)
            csh = (rng.uniform(-6, 6), rng.uniform(-5, 5)) if shake else (0, 0)
            draw_caption(frame, cur[1], cur[2], ln["_mask"], max(0, pop), hl_color, csh)

        arr = np.asarray(frame.convert("RGB")).astype(np.float32)
        flick = 1.0 - 0.07 * rng.random() - (0.45 if rng.random() < 0.02 else 0)
        fade_in = min(1.0, lt / 0.15) if (li == 0 or ln.get("_img") != lines[li - 1].get("_img")) else 1.0
        if js and li == len(lines) - 1 and ft > last_end:      # 정적 구간: 서서히 어두워짐
            fade_in *= max(0.25, 1 - (ft - last_end) / (js["t"] - last_end + 1e-6))
        arr = (arr + grains[fi % len(grains)]) * vign * flick * fade_in
        if ln.get("vhs"):
            arr[::4] *= 0.78                                    # 주사선
            arr[..., 0] = np.roll(arr[..., 0], 5, axis=1)        # 색 번짐
            band = int((ft * 260) % (H + 240)) - 120
            b0, b1 = max(0, band), min(H, band + 46)
            if b1 > b0:
                arr[b0:b1] = np.roll(arr[b0:b1], int(rng.integers(-40, 40)), axis=1) + 30
            if rng.random() < 0.04:
                arr = np.roll(arr, int(rng.integers(-10, 10)), axis=0)
        if ln.get("film"):
            for _ in range(int(rng.integers(0, 3))):                 # 필름 긁힘
                x = int(rng.integers(0, W))
                arr[:, x:x + 2] = arr[:, x:x + 2] * 0.5 + 90
            for _ in range(int(rng.integers(3, 12))):                # 먼지
                x, y = int(rng.integers(0, W - 6)), int(rng.integers(0, H - 6))
                arr[y:y + int(rng.integers(2, 6)), x:x + int(rng.integers(2, 6))] = rng.choice([15, 220])
            if rng.random() < 0.05:
                arr = np.roll(arr, int(rng.integers(-12, 12)), axis=0)
            arr *= 1.0 - 0.08 * rng.random()
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        if ln.get("glitch") and rng.random() < 0.4:
            arr = apply_glitch(arr, rng, rng.uniform(0.4, 1.0))
        elif rng.random() < 0.012:
            arr = apply_glitch(arr, rng, 0.35)
        enc.stdin.write(arr.tobytes())
        if fi % 300 == 0:
            print(f"  프레임 {fi}/{nframes}")
    enc.stdin.close()
    enc.wait()

    # 5) 합치기
    out = out_dir / f"{sid}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(work / "video.mp4"), "-i", str(work / "mix.wav"),
                    "-af", "loudnorm=I=-14:TP=-1.0:LRA=14,aresample=48000",
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", "-movflags", "+faststart", str(out)], check=True)
    print(f"완료: {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "stories" / "red_girl_taiwan.json"))
