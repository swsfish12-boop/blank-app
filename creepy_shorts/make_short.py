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


def draw_caption(img, text, offset, mask, pop):
    """화면 중하단에 큼직한 자막. pop: 0..1 등장 애니메이션"""
    f = font(92)
    d = ImageDraw.Draw(img)
    # 공백 기준 2줄 이하로 줄바꿈
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
            fill = (255, 214, 64) if hl else (255, 255, 255)
            ld.text((x, y), ch, font=f, fill=fill, stroke_width=9, stroke_fill=(0, 0, 0))
            x += ld.textlength(ch, font=f)
            char_i += 1
        char_i += 1  # 줄바꿈된 공백
        y += 124
    s = 1.0 + 0.14 * (1 - pop) ** 2
    if s != 1.0:
        nw, nh = int(W * s), int(420 * s)
        layer = layer.resize((nw, nh), Image.BILINEAR).crop(((nw - W) // 2, (nh - 420) // 2,
                                                             (nw - W) // 2 + W, (nh - 420) // 2 + 420))
    img.alpha_composite(layer, (0, 1330))


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
    if scene == "letter":
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
STYLE = (", cinematic horror film still, dark moody low-key lighting, desaturated, film grain, "
         "35mm photo, highly detailed, vertical composition, no text, no watermark")


def fetch_image(prompt, seed, path, tries=4):
    """Pollinations(무료, 키 불필요)로 세로 이미지를 받아 path에 저장. 이미 있으면 재사용."""
    import time
    import urllib.parse
    if path.exists() and path.stat().st_size > 10000:
        return path
    url = ("https://image.pollinations.ai/prompt/" + urllib.parse.quote(prompt + STYLE)
           + f"?width=1080&height=1920&nologo=true&private=true&seed={seed}&model=flux")
    for k in range(tries):
        r = subprocess.run(["curl", "-sS", "-m", "180", "-o", str(path), "-w", "%{http_code}", url],
                           capture_output=True, text=True)
        if r.stdout.strip() == "200" and path.exists() and path.stat().st_size > 10000:
            return path
        time.sleep(4 * (k + 1))
    raise RuntimeError(f"이미지 생성 실패: {prompt[:40]}... ({r.stdout} {r.stderr[:200]})")


def image_scene(path, args):
    """생성 이미지 -> 1080x1920 배경. 워터마크(우하단) 잘라내고 어둡게 색보정."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    im = im.crop((0, 0, w, int(h * 0.93)))  # 하단 워터마크 제거
    w, h = im.size
    sc = max(W / w, H / h)
    im = im.resize((int(w * sc + 1), int(h * sc + 1)), Image.LANCZOS)
    w, h = im.size
    im = im.crop(((w - W) // 2, (h - H) // 2, (w - W) // 2 + W, (h - H) // 2 + H))
    a = np.asarray(im).astype(np.float32)
    gray = a.mean(axis=2, keepdims=True)
    a = gray + (a - gray) * 0.7                                  # 채도 낮춤
    a = a * np.array([0.92, 0.95, 1.0], np.float32) * 0.82       # 차가운 톤 + 어둡게
    if args.get("grade") == "dark":
        # 밝게 나온 이미지를 공포 톤으로: 대비↑, 노출↓, 중앙 스포트라이트
        a = np.clip((a - 128) * 1.25 + 108, 0, 255) * 0.72
        yy2, xx2 = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx2 - W / 2) / (W * 0.55)) ** 2 + ((yy2 - H * 0.42) / (H * 0.42)) ** 2)
        a *= (1 - 0.85 * np.clip(r - 0.35, 0, 1))[..., None]
    # 자막 영역 가독성을 위한 하단 그라데이션
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    a *= 1 - 0.55 * np.clip((yy - 0.62) / 0.3, 0, 1)
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


# ---------------------------------------------------------------- audio design
def make_bed(total, hits, glitch_spans):
    n = int(total * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(3)
    drone = (0.5 * np.sin(2 * np.pi * 41.2 * t) + 0.35 * np.sin(2 * np.pi * 55.0 * t + 1)
             + 0.18 * np.sin(2 * np.pi * 82.6 * t)) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.07 * t))
    brown = np.cumsum(rng.normal(0, 1, n))
    brown -= np.convolve(brown, np.ones(2400) / 2400, mode="same")
    brown /= np.max(np.abs(brown)) + 1e-9
    bed = 0.10 * drone + 0.05 * brown
    # 고음 '이명' 레이어 (천천히 커졌다 작아짐)
    bed += 0.012 * np.sin(2 * np.pi * 1760 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.05 * t))
    for h in hits:
        i = int(h * SR)
        L = int(2.2 * SR)
        tt = np.arange(L) / SR
        boom = np.sin(2 * np.pi * (52 - 18 * tt) * tt) * np.exp(-tt * 2.4) * 0.55
        boom += rng.normal(0, 1, L) * np.exp(-tt * 18) * 0.12
        seg = bed[i:i + L]
        seg += boom[:len(seg)]
    for a, b in glitch_spans:
        i, j = int(a * SR), int(b * SR)
        for _ in range(8):
            s = rng.integers(i, max(i + 1, j - 2400))
            ln = rng.integers(600, 3000)
            bed[s:s + ln] += rng.normal(0, 0.18, len(bed[s:s + ln]))
    fade = int(0.6 * SR)
    bed[:fade] *= np.linspace(0, 1, fade)
    bed[-fade:] *= np.linspace(1, 0, fade)
    return bed


# ---------------------------------------------------------------- main
def main(story_path):
    story = json.loads(Path(story_path).read_text(encoding="utf-8"))
    sid = story["id"]
    work = ROOT / "build" / sid
    out_dir = ROOT / "output"
    work.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(exist_ok=True)

    # 1) TTS
    lines = story["lines"]
    voice_parts, t = [], LEAD_IN
    for i, ln in enumerate(lines):
        mp3 = work / f"line{i:02d}.mp3"
        if tts_backend(story) == "azure":
            azure_tts(ln["text"], story["voice"], story["rate"], story["pitch"], mp3)
            audio = decode(mp3)
            words = estimate_words(ln["text"], audio)
        else:
            words = asyncio.run(_tts(ln["text"], story["voice"], story["rate"], story["pitch"], mp3))
            audio = decode(mp3)
        dur = len(audio) / SR
        ln["_start"], ln["_dur"], ln["_audio"] = t, dur, audio
        ln["_chunks"] = [(t + s, txt, off) for s, txt, off in chunk_words(ln["text"], words)]
        ln["_mask"] = highlight_mask(ln["text"], ln.get("hl", []))
        t += dur + GAP
        print(f"  TTS {i + 1}/{len(lines)} {dur:.1f}s  {ln['text']}")
    total = t - GAP + TAIL
    for i, ln in enumerate(lines):
        if ln.get("img"):
            ln["_img"] = fetch_image(ln["img"], ln.get("seed", 1000 + i), work / f"img{i:02d}.jpg")
            print(f"  이미지 {i + 1}/{len(lines)} OK")
    print(f"총 길이 {total:.1f}s")

    voice = np.zeros(int(total * SR) + SR, np.float32)
    for ln in lines:
        i = int(ln["_start"] * SR)
        voice[i:i + len(ln["_audio"])] += ln["_audio"]
    write_wav(work / "voice.wav", voice[:int(total * SR)])

    hits = [ln["_start"] - 0.05 for k, ln in enumerate(lines)
            if k == 0 or ln["scene"] != lines[k - 1]["scene"] or ln.get("args", {}).get("year")]
    glitch_spans = [(ln["_start"], ln["_start"] + ln["_dur"]) for ln in lines if ln["scene"] == "glitch"]
    write_wav(work / "bed.wav", make_bed(total, hits, glitch_spans))

    # 2) 영상 프레임
    rng = np.random.default_rng(11)
    rnd = random.Random(5)
    grains = [rng.integers(-10, 10, (H, W, 1), dtype=np.int16) for _ in range(6)]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    vign = (1 - 0.75 * np.clip(((xx - W / 2) / (W * 0.75)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2, 0, 1))[..., None]
    cache = {}

    nframes = int(total * FPS)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-maxrate", "8M", "-bufsize", "16M", "-pix_fmt", "yuv420p",
                            str(work / "video.mp4")], stdin=subprocess.PIPE)
    f_small = font(34)
    f_tag = font(40)
    for fi in range(nframes):
        ft = fi / FPS
        li = max(k for k, ln in enumerate(lines) if k == 0 or ln["_start"] - GAP / 2 <= ft)
        ln = lines[li]
        seg_start = 0 if li == 0 else ln["_start"] - GAP / 2
        seg_end = lines[li + 1]["_start"] - GAP / 2 if li + 1 < len(lines) else total
        lt, ldur = ft - seg_start, seg_end - seg_start
        if ln.get("_img"):
            key = ("img", str(ln["_img"]), json.dumps(ln.get("args", {}), sort_keys=True))
        else:
            key = (ln["scene"], json.dumps(ln.get("args", {}), sort_keys=True))
        if key not in cache:
            if ln.get("_img"):
                cache[key] = image_scene(ln["_img"], ln.get("args", {}))
            else:
                cache[key] = scene_image(ln["scene"], ln.get("args", {}), rnd)
        base = cache[key]

        # 느린 줌 (켄 번스)
        p = min(1.0, lt / max(ldur, 0.1))
        z = 1.0 + 0.07 * p if li % 2 == 0 else 1.07 - 0.07 * p
        if ln["scene"] == "eyes":
            z = 1.0 + 0.16 * p
        cw, ch = W / z, H / z
        dx = (li % 3 - 1) * 20 * p
        x0 = min(max(0.0, (W - cw) / 2 + dx), W - cw)
        box = (x0, (H - ch) / 2, x0 + cw, (H + ch) / 2)
        frame = base.resize((W, H), Image.BILINEAR, box=box).convert("RGBA")
        scene_overlay(frame, ln["scene"], ln.get("args", {}), lt, ldur, ln["text"])

        # 상단 시리즈 태그 + 고지
        d = ImageDraw.Draw(frame)
        d.rounded_rectangle([60, 110, 460, 178], radius=14, fill=(150, 20, 20, 230))
        d.text((260, 144), "실화 괴담 · 미국", font=f_tag, fill=(255, 255, 255), anchor="mm")
        d.text((540, 1780), story["disclaimer"], font=f_small, fill=(170, 170, 170, 200), anchor="mm")
        if ln.get("_img"):
            d.text((1020, 144), "AI 재연 이미지", font=f_small, fill=(200, 200, 200, 210), anchor="rm",
                   stroke_width=2, stroke_fill=(0, 0, 0))

        # 자막
        cur = None
        for cstart, txt, off in ln["_chunks"]:
            if ft >= cstart - 0.03:
                cur = (cstart, txt, off)
        if cur and ft <= ln["_start"] + ln["_dur"] + 0.25:
            pop = min(1.0, (ft - cur[0]) / 0.13)
            draw_caption(frame, cur[1], cur[2], ln["_mask"], max(0, pop))

        arr = np.asarray(frame.convert("RGB")).astype(np.int16)
        # 깜빡임 + 장면 전환 페이드
        flick = 1.0 - 0.06 * rng.random() - (0.35 if rng.random() < 0.012 else 0)
        if ln["scene"] in ("doll_dark",) and rng.random() < 0.04:
            flick *= 0.35
        fade_in = min(1.0, lt / 0.18) if (li == 0 or ln["scene"] != lines[li - 1]["scene"]) else 1.0
        arr = arr + grains[fi % len(grains)]
        arr = (arr * vign * flick * fade_in)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        if ln["scene"] == "glitch" and rng.random() < 0.45:
            arr = apply_glitch(arr, rng, rng.uniform(0.4, 1.0))
        elif rng.random() < 0.01:
            arr = apply_glitch(arr, rng, 0.3)
        enc.stdin.write(arr.tobytes())
        if fi % 150 == 0:
            print(f"  프레임 {fi}/{nframes}")
    enc.stdin.close()
    enc.wait()

    # 3) 오디오 믹스 + 합치기
    out = out_dir / f"{sid}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(work / "video.mp4"),
                    "-i", str(work / "voice.wav"), "-i", str(work / "bed.wav"),
                    "-filter_complex",
                    "[1:a]highpass=f=70,aecho=0.8:0.55:70|140:0.22|0.12,volume=1.6[v];"
                    "[2:a]volume=0.9[b];[v][b]amix=inputs=2:normalize=0,"
                    "loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[a]",
                    "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", "-movflags", "+faststart", str(out)], check=True)
    print(f"완료: {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "stories" / "robert_the_doll.json"))
