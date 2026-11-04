# creepy_shorts — 실화 괴담 쇼츠 생성기

대본(JSON) 하나로 1080×1920 쇼츠를 만든다: 한국어 TTS(Edge TTS) + 단어 단위 자막 + 장면 그래픽 + 효과음/드론.

```bash
pip install -r requirements.txt
python3 make_short.py stories/robert_the_doll.json   # -> output/robert_the_doll.mp4
```

- `research_top10.md` : 전세계 실화 괴담 소재 10선 (출처·신뢰도 포함)
- `stories/*.json`    : 대본. 줄마다 `text`, `scene`, `hl`(강조 단어), `args`
- 장면 종류: envelope, location, year, doll, doll_dark, door, case, camera, glitch, letter, wall, eyes
- 네트워크: `speech.platform.bing.com` 접근 필요 (TTS)
