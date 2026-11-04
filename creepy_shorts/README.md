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

## 음성 엔진 선택
| | Edge TTS (기본) | Azure Speech (권장: 수익화 채널) |
|---|---|---|
| 비용 | 무료 | 월 50만 자 무료(F0), 이후 유료 |
| 상업 이용 | 불분명(비공식) | 허용(정식 서비스) |
| 자막 타이밍 | 단어 단위 실측 | 발화 구간 기준 추정 |

```bash
TTS_BACKEND=azure AZURE_SPEECH_REGION=koreacentral python3 make_short.py stories/robert_the_doll.json
```
스토리 JSON에 `"tts": "azure"`를 넣어도 된다. 키는 클라우드 환경의 네트워크 시크릿(헤더 `Ocp-Apim-Subscription-Key`)으로 주입하거나, 로컬에서는 `AZURE_SPEECH_KEY` 환경변수로 넣는다.
