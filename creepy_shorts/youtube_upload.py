#!/usr/bin/env python3
"""YouTube Data API v3 업로드 (resumable).

python3 youtube_upload.py stories/<id>.youtube.json
- 토큰: 클라우드 환경에서는 네트워크 프록시가 www.googleapis.com 요청에 Authorization 헤더를 주입한다.
  로컬에서는 YT_ACCESS_TOKEN 환경변수에 OAuth 액세스 토큰(youtube.upload 권한)을 넣는다.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def upload(meta_path):
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    video = ROOT / meta["file"]
    size = video.stat().st_size
    token = os.environ.get("YT_ACCESS_TOKEN", "injected-by-proxy")
    body = {
        "snippet": {"title": meta["title"][:100], "description": meta["description"][:5000],
                    "tags": meta.get("tags", []), "categoryId": str(meta.get("categoryId", 24)),
                    "defaultLanguage": "ko", "defaultAudioLanguage": "ko"},
        "status": {"privacyStatus": meta.get("privacy", "private"), "selfDeclaredMadeForKids": False,
                   "containsSyntheticMedia": bool(meta.get("synthetic", True)), "embeddable": True},
    }
    url = ("https://www.googleapis.com/upload/youtube/v3/videos"
           "?uploadType=resumable&part=snippet,status")
    r = subprocess.run(["curl", "-sS", "-m", "60", "-D", "-", "-o", "/dev/null", "-X", "POST", url,
                        "-H", f"Authorization: Bearer {token}",
                        "-H", "Content-Type: application/json; charset=UTF-8",
                        "-H", "X-Upload-Content-Type: video/mp4",
                        "-H", f"X-Upload-Content-Length: {size}",
                        "--data-binary", json.dumps(body, ensure_ascii=False).encode("utf-8")],
                       capture_output=True, text=True)
    status = r.stdout.split("\n", 1)[0]
    loc = [l.split(":", 1)[1].strip() for l in r.stdout.splitlines() if l.lower().startswith("location:")]
    if not loc:
        err = subprocess.run(["curl", "-sS", "-m", "60", "-X", "POST", url,
                              "-H", f"Authorization: Bearer {token}",
                              "-H", "Content-Type: application/json; charset=UTF-8",
                              "--data-binary", json.dumps(body, ensure_ascii=False).encode("utf-8")],
                             capture_output=True, text=True).stdout
        raise SystemExit(f"업로드 세션 시작 실패: {status}\n{err[:800]}")
    r2 = subprocess.run(["curl", "-sS", "-m", "1800", "-X", "PUT", loc[0],
                         "-H", f"Authorization: Bearer {token}",
                         "-H", "Content-Type: video/mp4", "--data-binary", f"@{video}"],
                        capture_output=True, text=True)
    try:
        res = json.loads(r2.stdout)
    except Exception:
        raise SystemExit(f"업로드 실패: {r2.stdout[:800]} {r2.stderr[:300]}")
    if "id" not in res:
        raise SystemExit(f"업로드 실패: {json.dumps(res, ensure_ascii=False)[:800]}")
    print(f"업로드 완료: https://youtube.com/shorts/{res['id']}  ({res['status'].get('privacyStatus')})")
    return res["id"]


if __name__ == "__main__":
    for p in sys.argv[1:]:
        upload(p)
