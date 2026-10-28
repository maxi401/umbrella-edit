#!/usr/bin/env python3
"""Fashion archive server; hosted access is protected by the deployment proxy."""
from __future__ import annotations

import argparse
import base64
import csv
from datetime import date, datetime, timedelta
import hashlib
import json
import mimetypes
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("FASHION_DATA_DIR", ROOT)).expanduser().resolve()
DB = DATA / "fashion_archive.sqlite3"
CATALOG = DATA / "output" / "image_catalog.csv"
SOURCE = Path(os.environ.get("FASHION_SOURCE_DIR", Path.home() / "Desktop" / "fashionArchive")).expanduser().resolve()
STATIC = ROOT / "static"
SEGMENT_LOCK = threading.Lock()
SEGMENT_RUNNING = False


def segment_uploads():
    """Refresh new upload cutouts locally, merging into the existing results."""
    global SEGMENT_RUNNING
    SEGMENT_LOCK.acquire()
    SEGMENT_RUNNING = True
    try:
        with connect() as con:
            uploads = [dict(r) for r in con.execute("SELECT id, relative_path, record_type, source_title, source_slide_index FROM records WHERE relative_path LIKE 'uploads/%'")]
        if not uploads:
            return
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "uploads.csv"
            with catalog.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["image_id", "relative_path", "record_type", "source_title"])
                writer.writeheader()
                writer.writerows({"image_id": r["id"], "relative_path": r["relative_path"],
                    "record_type": r["record_type"], "source_title": r["source_title"]} for r in uploads)
            python = Path("/opt/anaconda3/bin/python")
            executable = str(python) if python.exists() else sys.executable
            subprocess.run([executable, str(ROOT / "segment_archive.py"), "--source", DATA,
                "--catalog", str(catalog), "--output", str(DATA / "output" / "segmentation"), "--incremental"], cwd=ROOT, check=True,
                env={**os.environ, "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1")})
    except Exception as exc:
        print(f"[fashion-archive] segmentation worker failed: {exc}")
    finally:
        SEGMENT_RUNNING = False
        SEGMENT_LOCK.release()


def connect():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with connect() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS records (
            id TEXT PRIMARY KEY, relative_path TEXT NOT NULL, record_type TEXT NOT NULL,
            source_title TEXT DEFAULT '', source_slide_index TEXT DEFAULT '', brand TEXT DEFAULT '',
            price TEXT DEFAULT '', purchase_date TEXT DEFAULT '', location TEXT DEFAULT '',
            note TEXT DEFAULT '', owned_status TEXT DEFAULT 'unknown', worn_count INTEGER DEFAULT 0,
            last_worn TEXT DEFAULT '', tags TEXT DEFAULT '', human_verified INTEGER DEFAULT 0
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS outfit_logs (
            id TEXT PRIMARY KEY, record_id TEXT NOT NULL, worn_on TEXT NOT NULL, note TEXT DEFAULT ''
        )""")
        if CATALOG.exists():
            with CATALOG.open(encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    con.execute("""INSERT OR IGNORE INTO records
                        (id, relative_path, record_type, source_title, source_slide_index)
                        VALUES (?, ?, ?, ?, ?)""",
                        (row["image_id"], row["relative_path"], row["record_type"],
                         row.get("source_title", ""), row.get("source_slide_index", "")))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("[fashion-archive] " + fmt % args)

    def send_bytes(self, body: bytes, content_type="application/json; charset=utf-8", status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, obj, status=200):
        self.send_bytes(json.dumps(obj, ensure_ascii=False).encode("utf-8"), status=status)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/records":
            with connect() as con:
                rows = [dict(r) for r in con.execute("SELECT * FROM records ORDER BY source_title, source_slide_index, id")]
            return self.send_json(rows)
        if parsed.path == "/api/outfits":
            with connect() as con:
                rows = [dict(r) for r in con.execute("""SELECT o.*, r.source_title FROM outfit_logs o
                    JOIN records r ON r.id=o.record_id ORDER BY worn_on DESC""")]
            return self.send_json(rows)
        if parsed.path == "/api/today":
            today = date.today()
            week_start = (today - timedelta(days=6)).isoformat()
            with connect() as con:
                outfits = [dict(r) for r in con.execute("""SELECT * FROM records WHERE record_type='ootd'
                    ORDER BY source_title, id""")]
                totals = {r["record_type"]: r["n"] for r in con.execute("SELECT record_type, COUNT(*) n FROM records GROUP BY record_type")}
                total_logs = con.execute("SELECT COUNT(*) FROM outfit_logs").fetchone()[0]
                week_logs = con.execute("SELECT COUNT(*) FROM outfit_logs WHERE worn_on>=?", (week_start,)).fetchone()[0]
                favorite = con.execute("""SELECT r.source_title, COUNT(*) n FROM outfit_logs o JOIN records r ON r.id=o.record_id
                    WHERE o.worn_on>=? GROUP BY r.id ORDER BY n DESC, r.source_title LIMIT 1""", (week_start,)).fetchone()
            # A transparent rotation heuristic: minimize recent wear; deterministic daily tie-break.
            # Duplicate post slides are collapsed to one candidate per source title.
            unique = {}
            for r in outfits:
                key = r.get("source_title") or r["id"]
                prior = unique.get(key)
                if prior is None or int(r.get("source_slide_index") or 9999) < int(prior.get("source_slide_index") or 9999):
                    unique[key] = r
            candidates = list(unique.values())
            titled = [r for r in candidates if not str(r.get("source_title", "")).startswith("Screenshot ")]
            if titled:
                candidates = titled
            def score(r):
                last = r.get("last_worn") or ""
                days_since = 9999
                if last:
                    try: days_since = (today - date.fromisoformat(last)).days
                    except ValueError: pass
                recent_penalty = max(0, 7 - days_since) * 10
                return int(r.get("worn_count") or 0) * 100 + recent_penalty
            selected = None
            if candidates:
                best_score = min(score(r) for r in candidates)
                tied = sorted((r for r in candidates if score(r) == best_score), key=lambda r: r["id"])
                index = int(hashlib.sha1(today.isoformat().encode()).hexdigest()[:8], 16) % len(tied)
                selected = tied[index]
            return self.send_json({"date": today.isoformat(), "suggestion": selected,
                "counts": {"outfits": totals.get("ootd", 0), "closet_candidates": totals.get("owned_candidate", 0),
                    "interests": totals.get("interest_candidate", 0), "inspiration": totals.get("inspiration", 0),
                    "logged_wears": total_logs, "week_wears": week_logs},
                "weekly_favorite": dict(favorite) if favorite else None,
                "reason": "Selected from the least-worn outfit records, avoiding looks logged in the past week." if total_logs else "Wear history is empty, so this is a rotating pick from the saved OOTD archive. Confirm that the piece is still in her wardrobe before treating it as a wear suggestion."})
        if parsed.path == "/api/segments":
            result_path = DATA / "output" / "segmentation" / "segmentation_results.json"
            if not result_path.is_file():
                return self.send_json({"ready": False, "results": []})
            data = json.loads(result_path.read_text(encoding="utf-8"))
            data["ready"] = True
            data["processing"] = SEGMENT_RUNNING
            return self.send_json(data)
        if parsed.path.startswith("/segmentation/"):
            relative = Path(unquote(parsed.path.removeprefix("/segmentation/")))
            base = (DATA / "output" / "segmentation").resolve()
            path = (base / relative).resolve()
            if base not in path.parents or not path.is_file():
                return self.send_bytes(b"Not found", "text/plain", 404)
            return self.send_bytes(path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        if parsed.path.startswith("/images/"):
            image_id = unquote(parsed.path.removeprefix("/images/"))
            with connect() as con:
                row = con.execute("SELECT relative_path FROM records WHERE id=?", (image_id,)).fetchone()
            if not row:
                return self.send_bytes(b"Not found", "text/plain", 404)
            relative = Path(row["relative_path"])
            base = DATA if relative.parts and relative.parts[0] == "uploads" else SOURCE
            path = (base / relative).resolve()
            if base.resolve() not in path.parents or not path.is_file():
                return self.send_bytes(b"Image unavailable", "text/plain", 404)
            return self.send_bytes(path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        if parsed.path == "/" or parsed.path == "/index.html":
            return self.serve_static("index.html")
        if parsed.path.startswith("/static/"):
            return self.serve_static(unquote(parsed.path.removeprefix("/static/")))
        return self.send_bytes(b"Not found", "text/plain", 404)

    def serve_static(self, name):
        path = (STATIC / name).resolve()
        if STATIC.resolve() not in path.parents or not path.is_file():
            return self.send_bytes(b"Not found", "text/plain", 404)
        return self.send_bytes(path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream")

    def check_write_request(self):
        if os.environ.get("FASHION_HOSTED") == "1":
            origin = self.headers.get("Origin", "")
            expected = os.environ.get("RENDER_EXTERNAL_HOSTNAME", "")
            if not expected or origin != f"https://{expected}":
                self.send_json({"error": "Invalid request origin"}, 403)
                return False
        try:
            size = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            size = -1
        if size < 0 or size > 20 * 1024 * 1024:
            self.send_json({"error": "Request is too large or has no valid size"}, 413)
            return False
        return True

    def do_PATCH(self):
        if not self.check_write_request():
            return
        if not self.path.startswith("/api/records/"):
            return self.send_json({"error": "not found"}, 404)
        record_id = unquote(self.path.removeprefix("/api/records/"))
        data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
        allowed = {"record_type", "brand", "price", "purchase_date", "location", "note", "owned_status", "worn_count", "last_worn", "tags", "human_verified"}
        fields = [(k, v) for k, v in data.items() if k in allowed]
        if not fields:
            return self.send_json({"error": "no editable fields"}, 400)
        clause = ", ".join(f"{k}=?" for k, _ in fields)
        with connect() as con:
            cur = con.execute(f"UPDATE records SET {clause} WHERE id=?", [v for _, v in fields] + [record_id])
        return self.send_json({"updated": cur.rowcount == 1})

    def do_POST(self):
        if not self.check_write_request():
            return
        if self.path == "/api/ootd":
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
            record_id = data.get("id")
            with connect() as con:
                con.execute("INSERT INTO outfit_logs (id, record_id, worn_on, note) VALUES (?, ?, ?, ?)",
                    (str(uuid.uuid4()), record_id, data.get("date", ""), data.get("note", "")))
                con.execute("UPDATE records SET worn_count=worn_count+1, last_worn=? WHERE id=?",
                    (data.get("date", ""), record_id))
            return self.send_json({"saved": True})
        if self.path == "/api/upload":
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
            name = Path(data.get("name", "photo.jpg")).name
            ext = Path(name).suffix.lower()
            if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                return self.send_json({"error": "Use a JPG, PNG, or WebP image"}, 400)
            upload_dir = DATA / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            record_id = str(uuid.uuid4())
            relative = f"uploads/{record_id}{ext}"
            try:
                upload_dir.joinpath(f"{record_id}{ext}").write_bytes(base64.b64decode(data["data"], validate=True))
            except Exception as exc:
                return self.send_json({"error": f"Could not save image: {exc}"}, 400)
            with connect() as con:
                con.execute("""INSERT INTO records (id, relative_path, record_type, source_title, owned_status, human_verified)
                    VALUES (?, ?, ?, ?, 'unknown', 1)""", (record_id, relative, data.get("record_type", "ootd"), data.get("title") or "New photo"))
            threading.Thread(target=segment_uploads, daemon=True).start()
            return self.send_json({"saved": True, "segmentation_queued": True, "image_id": record_id})
        return self.send_json({"error": "not found"}, 404)


def main():
    global SOURCE
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address; use 0.0.0.0 to allow devices on your local Wi-Fi")
    args = parser.parse_args()
    SOURCE = args.source.expanduser().resolve()
    init_db()
    # Process app uploads that may have been added while the app was closed.
    with connect() as con:
        has_uploads = con.execute("SELECT 1 FROM records WHERE relative_path LIKE 'uploads/%' LIMIT 1").fetchone()
    if has_uploads:
        threading.Thread(target=segment_uploads, daemon=True).start()
    print(f"Fashion Archive is listening on {args.host}:{args.port}")
    print(f"Local image folder: {SOURCE}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
