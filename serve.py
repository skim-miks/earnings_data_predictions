"""Serve the app folder plus the Reddit endpoint the calendar's button calls.

Usage:
    uv run serve.py            # then open http://localhost:8765/calendar.html

    GET  /api/reddit/<TICKER>  # posts already stored, no call to Reddit
    POST /api/reddit/<TICKER>  # pull from Reddit now, store, and return
"""

import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

import reddit
from collect_data import load_universe

PORT = 8765
APP_DIR = Path(__file__).parent / "app"
COMPANIES = dict(zip(*[load_universe()[c] for c in ("ticker", "search_name")]))


class Handler(SimpleHTTPRequestHandler):
    def send_json(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def reddit_ticker(self) -> str | None:
        prefix = "/api/reddit/"
        if not self.path.startswith(prefix):
            return None
        ticker = self.path[len(prefix):].split("?")[0].upper()
        if ticker not in COMPANIES:
            self.send_json(404, {"error": f"{ticker} is not in the universe"})
            return None
        return ticker

    def do_GET(self) -> None:
        if not self.path.startswith("/api/"):
            return super().do_GET()
        ticker = self.reddit_ticker()
        if ticker:
            self.send_json(200, reddit.summarize(reddit.load(ticker)))

    def do_POST(self) -> None:
        ticker = self.reddit_ticker()
        if not ticker:
            if not self.path.startswith("/api/reddit/"):
                self.send_json(404, {"error": "not found"})
            return
        try:
            self.send_json(200, reddit.pull(ticker, COMPANIES[ticker]))
        except reddit.RedditNotConfigured as e:
            self.send_json(503, {"error": str(e)})
        except requests.HTTPError as e:
            self.send_json(502, {"error": f"Reddit returned {e.response.status_code}: {e.response.text[:200]}"})
        except requests.RequestException as e:
            self.send_json(502, {"error": f"Could not reach Reddit: {e}"})


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), partial(Handler, directory=str(APP_DIR)))
    print(f"http://localhost:{PORT}/calendar.html")
    server.serve_forever()
