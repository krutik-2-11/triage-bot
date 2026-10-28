"""Fetch duplicate issue pairs and distractor issues from ollama/ollama.

Usage: uv run --env-file .env python fetch_issues.py
"""
import json
import os
import re
import time
from pathlib import Path

import requests

REPO = "ollama/ollama"
API = "https://api.github.com"
DATA = Path("data")
MAX_PAIRS = 60          # stop once we have this many pairs
N_DISTRACTORS = 1000    # normal issues added as wrong answers
MAX_PAGES = 30          # issues list includes PRs, so we may need many pages
DUP_RE = re.compile(r"duplicate of\s+#(\d+)", re.IGNORECASE)

session = requests.Session()
session.headers.update({
    "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
})


def get(url, params=None):
    """GET a GitHub API URL, waiting if we hit the rate limit."""
    r = session.get(url, params=params, timeout=30)
    if r.status_code in (403, 429) and r.headers.get("X-RateLimit-Remaining") == "0":
        wait = int(r.headers["X-RateLimit-Reset"]) - int(time.time()) + 1
        print(f"Rate limited, waiting {wait}s...")
        time.sleep(max(wait, 1))
        return get(url, params)
    r.raise_for_status()
    return r.json()


def slim(issue):
    """Keep only the fields we need. No comments, so no leaked answers."""
    return {
        "number": issue["number"],
        "title": issue["title"],
        "body": (issue.get("body") or "")[:4000],
        "created_at": issue["created_at"],
        "url": issue["html_url"],
    }


def search_candidates():
    """Closed issues whose text mentions 'duplicate of'."""
    q = f'repo:{REPO} is:issue is:closed "duplicate of"'
    items = []
    for page in range(1, 6):  # up to 500 results
        res = get(f"{API}/search/issues", {"q": q, "per_page": 100, "page": page})
        items += res["items"]
        if len(res["items"]) < 100:
            break
        time.sleep(2)  # be gentle with the search API
    return items


def find_original(number):
    """Look in an issue's comments for 'Duplicate of #N'. Return N or None."""
    comments = get(f"{API}/repos/{REPO}/issues/{number}/comments", {"per_page": 100})
    for c in comments:
        m = DUP_RE.search(c.get("body") or "")
        if m and int(m.group(1)) != number:
            return int(m.group(1))
    return None


def main():
    DATA.mkdir(exist_ok=True)
    issues, pairs = {}, []

    candidates = search_candidates()
    # Fix 1: any issue that mentions "duplicate of" is probably a duplicate
    # of something, so it must never be used as a "wrong answer".
    candidate_nums = {c["number"] for c in candidates}
    print(f"Found {len(candidates)} candidate issues")

    for cand in candidates:
        if len(pairs) >= MAX_PAIRS:
            break
        orig_num = find_original(cand["number"])
        if orig_num is None:
            continue
        try:
            orig = get(f"{API}/repos/{REPO}/issues/{orig_num}")
        except requests.HTTPError:
            continue  # deleted or moved
        if "pull_request" in orig:
            continue  # points to a PR, not an issue
        if orig["created_at"] >= cand["created_at"]:
            continue  # original must be older
        issues[cand["number"]] = slim(cand)
        issues[orig_num] = slim(orig)
        pairs.append({"duplicate": cand["number"], "original": orig_num})
        print(f"  pair {len(pairs)}: #{cand['number']} -> #{orig_num}")

    # Distractors: closed issues that aren't pairs and aren't likely duplicates
    added, skipped, page = 0, 0, 1
    while added < N_DISTRACTORS and page <= MAX_PAGES:
        batch = get(f"{API}/repos/{REPO}/issues",
                    {"state": "closed", "per_page": 100, "page": page})
        if not batch:
            break
        for it in batch:
            if "pull_request" in it or it["number"] in issues:
                continue
            if it["number"] in candidate_nums:
                skipped += 1
                continue
            issues[it["number"]] = slim(it)
            added += 1
            if added >= N_DISTRACTORS:
                break
        page += 1
        print(f"  distractors so far: {added}")

    (DATA / "pairs.json").write_text(json.dumps(pairs, indent=2))
    (DATA / "issues.json").write_text(json.dumps(list(issues.values()), indent=2))
    print(f"\nSkipped {skipped} likely duplicates from distractors")
    print(f"Saved {len(pairs)} pairs and {len(issues)} issues to data/")


if __name__ == "__main__":
    main()