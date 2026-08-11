"""
fetch_arxiv.py

Fetches recent arXiv papers for the people listed in _data/people.yml and
writes them to _data/papers.yml for Jekyll to render.

Run from the project root:
    python3 scripts/fetch_arxiv.py

Check whether a candidate arXiv author ID belongs to the right person:
    python3 scripts/fetch_arxiv.py --check ding_p_1

Requirements:
    python3 -m pip install requests pyyaml

How people are matched
----------------------
Two lookup modes, per person, in _data/people.yml:

  1. `arxiv_author_id` (preferred, unambiguous) -- the arXiv author identifier,
     e.g. `feller_a_1`. Fetches that author's own feed, so no name collisions.
     Only lists papers the author has claimed on arXiv, so it can miss papers
     they never linked to their account.

  2. `display_name` / `arxiv_name` (fallback) -- a plain author-name search.
     Complete, but common names pull in strangers.

Set `arxiv: false` on an entry to skip a person entirely. Only people whose
`role` is in ARXIV_ROLES are searched (alumni and past visitors are excluded).

Adapted from the Berkeley probability group's Quarto site:
https://github.com/berkeley-probability/berkeley-probability.github.io
"""

import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import sleep
from urllib.parse import urlparse

import requests
import yaml

# ── Settings ───────────────────────────────────────────────────────────────────
PEOPLE_FILE = Path("_data/people.yml")
OUTPUT_FILE = Path("_data/papers.yml")
ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_AUTHOR_URL = "https://arxiv.org/a/{author_id}.atom"
ARXIV_ROLES = {"faculty", "postdoc", "grad"}
# Papers found by NAME must match one of these arXiv categories, which filters
# out same-name strangers. An entry is either a full category ("math.ST") or a
# whole archive ("stat" matches stat.ME, stat.ML, ...). Papers found by author
# ID skip this check -- the ID already identifies the person unambiguously.
ARXIV_CATEGORIES = {"stat", "econ", "math.ST", "math.PR", "cs.CY"}
# Papers to exclude by hand, e.g. a same-name stranger's paper that slips past
# the category filter. Use the bare arXiv ID without the version suffix; a
# short note on who/why keeps this list maintainable.
ARXIV_BLOCKLIST = {
    "2602.04456",  # "Growth First, Care Second?" -- a different Xinyi Wang
    "2510.01560",  # "AI Foundation Model for Time Series" -- a different Xinyi Wang
}
MAX_RESULTS = 300
WINDOW_DAYS = 365          # how far back to keep papers
MAX_AUTHORS_SHOWN = 8      # longer author lists get "et al."
REQUEST_TIMEOUT = 90
REQUEST_RETRIES = 4
REQUEST_SPACING = 3        # seconds between requests, to stay polite
# arXiv asks API clients to identify themselves and to back off when throttled.
USER_AGENT = "casual-causal-site/1.0 (+https://causal.berkeley.edu)"
ATOM = "http://www.w3.org/2005/Atom"
# ───────────────────────────────────────────────────────────────────────────────


def clean_text(value):
    """Normalize whitespace from arXiv's Atom fields."""
    return " ".join(value.split())


def normalize_name(name):
    """Loose key for comparing author names ("Chase H. Mathis" -> "chase h mathis")."""
    return " ".join(name.lower().replace(".", " ").split())


def group_member_names():
    """Every name in people.yml, for highlighting group authors in long lists.

    Uses all roles, not just ARXIV_ROLES -- if an alum co-authored a paper, we
    still want them visible rather than buried behind "et al.".
    """
    people = yaml.safe_load(PEOPLE_FILE.read_text()) or {}
    names = {}
    for entry in people.values():
        if not isinstance(entry, dict):
            continue
        name = entry.get("arxiv_name") or entry.get("display_name")
        if name:
            names[normalize_name(name)] = name
    return names


def load_roster():
    """Return (author_ids, names) for the people we should search for."""
    people = yaml.safe_load(PEOPLE_FILE.read_text()) or {}
    author_ids, names = [], []

    for entry in people.values():
        if not isinstance(entry, dict) or entry.get("arxiv") is False:
            continue
        if entry.get("role") not in ARXIV_ROLES:
            continue
        author_id = entry.get("arxiv_author_id")
        if author_id:
            author_ids.append(author_id)
            continue
        name = entry.get("arxiv_name") or entry.get("display_name")
        if name:
            names.append(name)

    return author_ids, names


def fetch(url, params=None):
    """Fetch a URL, retrying with backoff when arXiv throttles or stalls."""
    last_error = None

    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if attempt == REQUEST_RETRIES:
                break
            wait_seconds = 15 * attempt  # arXiv throttles bursts with 429/503
            print(
                f"  request failed ({exc.__class__.__name__}); "
                f"retry {attempt}/{REQUEST_RETRIES - 1} in {wait_seconds}s..."
            )
            sleep(wait_seconds)

    raise RuntimeError(
        f"Could not fetch {url} after {REQUEST_RETRIES} attempts. "
        f"Last error: {last_error}"
    ) from last_error


def format_authors(authors, members):
    """Format an author list, truncating very long collaborations.

    Group members past the cutoff are pulled forward so a 30-author paper still
    shows the people who actually belong to this group, e.g.
    "A, B, ..., H, Amanda Coston, Ezinne Nwankwo, Avi Feller, et al."
    """
    if len(authors) <= MAX_AUTHORS_SHOWN:
        return ", ".join(authors)

    shown = authors[:MAX_AUTHORS_SHOWN]
    rest = authors[MAX_AUTHORS_SHOWN:]
    ours = [a for a in rest if normalize_name(a) in members]
    listed = shown + ours
    # Only claim "et al." if names were actually left out -- pulling group
    # members forward can account for every remaining author.
    suffix = ", et al." if len(listed) < len(authors) else ""
    return ", ".join(listed) + suffix


def parse_entries(source, members=None):
    """Turn an arXiv Atom feed into a list of paper dicts.

    Accepts raw XML text or an already-parsed root element. Handles both the
    API response and the per-author `/a/<id>.atom` feed -- they share the same
    entry schema.
    """
    root = ET.fromstring(source) if isinstance(source, str) else source
    members = members or {}
    papers = []

    for entry in root.findall(f"{{{ATOM}}}entry"):
        url = entry.findtext(f"{{{ATOM}}}id", "").strip()
        # Drop the version suffix (v1, v2, ...) so IDs and links stay stable.
        arxiv_id = base_id(urlparse(url).path.removeprefix("/abs/").strip("/"))
        published = datetime.fromisoformat(
            entry.findtext(f"{{{ATOM}}}published", "").replace("Z", "+00:00")
        )
        authors = [
            clean_text(a.findtext(f"{{{ATOM}}}name", ""))
            for a in entry.findall(f"{{{ATOM}}}author")
        ]
        categories = [
            c.attrib["term"]
            for c in entry.findall(f"{{{ATOM}}}category")
            if c.attrib.get("term")
        ]
        papers.append({
            "title": clean_text(entry.findtext(f"{{{ATOM}}}title", "")),
            "arxiv_id": arxiv_id,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
            "authors": format_authors(authors, members),
            "date": published.strftime("%Y-%m-%d"),
            "date_display": published.strftime("%b %d, %Y"),
            "categories": categories,
            "_published": published,
        })

    return papers


def in_scope(paper):
    """True if a paper sits in one of the group's arXiv categories."""
    return any(
        category in ARXIV_CATEGORIES or category.split(".")[0] in ARXIV_CATEGORIES
        for category in paper["categories"]
    )


def base_id(arxiv_id):
    """Strip the version suffix so v1 and v3 of a paper dedupe together."""
    return re.sub(r"v\d+$", "", arxiv_id)


def collect_papers(author_ids, names, cutoff, members):
    """Fetch from every author feed plus one batched name query, deduped.

    `cutoff` is only used to tell whether the name search's result cap
    truncated anything we would actually have kept.
    """
    papers = []

    for author_id in author_ids:
        print(f"  author feed: {author_id}")
        papers += parse_entries(fetch(ARXIV_AUTHOR_URL.format(author_id=author_id)).text, members)
        sleep(REQUEST_SPACING)

    if names:
        print(f"  name search: {len(names)} people")
        query = " OR ".join(f'au:"{name}"' for name in names)
        response = fetch(ARXIV_API_URL, params={
            "search_query": query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": MAX_RESULTS,
        })
        found = parse_entries(response.text, members)
        # Results come back newest-first, so the cap only loses papers we care
        # about if even the oldest one is still inside the window.
        if len(found) == MAX_RESULTS and min(p["_published"] for p in found) >= cutoff:
            print(
                f"  WARNING: name search hit the {MAX_RESULTS}-result cap and "
                f"every result is inside the {WINDOW_DAYS}-day window, so older "
                "in-window papers were truncated. Raise MAX_RESULTS."
            )
        in_field = [p for p in found if in_scope(p)]
        print(
            f"  kept {len(in_field)}/{len(found)} name matches "
            f"in {'/'.join(sorted(ARXIV_CATEGORIES))}"
        )
        papers += in_field

    # Dedupe by paper (ignoring version), keeping the newest record of each,
    # and drop anything explicitly blocklisted.
    best = {}
    blocked = 0
    for paper in papers:
        key = base_id(paper["arxiv_id"])
        if key in ARXIV_BLOCKLIST:
            blocked += 1
            continue
        if key not in best or paper["_published"] > best[key]["_published"]:
            best[key] = paper
    if blocked:
        print(f"  dropped {blocked} blocklisted paper(s)")

    return sorted(best.values(), key=lambda p: p["_published"], reverse=True)


def check_author_id(author_id):
    """Print a candidate author feed so a human can confirm it's the right person."""
    root = ET.fromstring(fetch(ARXIV_AUTHOR_URL.format(author_id=author_id)).text)
    print(f"Feed title: {clean_text(root.findtext(f'{{{ATOM}}}title', ''))}")
    for link in root.findall(f"{{{ATOM}}}link"):
        if link.attrib.get("rel") == "describes":
            print(f"Linked profile: {link.attrib.get('href')}")

    entries = parse_entries(root)
    print(f"{len(entries)} papers listed. Most recent:")
    for paper in entries[:8]:
        print(f"  [{', '.join(paper['categories'][:3])}] {paper['title'][:70]}")
    print("\nConfirm these look like the right person before adding the ID.")


def main():
    if "--check" in sys.argv:
        try:
            author_id = sys.argv[sys.argv.index("--check") + 1]
        except IndexError:
            raise SystemExit("usage: python3 scripts/fetch_arxiv.py --check <author_id>")
        check_author_id(author_id)
        return

    author_ids, names = load_roster()
    if not author_ids and not names:
        raise SystemExit(f"No arXiv-eligible people found in {PEOPLE_FILE}.")
    print(f"Searching arXiv for {len(author_ids) + len(names)} people "
          f"({len(author_ids)} by author ID, {len(names)} by name)...")

    cutoff = datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)
    papers = collect_papers(author_ids, names, cutoff, group_member_names())
    print(f"Retrieved {len(papers)} distinct papers.")
    papers = [p for p in papers if p.pop("_published") >= cutoff]
    for paper in papers:  # trim to a sensible number of badges for display
        paper["categories"] = paper["categories"][:3]
    print(f"Kept {len(papers)} from the last {WINDOW_DAYS} days.")

    OUTPUT_FILE.write_text(
        yaml.safe_dump(
            {
                "last_updated": datetime.now(timezone.utc).strftime("%B %d, %Y"),
                "window_days": WINDOW_DAYS,
                "papers": papers,
            },
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
    )
    print(f"Written {len(papers)} papers to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
