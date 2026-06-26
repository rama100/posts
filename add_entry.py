#!/usr/bin/env python3
"""
Add a new entry as the first item in a section of index.html.
The entry is also added to the Recent section, which is capped at 5 items.

Usage:
    python add_entry.py --title "My New Article" --url "https://example.com/article" --section "AI Pulse"
    python add_entry.py --embed --url "https://www.linkedin.com/posts/..." --section "AI Pulse"
    python add_entry.py --embed --url "https://www.youtube.com/watch?v=..." --section "AI Demystified"

Sections (case-insensitive, partial match supported):
    Recent, AI Demystified, AI Playbook, AI Pulse, AI/ML Miscellany, Random
"""

import argparse
import re
import sys
from datetime import datetime
from html import escape
from pathlib import Path
import subprocess
from urllib.parse import parse_qs, unquote, urlparse

RECENT_SECTION_ID = "recent"
RECENT_ENTRY_LIMIT = 5
SECTION_END_PATTERN = re.compile(
    r'\s*</div>\s*<hr\s+style="border:\s*1px\s+solid\s+black;">',
    re.I,
)
LINKEDIN_EMBED_BASE = "https://www.linkedin.com/embed/feed/update/"
LINKEDIN_EMBED_WIDTH = 504
LINKEDIN_EMBED_HEIGHT = 589
YOUTUBE_EMBED_WIDTH = 560
YOUTUBE_EMBED_HEIGHT = 315

# Map friendly section names to the div IDs in index.html
SECTION_MAP = {
    "recent":          RECENT_SECTION_ID,
    "ai demystified":  "starter-pack",
    "starter-pack":    "starter-pack",
    "ai playbook":     "playbook",
    "playbook":        "playbook",
    "ai pulse":        "pulse",
    "pulse":           "pulse",
    "ai/ml miscellany": "data-science",
    "data-science":    "data-science",
    "miscellany":      "data-science",
    "random":          "random",
}


def resolve_section(name: str) -> str:
    """Resolve a user-supplied section name to a div ID."""
    key = name.strip().lower()

    # Exact match first
    if key in SECTION_MAP:
        return SECTION_MAP[key]

    # Partial / substring match
    for friendly, div_id in SECTION_MAP.items():
        if key in friendly or friendly in key:
            return div_id

    friendly_names = ["Recent", "AI Demystified", "AI Playbook", "AI Pulse",
                      "AI/ML Miscellany", "Random"]
    print(f"Error: Unknown section '{name}'.", file=sys.stderr)
    print(f"Valid sections: {', '.join(friendly_names)}", file=sys.stderr)
    sys.exit(1)


def find_section_entry_bounds(html: str, section_id: str) -> tuple[int, int]:
    """Return the content bounds after the section nav <hr> and before close."""
    # Find the section div: <div id="SECTION_ID" class="section level3">
    pattern = rf'(<div\s+id="{re.escape(section_id)}"\s+class="section level3">)'
    match = re.search(pattern, html)
    if not match:
        print(f"Error: Could not find section div with id='{section_id}' "
              f"in index.html.", file=sys.stderr)
        sys.exit(1)

    section_start = match.end()
    section_end_match = SECTION_END_PATTERN.search(html, section_start)
    if not section_end_match:
        print(f"Error: Could not find the end of section '{section_id}'.",
              file=sys.stderr)
        sys.exit(1)
    section_end = section_end_match.start()

    # Inside the section, find the first <hr> (which follows the nav bar div).
    hr_match = re.search(r'<hr\s*/?>', html[section_start:section_end])
    if not hr_match:
        print(f"Error: Could not find an <hr> tag in section '{section_id}'.",
              file=sys.stderr)
        sys.exit(1)

    return section_start + hr_match.end(), section_end


def build_link_entry(title: str, url: str, date_str: str) -> str:
    """Build the HTML for one link entry, without separator <hr> tags."""
    return f'<a href="{url}">\n{title} </a> ({date_str})'


def linkedin_embed_url(url: str) -> str | None:
    """Convert a LinkedIn post URL to the iframe src used by LinkedIn embeds."""
    raw_url = unquote(url.strip())
    parsed = urlparse(raw_url)
    if "linkedin.com" not in parsed.netloc.lower():
        return None

    urn_match = re.search(
        r'urn:li:(activity|share|ugcPost):(\d+)', raw_url, flags=re.I)
    if urn_match:
        kind = normalize_linkedin_urn_kind(urn_match.group(1))
        return f"{LINKEDIN_EMBED_BASE}urn:li:{kind}:{urn_match.group(2)}"

    post_match = re.search(
        r'-(activity|share|ugcPost)-(\d+)(?:[-/?#]|$)',
        raw_url,
        flags=re.I,
    )
    if post_match:
        kind = normalize_linkedin_urn_kind(post_match.group(1))
        return f"{LINKEDIN_EMBED_BASE}urn:li:{kind}:{post_match.group(2)}"

    return None


def normalize_linkedin_urn_kind(kind: str) -> str:
    """Return LinkedIn URN kind with the capitalization LinkedIn expects."""
    if kind.lower() == "ugcpost":
        return "ugcPost"
    return kind.lower()


def youtube_video_id(url: str) -> str | None:
    """Extract a YouTube video ID from common public URL forms."""
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower().removeprefix("www.")
    path_parts = [part for part in parsed.path.split("/") if part]

    if host == "youtu.be" and path_parts:
        return path_parts[0]

    if host not in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        return None

    if parsed.path == "/watch":
        return parse_qs(parsed.query).get("v", [None])[0]

    if len(path_parts) >= 2 and path_parts[0] in {"embed", "shorts", "live"}:
        return path_parts[1]

    return None


def youtube_embed_url(url: str) -> str | None:
    """Convert a YouTube URL to the iframe src used by YouTube embeds."""
    video_id = youtube_video_id(url)
    if not video_id:
        return None
    return f"https://www.youtube.com/embed/{video_id}"


def resolve_embed_url(url: str) -> tuple[str, str]:
    """Return (provider, embed src) for supported embed URLs."""
    linkedin_url = linkedin_embed_url(url)
    if linkedin_url:
        return "linkedin", linkedin_url

    youtube_url = youtube_embed_url(url)
    if youtube_url:
        return "youtube", youtube_url

    print("Error: Could not convert URL to an embed.", file=sys.stderr)
    print("Supported embed URLs: LinkedIn posts and YouTube videos.",
          file=sys.stderr)
    sys.exit(1)


def build_embed_entry(url: str) -> str:
    """Build the HTML for one iframe embed entry."""
    provider, embed_url = resolve_embed_url(url)
    embed_url = escape(embed_url, quote=True)

    if provider == "linkedin":
        return (
            f'<iframe src="{embed_url}" height="{LINKEDIN_EMBED_HEIGHT}" '
            f'width="{LINKEDIN_EMBED_WIDTH}" frameborder="0" '
            f'allowfullscreen="" title="Embedded post"></iframe>'
        )

    return (
        f'<iframe width="{YOUTUBE_EMBED_WIDTH}" '
        f'height="{YOUTUBE_EMBED_HEIGHT}" src="{embed_url}" '
        f'title="YouTube video player" frameborder="0" '
        f'allow="accelerometer; autoplay; clipboard-write; encrypted-media; '
        f'gyroscope; picture-in-picture; web-share" '
        f'referrerpolicy="strict-origin-when-cross-origin" '
        f'allowfullscreen></iframe>'
    )


def build_entry(title: str, url: str, date_str: str, embed: bool = False) -> str:
    """Build the HTML for one entry, without separator <hr> tags."""
    if embed:
        return build_embed_entry(url)
    return build_link_entry(title, url, date_str)


def insert_entry_in_section(html: str, entry: str, section_id: str) -> str:
    """Insert a new entry as the first item in the given non-Recent section."""
    insert_pos, _ = find_section_entry_bounds(html, section_id)
    new_entry = f'\n{entry}\n<hr>'
    return html[:insert_pos] + new_entry + html[insert_pos:]


def split_recent_entries(entries_html: str) -> list[str]:
    """Split the Recent section body into individual entry blocks."""
    entries = re.split(r'\s*<hr\s*/?>\s*', entries_html.strip())
    return [entry.strip() for entry in entries if entry.strip()]


def entry_key(entry: str) -> str:
    """Normalize an entry enough to avoid duplicate Recent links."""
    match = re.search(r'<a\s+href="([^"]+)">', entry)
    if match:
        return match.group(1).strip()
    match = re.search(r'<iframe\s+[^>]*src="([^"]+)"', entry)
    if match:
        return match.group(1).strip()
    return re.sub(r'\s+', ' ', entry).strip()


def update_recent_section(html: str, entry: str) -> str:
    """Add entry to Recent and keep only the most recent 5 entries."""
    entries_start, entries_end = find_section_entry_bounds(
        html, RECENT_SECTION_ID)
    existing_entries = split_recent_entries(html[entries_start:entries_end])

    new_key = entry_key(entry)
    existing_entries = [
        existing for existing in existing_entries
        if entry_key(existing) != new_key
    ]

    recent_entries = [entry] + existing_entries
    recent_entries = recent_entries[:RECENT_ENTRY_LIMIT]
    recent_html = '\n' + '\n<hr>\n'.join(recent_entries)

    return html[:entries_start] + recent_html + html[entries_end:]


def add_entry(html: str, title: str, url: str, section_id: str,
              date_str: str, embed: bool = False) -> str:
    """Insert a new entry and update the Recent section."""
    entry = build_entry(title, url, date_str, embed)
    if section_id != RECENT_SECTION_ID:
        html = insert_entry_in_section(html, entry, section_id)
    return update_recent_section(html, entry)


def git_commit_and_push():
    """Git add, commit, and push"""
    try:
        # Git add
        subprocess.run(['git', 'add', 'index.html'], check=True)

        # Git commit
        subprocess.run(['git', 'commit', '-m', 'Update index.html'], check=True)

        # Git push
        subprocess.run(['git', 'push'], check=True)

        print("\nSuccessfully committed and pushed to GitHub!")

    except subprocess.CalledProcessError as e:
        print(f"\nError during git operation: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Add a new entry to a section of index.html."
    )
    parser.add_argument("--title", default=None,
                        help="Title / link text for the new entry.")
    parser.add_argument("--url", required=True,
                        help="URL the entry should link to.")
    parser.add_argument("--section", required=True,
                        help="Section name (e.g. 'AI Pulse', 'Random').")
    parser.add_argument("--embed", action="store_true",
                        help="Add URL as a LinkedIn or YouTube embed.")
    parser.add_argument("--date", default=None,
                        help="Date label, e.g. 'March 2026'. "
                             "Defaults to current month/year.")
    parser.add_argument("--file", default="index.html",
                        help="Path to index.html (default: index.html).")

    args = parser.parse_args()

    if not args.embed and not args.title:
        parser.error("--title is required unless --embed is used.")

    section_id = resolve_section(args.section)
    date_str = args.date or datetime.now().strftime("%B %Y")

    path = Path(args.file)
    if not path.exists():
        print(f"Error: File '{path}' not found.", file=sys.stderr)
        sys.exit(1)

    html = path.read_text(encoding="utf-8")
    updated = add_entry(html, args.title, args.url, section_id, date_str,
                        args.embed)
    path.write_text(updated, encoding="utf-8")

    if args.embed:
        print(f"Added embed as the first entry in section '{section_id}'.")
    else:
        print(f"Added \"{args.title}\" as the first entry in "
              f"section '{section_id}' with date ({date_str}).")

    # Git commit and push
    git_commit_and_push()

if __name__ == "__main__":
    main()
