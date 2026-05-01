#!/usr/bin/env python3
"""
Add a new entry as the first item in a section of index.html.
The entry is also added to the Recent section, which is capped at 5 items.

Usage:
    python add_entry.py --title "My New Article" --url "https://example.com/article" --section "AI Pulse"

Sections (case-insensitive, partial match supported):
    Recent, AI Demystified, AI Playbook, AI Pulse, AI/ML Miscellany, Random
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
import subprocess

RECENT_SECTION_ID = "recent"
RECENT_ENTRY_LIMIT = 5
SECTION_END_MARKER = '\n</div>\n<hr style="border: 1px solid black;">'

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
    section_end = html.find(SECTION_END_MARKER, section_start)
    if section_end == -1:
        print(f"Error: Could not find the end of section '{section_id}'.",
              file=sys.stderr)
        sys.exit(1)

    # Inside the section, find the first <hr> (which follows the nav bar div).
    hr_match = re.search(r'<hr\s*/?>', html[section_start:section_end])
    if not hr_match:
        print(f"Error: Could not find an <hr> tag in section '{section_id}'.",
              file=sys.stderr)
        sys.exit(1)

    return section_start + hr_match.end(), section_end


def build_entry(title: str, url: str, date_str: str) -> str:
    """Build the HTML for one link entry, without separator <hr> tags."""
    return f'<a href="{url}">\n{title} </a> ({date_str})'


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
              date_str: str) -> str:
    """Insert a new entry and update the Recent section."""
    entry = build_entry(title, url, date_str)
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
    parser.add_argument("--title", required=True,
                        help="Title / link text for the new entry.")
    parser.add_argument("--url", required=True,
                        help="URL the entry should link to.")
    parser.add_argument("--section", required=True,
                        help="Section name (e.g. 'AI Pulse', 'Random').")
    parser.add_argument("--date", default=None,
                        help="Date label, e.g. 'March 2026'. "
                             "Defaults to current month/year.")
    parser.add_argument("--file", default="index.html",
                        help="Path to index.html (default: index.html).")

    args = parser.parse_args()

    section_id = resolve_section(args.section)
    date_str = args.date or datetime.now().strftime("%B %Y")

    path = Path(args.file)
    if not path.exists():
        print(f"Error: File '{path}' not found.", file=sys.stderr)
        sys.exit(1)

    html = path.read_text(encoding="utf-8")
    updated = add_entry(html, args.title, args.url, section_id, date_str)
    path.write_text(updated, encoding="utf-8")

    print(f"Added \"{args.title}\" as the first entry in "
          f"section '{section_id}' with date ({date_str}).")

    # Git commit and push
    git_commit_and_push()

if __name__ == "__main__":
    main()
