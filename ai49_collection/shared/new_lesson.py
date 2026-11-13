#!/usr/bin/env python3
"""Scaffold one lesson of "Modern AI in 49 Papers" from shared/template/, using its entry in collection.js.

Usage (from anywhere):
  python ai49_collection/shared/new_lesson.py <slug> [--mark W2V] [--code-url URL --code-name org/repo] [--force]

Creates ai49_collection/<slug>/ with index.html, widgets.js, img/, README.md and _work/{inventory,outline}.md
(_work/ is git-ignored). Fills title, lesson number, paper citation and storage prefix; everything still to be
written is marked {{…}} so check_book.py reports it until it is done.
"""
import argparse
import json
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent          # ai49_collection/
TEMPLATE = ROOT / 'shared' / 'template'


def load_lessons():
    js = (ROOT / 'collection.js').read_text(encoding='utf-8')
    return [json.loads(m) for m in re.findall(r'^\s*(\{"n": .*\})\s*,?\s*$', js, flags=re.M)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('slug')
    ap.add_argument('--mark', help='2–4 character logo mark (default: from the title)')
    ap.add_argument('--code-url', default='{{CODE_URL}}')
    ap.add_argument('--code-name', default='{{CODE_NAME}}')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()

    lessons = load_lessons()
    if len(lessons) != 49:
        sys.exit(f'expected 49 lessons in collection.js, found {len(lessons)}')
    me = next((l for l in lessons if l['slug'] == a.slug), None)
    if not me:
        sys.exit(f'no lesson with slug {a.slug!r}; known: ' + ', '.join(l['slug'] for l in lessons))
    dest = ROOT / a.slug
    if (dest / 'index.html').exists() and not a.force:
        sys.exit(f'{dest / "index.html"} exists (use --force to overwrite)')
    for sub in ('img', '_work'):
        (dest / sub).mkdir(parents=True, exist_ok=True)

    mark = a.mark or ''.join(w[0] for w in re.findall(r'[A-Za-z0-9]+', me['title']))[:3].upper()
    p = me['paper']
    subs = {
        '{{SLUG}}': me['slug'], '{{N}}': str(me['n']), '{{TITLE}}': me['title'], '{{MARK}}': mark,
        '{{PAPER_URL}}': p['url'], '{{PAPER_TITLE}}': p['title'], '{{AUTHORS}}': p['authors'],
        '{{VENUE}}': p['venue'], '{{YEAR}}': str(p['year']),
        '{{CODE_URL}}': a.code_url, '{{CODE_NAME}}': a.code_name,
    }
    for name in ('index.html', 'widgets.js'):
        text = (TEMPLATE / name).read_text(encoding='utf-8')
        for k, v in subs.items():
            text = text.replace(k, v)
        (dest / name).write_text(text, encoding='utf-8')

    readme = dest / 'README.md'
    if not readme.exists() or a.force:
        readme.write_text(f"""# {me['title']}: Interactive Web Book

Lesson {me['n']} of [Modern AI in 49 Papers](../). {{{{What the book teaches, from zero, in one paragraph.}}}}

## What's inside

- {{{{N}}}} chapters in {{{{P}}}} parts: {{{{parts}}}}, plus an equation cheat sheet, glossary and sources.
- {{{{W}}}} interactive widgets: {{{{list}}}}.
- {{{{Q}}}} self-check quiz questions.

## Sources

- Paper: {p['authors']}. *{p['title']}*. {p['venue']}, {p['year']}. {p['url']}
- Code: {{{{repository and licence, or "no official code"}}}}.
- Figures: {{{{reused under which licence, or "all diagrams are original"}}}}.

This is an independent educational companion, not an official publication of the authors. Numbers quoted from the paper come from {{{{version}}}}; toy-model numbers come from the small models that run in the page.

## Files

```
index.html     the book (uses ../shared/ for the engine, styles, maths library and refresher widgets)
widgets.js     widgets for this paper
img/           cover.jpg and figures
```
""", encoding='utf-8')

    work = {
        'inventory.md': f"# Inventory: {me['title']}\n\nPaper: {p['title']} ({p['url']}) · local PDF: ../papers/ · version read: …\nCode: … · licence: …\n\n"
                        "## Equations (number · where · one-line meaning · chapter)\n## Algorithms\n## Figures (number · what it shows · licence: reuse / redraw · chapter)\n"
                        "## Tables (number · what it reports · chapter)\n## Headline numbers (value · where · chapter)\n## Concepts the paper assumes (→ Part I)\n"
                        "## The paper's own running examples (beginner-friendly? → toy plan)\n",
        'outline.md': f"# Outline: {me['title']}\n\nToy model: what it is, what is exact, what is a stand-in.\n\n"
                      "## Part I · Foundations\n| # | id | Title | One idea | Widgets | Inventory items covered | Quiz topics |\n|---|----|-------|----------|---------|-------------------------|-------------|\n\n"
                      "## Coverage check\n- [ ] every equation → a chapter\n- [ ] every figure/table → a chapter\n- [ ] every headline number → a chapter\n",
    }
    for name, text in work.items():
        f = dest / '_work' / name
        if not f.exists():
            f.write_text(text, encoding='utf-8')

    print(f'Scaffolded {dest} for lesson {me["n"]}: {me["title"]}.')
    import csv
    for row in csv.DictReader(open(ROOT / 'papers.csv', encoding='utf-8-sig')):
        if row['slug'] == a.slug:
            print(f'Paper PDF ({row["downloaded_version"]}):', ROOT / row['local_file'])
    print('Next: fill _work/inventory.md and _work/outline.md, then write the chapters (see STANDARD.md).')


if __name__ == '__main__':
    main()
