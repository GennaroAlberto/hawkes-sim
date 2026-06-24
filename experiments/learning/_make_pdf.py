"""Render LEARNING_REPORT.md -> a self-contained HTML (base64 figures) for PDF export."""
import base64
import os
import re

import markdown

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MD = os.path.join(ROOT, "LEARNING_REPORT.md")
HTML = os.path.join(ROOT, "LEARNING_REPORT.html")

with open(MD) as f:
    text = f.read()


def embed(m):
    path = os.path.join(ROOT, m.group(2))
    if not os.path.exists(path):
        return m.group(0)
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return f'![{m.group(1)}](data:image/png;base64,{b64})'


text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', embed, text)
body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc"])

CSS = """
@page { size: A4; margin: 20mm 18mm; }
* { box-sizing: border-box; }
body { font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif; color: #1a1a1a;
       line-height: 1.5; max-width: 820px; margin: 0 auto; font-size: 11pt; }
h1 { font-size: 22pt; border-bottom: 3px solid #1f3a5f; padding-bottom: 8px; color: #1f3a5f; }
h2 { font-size: 15pt; color: #1f3a5f; margin-top: 1.6em; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
h3 { font-size: 12.5pt; color: #2c4a6e; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 9.5pt; }
th, td { border: 1px solid #cfd8e3; padding: 6px 9px; text-align: left; vertical-align: top; }
th { background: #1f3a5f; color: #fff; }
tr:nth-child(even) td { background: #f4f7fb; }
code { background: #eef1f5; padding: 1px 4px; border-radius: 3px; font-size: 9.5pt; }
blockquote { border-left: 4px solid #1f9d55; background: #f0fbf4; margin: 1em 0; padding: 8px 14px; }
img { max-width: 100%; display: block; margin: 1em auto; border: 1px solid #ddd; }
strong { color: #122; }
hr { border: none; border-top: 1px solid #ddd; margin: 1.5em 0; }
"""

html = f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
with open(HTML, "w") as f:
    f.write(html)
print("wrote", HTML)
