# Nour-Bouajina.github.io

Personal research log, published automatically at https://nour-bouajina.github.io/ whenever this repo is pushed to `main`. No local build step — GitHub Pages builds the Jekyll site for you.

## How to add a new post

1. Add a file to `_posts/` named `YYYY-MM-DD-short-title.md`.
2. Start it with this front matter, then write Markdown below it:

   ```
   ---
   layout: post
   title: "Whatever title"
   date: 2026-08-11
   category: ssmds   # or: pic
   ---

   Your notes here. Regular Markdown — headings, links, images, code blocks all work.
   ```
3. To add an image or file, put it in `assets/` and reference it as `![caption](/assets/yourfile.png)`.
4. Commit and push. The live site updates in a minute or two — check the "Actions" tab on GitHub if it doesn't.

That's the whole workflow — no HTML/CSS/JS required, no local server, no build command.

## Structure

- `_posts/` — all posts, tagged `category: ssmds` or `category: pic`
- `ssmds.md` / `pic.md` — per-project pages that list posts by category
- `index.md` — home page
- `_config.yml` — site title/theme (minima)
