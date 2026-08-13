# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Jekyll blog (theme: `minima`, dark skin) that serves as a running research log for two projects: **SSMDS** and **PIC**. It has no application code — content is Markdown pages and dated posts.

## Build / deploy

There is no local build step. GitHub Pages builds and deploys this repo automatically on every push to the default branch (see the comment in `_config.yml`). If you want to preview locally anyway, this repo has no `Gemfile`, so a local Jekyll install (`gem install jekyll bundler`, then `bundle exec jekyll serve`) would need to be set up first — it isn't currently.

## Structure and conventions

- `index.md` — home page, links to the two project pages.
- `pic.md`, `ssmds.md` — per-project landing pages. Each renders posts filtered by Jekyll category (`site.categories.pic`, `site.categories.ssmds`) via a Liquid loop. Do not hardcode post lists here — new posts appear automatically as long as their category is set correctly.
- `_posts/` — dated research-log entries, filename format `YYYY-MM-DD-title.md`. Each post's front matter must set `category` to either `ssmds` or `pic` — this is what routes it to the correct landing page. Front matter shape:
  ```yaml
  ---
  layout: post
  title: "..."
  date: YYYY-MM-DD
  category: ssmds  # or pic
  ---
  ```
- `_config.yml` — site title/description, theme, and `header_pages` (nav bar entries: `ssmds.md`, `pic.md`).
- Actual code for each project lives in the separate `SSMDS` and `PIC` repos, not here — this repo only publishes notes about them, and pages here link out to those repos.

## Working draft posts

Some posts (e.g. `_posts/2026-08-11-normalizing-flows-for-pic.md`) are scaffolded with HTML-comment section prompts (Motivation, Model and Theory, How It's Applied Here, Demo, Results) meant to be filled in incrementally as the work progresses, then pushed. When asked to draft or continue a research-log post, follow this same section structure and leave unfinished sections as comments rather than deleting them.

## Workflow: auto-commit and push

After making a substantial edit to a file (filling in or rewriting meaningful content, fixing a significant error) — as opposed to a trivial typo/formatting fix — commit and push it automatically, without asking for confirmation first. Use a concise, descriptive commit message.

If the working tree has unrelated pending changes that weren't part of the current task, flag them to the user before folding them into the commit.
