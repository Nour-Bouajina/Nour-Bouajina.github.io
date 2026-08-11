---
layout: page
title: SSMDS
permalink: /ssmds/
---

Notes and progress on the SSMDS research project, newest first.

Code lives at [github.com/Nour-Bouajina/SSMDS](https://github.com/Nour-Bouajina/SSMDS).

<ul>
{% for post in site.categories.ssmds %}
  <li>
    <span>{{ post.date | date: "%Y-%m-%d" }}</span> —
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
  </li>
{% endfor %}
</ul>
