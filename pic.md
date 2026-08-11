---
layout: page
title: PIC
permalink: /pic/
---

Notes and progress on the PIC research project, newest first.

Code lives at [github.com/Nour-Bouajina/PIC](https://github.com/Nour-Bouajina/PIC).

<ul>
{% for post in site.categories.pic %}
  <li>
    <span>{{ post.date | date: "%Y-%m-%d" }}</span> —
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
  </li>
{% endfor %}
</ul>
