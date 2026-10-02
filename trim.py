def trim_linkedin(li: dict) -> dict:
    loc = (li.get("location") or {}).get("parsed", {}) or {}
    return {
        "name": f"{li.get('firstName','')} {li.get('lastName','')}".strip(),
        "headline": li.get("headline"),
        "about": (li.get("about") or "")[:600],
        "location": loc.get("city"),
        "current_position": [
            {"title": p.get("position"), "company": p.get("companyName")}
            for p in (li.get("currentPosition") or [])[:2]
        ],
        "experience": [
            {"title": e.get("position"), "company": e.get("companyName")}
            for e in (li.get("experience") or [])[:4]
        ],
        "education": [
            {"school": e.get("schoolName"), "field": e.get("fieldOfStudy")}
            for e in (li.get("profileTopEducation") or [])[:2]
        ],
        # Weak signal: who/what they follow professionally, NOT personal hobbies.
        "followed_topics": [
            el.get("title")
            for cat in (li.get("interests") or [])
            for el in (cat.get("elements") or [])[:2]
            if cat.get("interestName") in ("Groups", "Newsletters")
        ][:6],
    }


def trim_instagram(ig: dict) -> dict:
    posts = ig.get("latestPosts") or []
    captions = [(p.get("caption") or "")[:200] for p in posts if p.get("caption")][:8]
    hashtags = list({h for p in posts for h in (p.get("hashtags") or []) if h})[:15]
    mentions = list({m for p in posts for m in (p.get("mentions") or []) if m})[:10]
    return {
        "bio": ig.get("biography"),
        "external_url": ig.get("externalUrl"),
        "followers": ig.get("followersCount"),
        "is_private": ig.get("private"),
        "post_count": ig.get("postsCount"),
        "recent_captions": captions,
        "hashtags": hashtags,
        "mentions": mentions,
    }