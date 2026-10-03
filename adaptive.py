from __future__ import annotations

from datetime import date
from db import due_revisions, get_topic_mastery, weak_strong_areas


def difficulty_for_topic(student_id: str, subject: str, topic: str) -> str:
    row = get_topic_mastery(student_id, subject, topic)
    if row is None or not row["enough_data"]:
        return "Medium"  # not enough evidence yet to justify changing the level
    score = float(row["mastery_score"])
    if score < 45:
        return "Easy"
    if score < 75:
        return "Medium"
    return "Hard"


def next_best_action(student_id: str) -> dict[str, str]:
    due = due_revisions(student_id)
    if due:
        item = due[0]
        return {"title": "Smart Revision", "reason": f"{item['topic']} is due for revision and its mastery is {float(item['mastery']):.0f}%.", "subject": item["subject"], "topic": item["topic"]}
    areas = weak_strong_areas(student_id, 1)
    if areas["weak"]:
        item = areas["weak"][0]
        return {"title": "Practice a Weak Topic", "reason": f"{item['topic']} is at {float(item['mastery_score']):.0f}% mastery ({item['label']}) after {item['attempts']} answers.", "subject": item["subject"], "topic": item["topic"]}
    if areas["building"]:
        item = areas["building"][0]
        return {"title": "Build Your Profile", "reason": f"Only {item['attempts']} answer(s) recorded for {item['topic']}. Answer a few more so your strengths and weaknesses can be measured accurately.", "subject": item["subject"], "topic": item["topic"]}
    return {"title": "Start Learning", "reason": "No learning history is available yet. Start a topic to build your profile.", "subject": "", "topic": ""}
