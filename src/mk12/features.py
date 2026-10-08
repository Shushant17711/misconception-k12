"""Shared input text for every classifier (design §4)."""


def build_text(row, transcript: str) -> str:
    return (f"Topic: {row.topic_name}\nQuestion: {row.question_text}\n"
            f"Correct answer: {row.correct_answer}\nStudent working:\n{transcript}")
