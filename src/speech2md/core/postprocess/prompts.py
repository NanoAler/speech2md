PROMPT_RU = (
    "Ты — помощник по очистке распознанной речи.\n"
    "Исправь только очевидные ошибки распознавания (неправильные окончания, "
    "потерянные слова, неверные термины). Не меняй смысл, стиль и структуру речи.\n"
    "Оформи результат как Markdown. Сохрани абзацы и интонационные паузы как пустые строки.\n"
    "Не добавляй ничего от себя.\n\n"
    "{text}"
)

PROMPT_EN = (
    "You are a speech-recognition cleanup assistant.\n"
    "Fix only obvious recognition errors (wrong endings, missing words, incorrect terms). "
    "Do not change the meaning, style, or structure of the speech.\n"
    "Format the output as Markdown. Preserve paragraphs and pauses as blank lines.\n"
    "Do not add anything extra.\n\n"
    "{text}"
)

PROMPTS: dict[str, str] = {
    "ru": PROMPT_RU,
    "en": PROMPT_EN,
}


def get_prompt(language: str = "", template_override: str = "") -> str:
    if template_override:
        return template_override
    key = language if language in PROMPTS else "en"
    return PROMPTS[key]
