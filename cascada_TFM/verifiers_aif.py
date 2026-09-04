
import re

def _words(text):
    return re.findall(r"\S+", text or "")

def _sentences(text):
    partes = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p for p in partes if p.strip()]

def _paragraphs(text):
    partes = re.split(r"\n\s*\n", (text or "").strip())
    return [p for p in partes if p.strip()]

def _clean_word_re(word):
    return re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)

def contains_phrase(output, params):
    return params["phrase"].lower() in output.lower()

def contains_word(output, params):
    return bool(_clean_word_re(params["word"]).search(output))

def contains_number(output, params):
    return bool(re.search(r"\d", output))

def keyword_frequency(output, params):
    n = len(_clean_word_re(params["keyword"]).findall(output))
    return n >= params["n"]

def excludes_word(output, params):
    return not _clean_word_re(params["word"]).search(output)

def no_exclamation(output, params):
    return "!" not in output and "¡" not in output

def no_commas(output, params):
    return "," not in output

def no_questions(output, params):
    return "?" not in output and "¿" not in output

def starts_with(output, params):
    return output.strip().lower().startswith(params["prefix"].strip().lower())

def ends_with(output, params):
    return output.strip().lower().endswith(params["suffix"].strip().lower())

def postscript(output, params):
    marker = params.get("marker", "P.D.")
    return marker.lower() in output.lower()

def wrapped_in_quotes(output, params):
    s = output.strip()
    pares = [('"', '"'), ("'", "'"), ("«", "»"), (""", """), ("“", "”")]
    return any(s.startswith(a) and s.endswith(b) and len(s) > 1 for a, b in pares)

def max_words(output, params):
    return len(_words(output)) <= params["n"]

def min_words(output, params):
    return len(_words(output)) >= params["n"]

def word_range(output, params):
    n = len(_words(output))
    return params["n1"] <= n <= params["n2"]

def exact_sentences(output, params):
    return len(_sentences(output)) == params["n"]

def min_sentences(output, params):
    return len(_sentences(output)) >= params["n"]

def max_sentences(output, params):
    return len(_sentences(output)) <= params["n"]

def sentence_range(output, params):
    n = len(_sentences(output))
    return params["n1"] <= n <= params["n2"]

def exact_paragraphs(output, params):
    return len(_paragraphs(output)) == params["n"]

def single_paragraph(output, params):
    return len(_paragraphs(output)) == 1

_NUM_LIST_RE = re.compile(r"^\s*\d+[.)]\s+\S", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*[-*•]\s+\S", re.MULTILINE)
_SECTION_RE = re.compile(r"^\s*(?:E\d+\s*[:.]|#{1,6}\s)", re.MULTILINE)

def numbered_list(output, params):
    return len(_NUM_LIST_RE.findall(output)) >= 2

def numbered_list_n(output, params):
    return len(_NUM_LIST_RE.findall(output)) == params["n"]

def bullet_list(output, params):
    return len(_BULLET_RE.findall(output)) >= 2

def exact_bullets(output, params):
    return len(_BULLET_RE.findall(output)) == params["n"]

def sections(output, params):
    return len(_SECTION_RE.findall(output)) >= 2

def sections_n(output, params):
    return len(_SECTION_RE.findall(output)) == params["n"]

def two_responses(output, params):
    sep = params["separator"]
    partes = output.split(sep)
    return len(partes) == 2 and all(p.strip() for p in partes)

def all_lowercase(output, params):
    return output == output.lower()

def all_uppercase(output, params):
    return output == output.upper()

def all_questions(output, params):
    frases = _sentences(output)
    return bool(frases) and all(f.strip().endswith("?") for f in frases)

def capital_words_min(output, params):
    palabras = re.findall(r"[A-ZÁÉÍÓÚÑÜ]{2,}", output)
    return len(palabras) >= params["n"]

def includes_example(output, params):
    return "por ejemplo" in output.lower()

VERIFIERS = {
    "contains_phrase": contains_phrase,
    "contains_word": contains_word,
    "contains_number": contains_number,
    "keyword_frequency": keyword_frequency,
    "excludes_word": excludes_word,
    "no_exclamation": no_exclamation,
    "no_commas": no_commas,
    "no_questions": no_questions,
    "starts_with": starts_with,
    "ends_with": ends_with,
    "postscript": postscript,
    "wrapped_in_quotes": wrapped_in_quotes,
    "max_words": max_words,
    "min_words": min_words,
    "word_range": word_range,
    "exact_sentences": exact_sentences,
    "min_sentences": min_sentences,
    "max_sentences": max_sentences,
    "sentence_range": sentence_range,
    "exact_paragraphs": exact_paragraphs,
    "single_paragraph": single_paragraph,
    "numbered_list": numbered_list,
    "numbered_list_n": numbered_list_n,
    "bullet_list": bullet_list,
    "exact_bullets": exact_bullets,
    "sections": sections,
    "sections_n": sections_n,
    "two_responses": two_responses,
    "all_lowercase": all_lowercase,
    "all_uppercase": all_uppercase,
    "all_questions": all_questions,
    "capital_words_min": capital_words_min,
    "includes_example": includes_example,
}

def verify_aif(item, output):
    if not output:
        return None
    aux = item.get("aux_metadata", {})
    verifier_name = aux.get("verifier")
    fn = VERIFIERS.get(verifier_name)
    if fn is None:
        return None
    try:
        import json
        params = json.loads(aux.get("constraint_params") or "{}")
        return bool(fn(output, params))
    except Exception:
        return None
