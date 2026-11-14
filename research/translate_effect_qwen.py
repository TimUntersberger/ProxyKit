"""Offline draft translation of extracted French/Japanese card effects."""

import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

from card_rule_parser import parse_effects

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "qwen3-1.7b"
LANGUAGES = {"ja": "Japanese", "fr": "French"}
JAPANESE_LETTERS = re.compile(r"[\u3040-\u309f\u30a0-\u30fa\u30fc-\u30ff\u3400-\u9fff]")


def normalize_card_terms(source: str, translated: str, language: str) -> str:
    if language != "ja":
        return translated
    result = translated.strip()
    result = re.sub(r"(?m)^\s*・\s*", "* ", result)
    if "登場時" in source:
        result = re.sub(r"^(?:When played|When (?:this|the) card is played|At the time of appearance),?\s*",
                        "[On Play] ", result, flags=re.I)
    if "KO時" in source:
        result = re.sub(r"^(?:When KO|On K\.?O\.?:?)\b[,:]?\s*", "[On K.O.] ", result, flags=re.I)
        result = re.sub(r"^\[On K\.O\.\] add\b", "[On K.O.] Add", result, flags=re.I)
    if "トリガー" in source:
        result = re.sub(r"\bTrigger\s*:\s*", "\n\n[Trigger] ", result, count=1, flags=re.I)
    if "特徴" in source:
        result = re.sub(r"\bcharacteristic\b", "type", result, flags=re.I)
        if "含む" in source:
            result = re.sub(r'\bhas the type\s+"([^"]+)"',
                            r'has a type including "\1"', result, flags=re.I)
    # A comma after a quoted Japanese type is sentence punctuation, not part of the type.
    for label in re.findall(r"『([^』]+)』", source):
        if "," not in label:
            result = result.replace(f'"{label},"', f'"{label}",')
    if "持ち主のデッキ" in source and not re.search(r"\bowner's deck\b", result, re.I):
        matches = list(re.finditer(r"\byour deck\b", result, flags=re.I))
        if matches:
            last = matches[-1]
            result = result[:last.start()] + "the owner's deck" + result[last.end():]
    if "デッキの上から" in source:
        result = re.sub(r"\bfrom your deck\b", "from the top of your deck", result, count=1, flags=re.I)
    if "ライフ" in source:
        result = re.sub(r"\bthe Life\b(?!\s+cards)", "your Life cards", result, count=1, flags=re.I)
        result = re.sub(r"\b(?:the opponent's|your opponent's) Life\b(?!\s+cards)",
                        "your opponent's Life cards", result, flags=re.I)
    if "トラッシュに" in source:
        result = re.sub(
            r"\bThen,? add (up to \d+ cards?) (from [^.]+?) to the Trash\b",
            r"Then, trash \1 \2", result, count=1, flags=re.I,
        )
    if "コスト" in source:
        result = re.sub(r"\bcharacter with cost\b", "Character with a cost of", result, flags=re.I)
    if "デッキの下" in source:
        result = re.sub(r"\b(?:in|into|below) the bottom of\b", "at the bottom of", result, flags=re.I)
        result = re.sub(r"\bin the bottom of\b", "at the bottom of", result, flags=re.I)
    result = re.sub(r"[ \t]+\n", "\n", result)
    return result.strip()


def rule_warnings(source: str, translated: str, language: str) -> list[str]:
    warnings = []
    normalized_source = source.lower().replace("’", "'")
    isolated_digits = re.findall(r"(?m)^[ \t]*\d+[ \t]*$", source)
    if isolated_digits:
        warnings.append("The OCR contains an isolated number line; check it against the card.")
    if language == "ja" and "凹く" in source:
        warnings.append("The OCR contains a likely misread Japanese character; check that verb against the card.")
    if language == "ja" and ("起助メイン" in source or "キヤラ" in source):
        warnings.append("The OCR appears to misread an ability label or Character; check those words against the card.")
    if language == "ja" and any(term in source for term in ("組手", "ドンリ", "ターンロ", "ドシ!")):
        warnings.append("The OCR appears to misread an opponent, DON!!, or turn-limit term; check those words against the card.")
    if language == "ja" and "ダーン1回" in source:
        warnings.append("The OCR may have misread the Once Per Turn label; check it against the card.")
    if language == "ja" and source.count("相手のアタック時") > 1:
        warnings.append("The OCR repeats part of the opponent-attack ability; review the repeated lines before trusting the translation.")
    # The "1" in ターン1回 becomes the words "Once Per Turn" in English.
    numeric_source = re.sub(r"(?m)^[ \t]*\d+[ \t]*$", "", source)
    if language == "ja":
        numeric_source = numeric_source.replace("ターン1回", "").replace("ダーン1回", "")
        if re.search(r"以下から1つを選ぶ", source) and re.search(r"chooses one", translated, re.I):
            numeric_source = numeric_source.replace("以下から1つを選ぶ", "以下から選ぶ", 1)
    numeric_source = unicodedata.normalize("NFKC", numeric_source)
    if language == "ja" and re.search(r"\btop card\b", translated, re.I):
        numeric_source = numeric_source.replace("上から1枚", "上から枚", 1)
    source_numbers = re.findall(r"(?<!\d)\d+(?!\d)", numeric_source)
    translated_numbers = re.findall(r"(?<!\d)\d+(?!\d)", translated)
    if sorted(source_numbers) != sorted(translated_numbers):
        warnings.append("Check the numbers against the original card.")
    lower = translated.lower()
    if ((language == "ja" and ("ターン1回" in source or "ダーン1回" in source)) or
            (language == "fr" and "une fois par tour" in normalized_source)) and "once per turn" not in lower:
        warnings.append("The Once Per Turn restriction may be missing.")
    if ((language == "ja" and "まで" in source) or
            (language == "fr" and "jusqu'à" in normalized_source)) and "up to" not in lower:
        warnings.append("An up-to limit may be missing or changed.")
    if language == "ja" and "キャラ" in source and "character" not in lower:
        warnings.append("Check whether Character was translated correctly.")
    if language == "ja" and "持ち主" in source and "owner" not in lower:
        warnings.append("Check whose deck the card goes to.")
    if language == "ja" and "トリガー" in source and "[trigger]" not in lower:
        warnings.append("The Trigger ability label may be missing.")
    if language == "ja" and "自分の" in source and not re.search(r"\b(?:your|you)\b", lower):
        warnings.append("Check whether the effect refers to your cards or your opponent's cards.")
    if language == "ja" and "相手のアタック時" in source and not re.search(r"opponent'?s? (?:attack|attacks)", lower):
        warnings.append("Check the opponent's attack timing.")
    if language == "ja" and "相手は" in source and re.search(r"\bthe player chooses\b", lower):
        warnings.append("Check whether the opponent or the player chooses the option.")
    return warnings


class LocalTranslator:
    """Load the offline model once, and use it only for unresolved segments."""

    def __init__(self, language: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not MODEL.joinpath("config.json").is_file():
            raise FileNotFoundError(f"Local translation model missing: {MODEL}")
        print("Loading local translation model...", flush=True)
        self.torch = torch
        self.language = language
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL, local_files_only=True,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )

    def generate(self, source: str, context: str = "", concise: bool = False) -> str:
        instruction = (
            f"Translate only this {LANGUAGES[self.language]} card-rule segment into English. "
            "Preserve numbers, owners, targets, costs, and restrictions. "
            "Do not repeat already translated context. Output English only."
        )
        if concise:
            messages = [{"role": "user", "content": instruction + "\n" + source}]
        else:
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content":
                 (f"Previous parsed context: {context}\n" if context else "") + source},
            ]
        ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt",
            add_generation_prompt=True, enable_thinking=False,
        ).to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                input_ids=ids, attention_mask=self.torch.ones_like(ids),
                max_new_tokens=256, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(output[0, ids.shape[-1]:], skip_special_tokens=True).strip()

    def translate(self, source: str, context: str = "") -> str:
        result = self.generate(source, context)
        if not result or JAPANESE_LETTERS.search(result):
            print("Retrying unresolved segment with a shorter prompt...", flush=True)
            result = self.generate(source, concise=True)
        if not result or JAPANESE_LETTERS.search(result):
            raise RuntimeError("The local model did not produce an English translation for an unresolved effect segment. Correct the OCR text and retry.")
        return normalize_card_terms(source, result, self.language)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("draft", type=Path)
    parser.add_argument("--language", choices=LANGUAGES, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--text-stdin", action="store_true",
                        help="translate edited OCR text supplied on standard input")
    args = parser.parse_args()
    draft = json.loads(args.draft.read_text(encoding="utf-8"))
    source = (sys.stdin.buffer.read().decode("utf-8") if args.text_stdin
              else draft["draft_text"]).strip()
    if not source:
        parser.error("OCR draft is empty")
    segments = parse_effects(source) if args.language == "ja" else []
    used_model = args.language != "ja" or any(segment.english is None for segment in segments)
    if args.language == "ja":
        model = None
        for index, segment in enumerate(segments):
            if segment.english is not None:
                continue
            if model is None:
                model = LocalTranslator(args.language)
            context = " ".join(
                previous.english or "" for previous in segments[:index]
                if previous.ability_index == segment.ability_index
            ).strip()
            print(f"Translating unresolved segment {index + 1}/{len(segments)}...", flush=True)
            segment.english = model.translate(segment.source, context)
        paragraphs = []
        for index, segment in enumerate(segments):
            if not paragraphs or segment.ability_index != segments[paragraphs[-1][0]].ability_index:
                paragraphs.append([index])
            else:
                paragraphs[-1].append(index)
        result = "\n".join(
            " ".join(segments[index].english for index in group).strip()
            for group in paragraphs
        )
    else:
        result = LocalTranslator(args.language).translate(source)
    payload = {
        "source": draft.get("source"),
        "source_language": args.language,
        "ocr_text": source,
        "translation": result,
        "model": ("Japanese phrase parser + Qwen3-1.7B" if used_model else "Japanese phrase parser")
        if args.language == "ja" else "Qwen3-1.7B",
        "segments": [segment.to_json() for segment in segments],
        "warnings": rule_warnings(source, result, args.language),
        "review_required": True,
    }
    if args.language == "ja" and used_model:
        payload["warnings"].append("Some effect wording was translated by the local model; review those segments carefully.")
    for note in dict.fromkeys(segment.ocr_note for segment in segments if segment.ocr_note):
        payload["warnings"].append(note)
    output = args.output or args.draft.with_name(args.draft.stem + "_translated.json")
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved draft translation: {output}")
    print(result)


if __name__ == "__main__":
    main()
