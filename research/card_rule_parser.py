"""Parse reusable Japanese card-rule phrases before asking a model to translate.

Each recognized segment records the facts used to render English. Unknown segments
remain visible for a local model and for review; they are never silently discarded.
"""

from dataclasses import asdict, dataclass
import re
import unicodedata


ABILITY_START = re.compile(
    r"登場時|相手のアタック時|アタック時|(?:起動|起助)メイン|KO時|トリガー|ターン1回"
)
STANDALONE_KEYWORD = re.compile(r"速攻[:：](?:キャ|キヤ)ラ")
CARD_TYPES = {"ロックス海賊団": "Rocks Pirates"}


@dataclass
class Segment:
    source: str
    kind: str
    english: str | None
    facts: dict
    ability_index: int = 0
    ocr_note: str | None = None

    def to_json(self) -> dict:
        return asdict(self)


def compact_ocr(source: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", source))


def split_abilities(source: str) -> list[str]:
    text = compact_ocr(source)
    starts = [0]
    for match in STANDALONE_KEYWORD.finditer(text):
        starts.extend((match.start(), match.end()))
    for match in ABILITY_START.finditer(text):
        if match.start() and text[match.start() - 1] == "。":
            starts.append(match.start())
    starts = sorted(set(starts))
    return [text[start:end] for start, end in zip(starts, starts[1:] + [len(text)]) if text[start:end]]


def cards(count: str) -> str:
    return "card" if int(count) == 1 else "cards"


def remove_confirmed_duplicate_prefix(ability: str) -> tuple[str, str | None]:
    """Use a later complete rule when OCR repeats an unfinished ability prefix."""
    label = "相手のアタック時"
    positions = [match.start() for match in re.finditer(label, ability)]
    if len(positions) != 2:
        return ability, None
    prefix, later = ability[:positions[1]], ability[positions[1]:]
    if prefix.count("『") <= prefix.count("』") or later.count("『") != later.count("』"):
        return ability, None
    parsed = parse_known(later)
    if parsed is None or parsed[1] != len(later) or parsed[0].kind != "opponent_attack_power":
        return ability, None
    leading_attack_label = "アタック時" if prefix.startswith("アタック時") else ""
    return leading_attack_label + later, "Removed an unfinished repeated OCR prefix while preserving the When Attacking label; compare the effect with the card."


def parse_known(text: str) -> tuple[Segment, int] | None:
    """Recognize one complete leading phrase, returning its consumed length."""
    m = STANDALONE_KEYWORD.match(text)
    if m:
        return Segment(m.group(), "rush_character_keyword", "<Rush:Character>",
                       {"keyword": "Rush", "applies_to": "Character"}), m.end()

    m = re.match(
        r"(?P<when_attacking>アタック時)?相手のアタック時(?:ターン1回|ダーン1回)自分の手札から"
        r"『(?P<type>[^』]+)』を含む特徴を持つカード(?P<trash>\d+)枚を"
        r"捨てることができる[:：]相手の(?:キャ|キヤ)ラ(?P<targets>\d+)枚までを、"
        r"このターン中、パワー(?P<power>[+-]\d+)。?", text,
    )
    if m and m["type"] in CARD_TYPES:
        type_name = CARD_TYPES[m["type"]]
        timing = "[When Attacking]" if m["when_attacking"] else ""
        english = (
            timing + "[On Your Opponent's Attack][Once Per Turn] You may trash "
            f"{int(m['trash'])} {cards(m['trash'])} with a type including "
            f"\"{type_name}\" from your hand: Give up to {int(m['targets'])} of your "
            f"opponent's Characters {int(m['power']):+d} power during this turn."
        )
        facts = {"timing": "when_attacking_or_opponent_attack" if m["when_attacking"] else "opponent_attack", "once_per_turn": True,
                 "trash_from_hand": int(m["trash"]), "card_type_includes": type_name,
                 "opponent_characters_at_most": int(m["targets"]),
                 "power": int(m["power"]), "duration": "this_turn"}
        return Segment(m.group(), "opponent_attack_power", english, facts), m.end()

    m = re.match(r"登場時カード(?P<count>\d+)枚を引く。", text)
    if m:
        count = int(m["count"])
        return Segment(m.group(), "draw_on_play", f"[On Play] Draw {count} {cards(m['count'])}.",
                       {"timing": "on_play", "draw": count}), m.end()

    m = re.match(
        r"ターン1回自分の『(?P<type>[^』]+)』を含む特徴を持つリーダーが"
        r"アタックした時かアタックされた時、自分の手札(?P<trash>\d+)枚を"
        r"捨てて発動できる。", text,
    )
    if m and m["type"] in CARD_TYPES:
        type_name = CARD_TYPES[m["type"]]
        count = int(m["trash"])
        english = (
            f"[Once Per Turn] When your Leader with a type including \"{type_name}\" "
            "attacks or is attacked, you may trash "
            f"{count} {cards(m['trash'])} from your hand to activate this effect."
        )
        return Segment(m.group(), "leader_battle_activation", english,
                       {"once_per_turn": True, "leader_type_includes": type_name,
                        "timing": "leader_attacks_or_is_attacked", "trash_from_hand": count}), m.end()

    m = re.match(r"自分のリーダーを、このバトル中、パワー(?P<power>[+-]\d+)。", text)
    if m:
        power = int(m["power"])
        return Segment(m.group(), "leader_battle_power",
                       f"Your Leader gains {power:+d} power during this battle.",
                       {"target": "your_leader", "power": power, "duration": "this_battle"}), m.end()

    m = re.match(
        r"登場時相手は以下から(?P<choices>\d+)つを選ぶ。"
        r"・?カード(?P<draw>\d+)枚を引く。"
        r"相手は自身の手札(?P<trash>\d+)枚を捨てる。", text,
    )
    if m:
        choice = "one" if int(m["choices"]) == 1 else str(int(m["choices"]))
        english = (
            f"[On Play] Your opponent chooses {choice}:\n"
            f"* Draw {int(m['draw'])} {cards(m['draw'])}.\n"
            f"* Your opponent trashes {int(m['trash'])} {cards(m['trash'])} from their hand."
        )
        return Segment(m.group(), "opponent_choice", english, {k: int(v) for k, v in m.groupdict().items()}), m.end()

    m = re.match(
        r"登場時相手の元々のコスト(?P<cost>\d+)以下の(?:キャ|キヤ)ラ"
        r"(?P<targets>\d+)枚までは、次の相手のエンドフェイズ終了時まで、"
        r"アタックできない。", text,
    )
    if m:
        english = (
            f"[On Play] Up to {int(m['targets'])} of your opponent's Characters "
            f"with a base cost of {int(m['cost'])} or less cannot attack until "
            "the end of your opponent's next End Phase."
        )
        return Segment(m.group(), "attack_lock", english, {"timing": "on_play", "cost_at_most": int(m["cost"]), "targets_at_most": int(m["targets"])}), m.end()

    m = re.match(
        r"相手のアタック時ターン(?P<times>\d+)回自分の手札"
        r"(?P<trash>\d+)枚を捨てることができる[:：]自分のリーダーか"
        r"(?:キャ|キヤ)ラ(?P<targets>\d+)枚までを、このバトル中、"
        r"パワー(?P<power>[+-]\d+)。?", text,
    )
    if m and int(m["times"]) == 1:
        verb = "gains" if int(m["targets"]) == 1 else "gain"
        english = (
            "[On Your Opponent's Attack][Once Per Turn] "
            f"You may trash {int(m['trash'])} {cards(m['trash'])} from your hand: "
            f"Up to {int(m['targets'])} of your Leader or Characters {verb} "
            f"{m['power']} power during this battle."
        )
        facts = {"timing": "opponent_attack", "once_per_turn": True, "trash_from_hand": int(m["trash"]), "targets_at_most": int(m["targets"]), "power": int(m["power"])}
        return Segment(m.group(), "battle_power", english, facts), m.end()

    m = re.match(
        r"(?:起動|起助)メイン自分のドン!!(?P<don>\d+)枚とこの"
        r"(?:キャ|キヤ)ラをレストにできる[:：]相手の(?:キャ|キヤ)ラ"
        r"(?P<targets>\d+)枚までは、次の相手のエンドフェイズ終了時まで、"
        r"アタックできない。?", text,
    )
    if m:
        english = (
            f"[Activate:Main] You may rest {int(m['don'])} of your DON!! {cards(m['don'])} "
            f"and this character: Up to {int(m['targets'])} of your opponent's "
            "Characters cannot attack until the end of your opponent's next End Phase."
        )
        facts = {"timing": "activate_main", "rest_don": int(m["don"]), "targets_at_most": int(m["targets"]), "duration": "opponent_next_end_phase"}
        return Segment(m.group(), "attack_lock", english, facts), m.end()

    m = re.match(
        r"自分のリーダーが多色で、(?:相手|組手)の場の(?:ドン!!|ドンリ)が"
        r"(?P<don>\d+)枚以上ある場合、この(?:キャ|キヤ)ラは、速攻を得る。", text,
    )
    if m:
        english = (
            f"If your Leader is multicolored and your opponent has {int(m['don'])} "
            "or more DON!! cards on their field, this Character gains <Rush>."
        )
        return Segment(m.group(), "conditional_rush", english, {"leader": "multicolored", "opponent_don_at_least": int(m["don"]), "gain": "Rush"}), m.end()

    # Cost and effects are parsed independently so an unfamiliar follow-up can
    # go to the model without changing the known DON!! cost or active action.
    m = re.match(
        r"アタック時(?:ターン1回|ターンロ)自分のアクティブの"
        r"(?:ドン!!|ドン)(?P<don>\d+)枚を(?:ドン!!|ドシ!)デッキに"
        r"戻すことができる[:：]", text,
    )
    if m:
        english = (
            f"[When Attacking][Once Per Turn] Return {int(m['don'])} "
            f"of your active DON!! {cards(m['don'])} to your DON!! deck:"
        )
        return Segment(m.group(), "return_active_don_cost", english, {"timing": "when_attacking", "once_per_turn": True, "return_active_don": int(m["don"])}), m.end()

    m = re.match(r"この(?:キャ|キヤ)ラをアクティブにする。", text)
    if m:
        return Segment(m.group(), "set_active", "Set this Character as active.", {"target": "this_character"}), m.end()

    m = re.match(r"その後、自分のライフの上から(?P<count>\d+)枚を手札に加える。?", text)
    if m:
        count = int(m["count"])
        english = f"Then, add {'the top card' if count == 1 else f'the top {count} cards'} of your Life to your hand."
        return Segment(m.group(), "life_to_hand", english, {"from": "your_life_top", "count": count}), m.end()

    m = re.match(
        r"KO時自分のデッキの上から(?P<deck>\d+)枚までを、ライフの上に加える。"
        r"その後、相手のライフの上から(?P<life>\d+)枚までを、"
        r"トラッシュに(?:置|凹)く。", text,
    )
    if m:
        english = (
            f"[On K.O.] Add up to {int(m['deck'])} {cards(m['deck'])} from the top of your deck "
            "to the top of your Life cards. Then, trash up to "
            f"{int(m['life'])} {cards(m['life'])} from the top of your opponent's Life cards."
        )
        return Segment(m.group(), "ko_life_exchange", english, {"deck_to_life_at_most": int(m["deck"]), "opponent_life_to_trash_at_most": int(m["life"])}), m.end()

    m = re.match(
        r"トリガー自分のリーダーが『(?P<type>[^』]+)』を含む特徴を持つ場合、"
        r"カード(?P<draw>\d+)枚を引き、コスト(?P<cost>\d+)以下の"
        r"(?:キャ|キヤ)ラ(?P<targets>\d+)枚までを、持ち主のデッキの下に置く。?", text,
    )
    if m:
        english = (
            f"[Trigger] If your Leader has a type including \"{m['type']}\", "
            f"draw {int(m['draw'])} {cards(m['draw'])} and place up to "
            f"{int(m['targets'])} Characters with a cost of {int(m['cost'])} or less "
            "at the bottom of the owner's deck."
        )
        return Segment(m.group(), "trigger_draw_return", english, {"leader_type_includes": m["type"], "draw": int(m["draw"]), "character_cost_at_most": int(m["cost"]), "targets_at_most": int(m["targets"]), "destination": "owner_deck_bottom"}), m.end()

    # Keep known ability labels even when the following effect is unfamiliar.
    m = re.match(r"相手のアタック時", text)
    if m:
        return Segment(m.group(), "opponent_attack_label", "[On Your Opponent's Attack]",
                       {"timing": "opponent_attack"}), m.end()
    m = re.match(r"ターン1回", text)
    if m:
        return Segment(m.group(), "once_per_turn_label", "[Once Per Turn]",
                       {"once_per_turn": True}), m.end()
    return None


def parse_effects(source: str) -> list[Segment]:
    result = []
    for ability_index, ability in enumerate(split_abilities(source)):
        remaining, ocr_note = remove_confirmed_duplicate_prefix(ability)
        while remaining:
            parsed = parse_known(remaining)
            if parsed is None:
                result.append(Segment(remaining, "unresolved", None, {}, ability_index))
                break
            segment, consumed = parsed
            segment.ability_index = ability_index
            if ocr_note:
                segment.ocr_note = ocr_note
                ocr_note = None
            result.append(segment)
            remaining = remaining[consumed:]
    return result
