import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from card_rule_parser import parse_effects


class CardRuleParserTests(unittest.TestCase):
    def test_choice_and_opponent_attack(self):
        source = (
            "登場時 相手は以下から1つを選ぶ。\n・カード2枚を引く。\n"
            "相手は自身の手札2枚を捨てる。\n"
            "相手のアタック時ターン1回自分の手札1枚を捨てることができる:自\n"
            "分のリーダーかキャラ1枚までを、このバトル中、パワー+1000。"
        )
        parts = parse_effects(source)
        self.assertEqual([p.kind for p in parts], ["opponent_choice", "battle_power"])
        self.assertEqual([p.ability_index for p in parts], [0, 1])
        self.assertIn("Your opponent trashes 2 cards", parts[0].english)
        self.assertIn("[Once Per Turn]", parts[1].english)

    def test_attack_lock_and_don_cost(self):
        source = (
            "登場時 相手の元々のコスト6以下のキャラ1枚までは、次の相手のエン\n"
            "ドフェイズ終了時まで、アタックできない。\n"
            "起助メイン自分のドン!!3枚とこのキャラをレストにできる:相手のキヤ\n"
            "ラ1枚までは、次の相手のエンドフェイズ終了時まで、アタックできない。"
        )
        parts = parse_effects(source)
        self.assertEqual([p.kind for p in parts], ["attack_lock", "attack_lock"])
        self.assertEqual(parts[1].facts["rest_don"], 3)
        self.assertIn("DON!! cards", parts[1].english)

    def test_rush_cost_and_two_effects(self):
        source = (
            "自分のリーダーが多色で、組手の場のドンリが5枚以上ある場合、この\n"
            "キャラは、速攻を得る。\n"
            "アタック時ターンロ自分のアクティブのドン2枚をドシ!デッキに\n"
            "戻すことができる：このキャラをアクティブにする。その後、自分のライフ\n"
            "の上から１枚を手札に加える。"
        )
        parts = parse_effects(source)
        self.assertEqual([p.kind for p in parts],
                         ["conditional_rush", "return_active_don_cost", "set_active", "life_to_hand"])
        self.assertEqual([p.ability_index for p in parts], [0, 1, 1, 1])
        self.assertEqual(parts[1].facts["return_active_don"], 2)
        self.assertEqual(parts[3].facts["count"], 1)

    def test_ko_and_trigger(self):
        source = (
            "KO時 自分のデッキの上から1枚までを、ライフの上に加える。その後、\n"
            "相手のライフの上から1枚までを、トラッシュに凹く。\n"
            "トリガー 自分のリーダーが『CP』を含む特徴を持つ場合、カード1枚を\n"
            "引き、コスト8以下のキャラ1枚までを、持ち主のデッキの下に置く。"
        )
        parts = parse_effects(source)
        self.assertEqual([p.kind for p in parts], ["ko_life_exchange", "trigger_draw_return"])
        self.assertIn("owner's deck", parts[1].english)

    def test_unrecognized_tail_is_preserved(self):
        parts = parse_effects("このキャラをアクティブにする。未知の効果。")
        self.assertEqual([p.kind for p in parts], ["set_active", "unresolved"])
        self.assertEqual(parts[1].source, "未知の効果。")

    def test_on_play_draw_and_leader_attack_cost(self):
        source = (
            "登場時カード1枚を引く。\n"
            "ターン1回自分の『ロックス海賊団』を含む特徴を持つリーダーがア\n"
            "タックした時かアタックされた時、自分の手札1枚を捨てて発動できる。\n"
            "自分のリーダーを、このバトル中、パワー+3000。"
        )
        parts = parse_effects(source)
        self.assertEqual([p.kind for p in parts],
                         ["draw_on_play", "leader_battle_activation", "leader_battle_power"])
        self.assertEqual([p.ability_index for p in parts], [0, 1, 1])
        self.assertIn("Rocks Pirates", parts[1].english)
        self.assertEqual(parts[1].facts["trash_from_hand"], 1)
        self.assertEqual(parts[2].facts["power"], 3000)

    def test_rush_character_and_opponent_attack(self):
        source = (
            "速攻：キャラ\n"
            "相手のアタック時 ターン1回 自分の手札から『ロックス海賊団』を含む特徴を持つ"
            "カード1枚を捨てることができる：相手のキャラ1枚までを、"
            "このターン中、パワー-3000。"
        )
        parts = parse_effects(source)
        self.assertEqual([p.kind for p in parts],
                         ["rush_character_keyword", "opponent_attack_power"])
        self.assertEqual([p.ability_index for p in parts], [0, 1])
        self.assertEqual(parts[0].english, "<Rush:Character>")
        self.assertEqual(parts[1].facts["power"], -3000)
        self.assertEqual(parts[1].facts["trash_from_hand"], 1)
        self.assertIn("Give up to 1", parts[1].english)

    def test_repeated_attack_ocr_uses_complete_later_effect(self):
        source = (
            "速攻：キャラ\nアタック時 相手のアタック時ダーン1回 "
            "自分の手札から『ロックス 相手のアタック時 ターン1回 "
            "自分の手札から『ロックス海賊団』を含む特徴を持つカード1枚を"
            "捨てることができる：相手のキャラ1枚までを、このターン中、パワー-3000。"
        )
        parts = parse_effects(source)
        self.assertEqual(parts[0].kind, "rush_character_keyword")
        self.assertEqual(parts[1].kind, "opponent_attack_power")
        self.assertIn("Removed an unfinished repeated OCR prefix", parts[1].ocr_note)
        self.assertTrue(parts[1].english.startswith("[When Attacking][On Your Opponent's Attack]"))
        self.assertEqual(parts[1].facts["timing"], "when_attacking_or_opponent_attack")

    def test_incomplete_repeated_attack_is_not_discarded(self):
        parts = parse_effects("相手のアタック時『ロックス相手のアタック時未知の効果。")
        self.assertEqual(parts[0].kind, "opponent_attack_label")
        self.assertEqual(parts[-1].kind, "unresolved")

    def test_opponent_attack_label_survives_unknown_effect(self):
        parts = parse_effects("相手のアタック時ターン1回未知の効果。")
        self.assertEqual([p.kind for p in parts],
                         ["opponent_attack_label", "once_per_turn_label", "unresolved"])
        self.assertEqual(parts[0].english, "[On Your Opponent's Attack]")


if __name__ == "__main__":
    unittest.main()
