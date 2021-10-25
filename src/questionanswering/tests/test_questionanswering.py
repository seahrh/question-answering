import pytest
from questionanswering import dice_coefficient, preprocess


class TestDiceCoefficient:
    def test_empty_intersection(self):
        assert dice_coefficient(true_start=0, true_end=0, pred_start=1, pred_end=1) == 0
        assert dice_coefficient(true_start=1, true_end=1, pred_start=0, pred_end=0) == 0

    def test_equal_sets(self):
        assert dice_coefficient(true_start=1, true_end=1, pred_start=1, pred_end=1) == 1
        assert dice_coefficient(true_start=0, true_end=9, pred_start=0, pred_end=9) == 1

    def test_end_position_comes_before_start(self):
        assert dice_coefficient(true_start=1, true_end=0, pred_start=0, pred_end=1) == 0
        assert dice_coefficient(true_start=0, true_end=1, pred_start=1, pred_end=0) == 0
        assert dice_coefficient(true_start=1, true_end=0, pred_start=1, pred_end=0) == 0

    def test_one_is_a_subset_of_the_other(self):
        assert (
            dice_coefficient(true_start=2, true_end=3, pred_start=0, pred_end=5) == 0.5
        )
        assert (
            dice_coefficient(true_start=0, true_end=5, pred_start=2, pred_end=3) == 0.5
        )

    def test_overlapping_from_the_side(self):
        assert (
            dice_coefficient(true_start=4, true_end=7, pred_start=0, pred_end=5) == 0.4
        )
        assert (
            dice_coefficient(true_start=0, true_end=5, pred_start=4, pred_end=7) == 0.4
        )


class TestPreprocess:
    def test_compress_whitespace(self):
        assert preprocess("  foo  bar  ") == "foo bar"

    def test_remove_punctuation(self):
        assert preprocess("(a) (b) (c)") == "a b c"
        assert preprocess('"a" "b" "c"') == "a b c"
        assert preprocess("[a] [b] [c]") == "a b c"

    @pytest.mark.skip(reason="deprecated")
    def test_remove_leading_punctuation(self):
        assert preprocess(".,-:;–'\" foo") == "foo"

    @pytest.mark.skip(reason="deprecated")
    def test_remove_trailing_punctuation(self):
        assert preprocess("foo .,-:;–'\"") == "foo"

    def test_normalize_curly_quotes(self):
        # double quotes are then removed in another step
        assert preprocess("1 ‘2’ “3” 4") == "1 '2' 3 4"

    @pytest.mark.skip(reason="deprecated")
    def test_remove_enclosing_single_quotes(self):
        assert preprocess("'a1'") == "a1"
        assert preprocess("'a1' 'b2' 'c3'") == "a1 b2 c3"
        assert preprocess("tom's cat") == "tom's cat"
        assert preprocess("o'clock") == "o'clock"
        assert preprocess("'12 o'clock'") == "12 o'clock"
        assert preprocess("'tom's 12 o'clock'") == "toms 12 oclock"

    @pytest.mark.skip(reason="deprecated")
    def test_remove_repeated_quotes(self):
        assert preprocess("1 ''2''' 3") == "1 2 3"

    def test_isolate_punctuation(self):
        assert preprocess("a: b") == "a : b"
        assert preprocess("a; b") == "a ; b"
        assert preprocess("a, b") == "a , b"
        assert preprocess("a. b") == "a . b"
        assert preprocess("a $b") == "a $ b"
        assert preprocess("a% b") == "a % b"
        assert preprocess("a+b") == "a + b"
        assert preprocess("a-b") == "a - b"
        assert preprocess("a*b") == "a * b"
        assert preprocess("a/b") == "a / b"
