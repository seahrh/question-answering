from questionanswering import dice_coefficient


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
