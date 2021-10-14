import pytest
from questionanswering.squad import position_labels


class TestPositionLabels:
    def test_answer_not_exists(self):
        assert (
            position_labels(
                offset_mapping=[[(0, 1), (0, 1)], [(0, 1), (0, 1)]],
                overflow_to_sample_mapping=[0, 1],
                answer_start=[-1, -1],
                answer_length=[0, 0],
            )
            == ([0, 0], [0, 0])
        )

    def test_answer_exists(self):
        assert (
            position_labels(
                offset_mapping=[
                    [(0, 0), (0, 10), (10, 20), (20, 30), (0, 0)],
                    [(0, 0), (0, 10), (10, 20), (20, 30), (0, 0)],
                    [(0, 0), (0, 10), (10, 20), (20, 30), (0, 0)],
                    [(0, 0), (0, 10), (10, 20), (20, 30), (0, 0)],
                    [(0, 0), (0, 10), (10, 20), (20, 30), (0, 0)],
                    [(0, 0), (0, 10), (10, 20), (20, 30), (0, 0)],
                ],
                overflow_to_sample_mapping=[0, 1, 2, 3, 4, 5],
                answer_start=[0, 10, 20, 0, 10, 0],
                answer_length=[10, 10, 10, 20, 20, 30],
            )
            == ([1, 2, 3, 1, 2, 1], [1, 2, 3, 2, 3, 3])
        )

    def test_answer_exists_in_multiple_windows(self):
        assert (
            position_labels(
                offset_mapping=[
                    [(0, 0), (0, 10), (10, 20), (20, 30), (0, 0)],
                    [(0, 0), (20, 30), (30, 40), (40, 50), (0, 0)],
                    [(0, 0), (0, 10), (10, 20), (20, 30), (0, 0)],
                    [(0, 0), (20, 30), (30, 40), (40, 50), (0, 0)],
                    [(0, 0), (0, 10), (10, 20), (20, 30), (0, 0)],
                    [(0, 0), (20, 30), (30, 40), (40, 50), (0, 0)],
                ],
                overflow_to_sample_mapping=[0, 0, 1, 1, 2, 2],
                answer_start=[0, 20, 40],
                answer_length=[10, 10, 10],
            )
            == ([1, 0, 3, 1, 0, 3], [1, 0, 3, 1, 0, 3])
        )

    def test_throw_error_when_end_comes_before_start(self):
        with pytest.raises(ValueError, match=r"end must not come before start"):
            position_labels(
                offset_mapping=[
                    [(0, 0), (0, 9), (10, 20), (20, 30), (0, 0)],
                ],
                overflow_to_sample_mapping=[0],
                answer_start=[0],
                answer_length=[10],
            )

    def test_throw_error_when_answer_cannot_be_found(self):
        with pytest.raises(ValueError, match=r"answer span cannot be found"):
            position_labels(
                offset_mapping=[
                    [(0, 0), (0, 10), (10, 20), (20, 30), (0, 0)],
                ],
                overflow_to_sample_mapping=[0],
                answer_start=[1],
                answer_length=[10],
            )
        with pytest.raises(ValueError, match=r"answer span cannot be found"):
            position_labels(
                offset_mapping=[
                    [(0, 0), (0, 10), (10, 20), (20, 30), (0, 0)],
                    [(0, 0), (20, 30), (30, 40), (40, 50), (0, 0)],
                ],
                overflow_to_sample_mapping=[0, 0],
                answer_start=[60],
                answer_length=[10],
            )
