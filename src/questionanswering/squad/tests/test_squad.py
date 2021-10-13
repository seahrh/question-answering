from questionanswering.squad import position_labels


class TestPositionLabels:
    def test_answer_not_exists(self):
        assert (
            position_labels(
                offset_mapping=[[(0, 1), (0, 1)], [(0, 1), (0, 1)]],
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
                answer_start=[0, 10, 20, 0, 10, 0],
                answer_length=[10, 10, 10, 20, 20, 30],
            )
            == ([1, 2, 3, 1, 2, 1], [1, 2, 3, 2, 3, 3])
        )
