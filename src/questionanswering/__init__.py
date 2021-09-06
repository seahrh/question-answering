__all__ = ["dice_coefficient"]

# log = get_logger()


def dice_coefficient(
    true_start: int, true_end: int, pred_start: int, pred_end: int
) -> float:
    # If end position is before start, consider as zero length
    t_len = max(0, true_end - true_start + 1)
    p_len = max(0, pred_end - pred_start + 1)
    if t_len == 0 or p_len == 0:
        return 0
    intersection = set(range(true_start, true_end + 1)) & set(
        range(pred_start, pred_end + 1)
    )
    return 2 * len(intersection) / (t_len + p_len)
