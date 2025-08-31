def action_sum(xs):
    total = 0.0
    for i in range(1, len(xs)):
        total += 1.0
    return total


def is_deceptive(xs, mult=1.3):
    a = action_sum(xs)
    return a > mult, a