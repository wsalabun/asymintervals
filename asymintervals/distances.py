import numpy as np
from .asymintervals import AIN

__all__ = ['w1', 'w2', 'winf']

def _calculate_AB_intervals(x: AIN, y: AIN) -> tuple[tuple[float, float, float, float], ...]:
    # Both degenerate
    if abs(x.a - x.b) < 1e-14 and abs(y.a - y.b) < 1e-14:
        return ((0.0, 1.0, x.c - y.c, 0.0),)

    # X degenerate
    if abs(x.a - x.b) < 1e-14:
        t2 = y.alpha * (y.c - y.a)
        return (
            (0.0, t2, x.c - y.a, -1 / y.alpha),
            (t2, 1.0, x.c - y.c + t2 / y.beta, -1 / y.beta)
        )

    # Y degenerate
    if abs(y.a - y.b) < 1e-14:
        t1 = x.alpha * (x.c - x.a)
        return (
            (0.0, t1, x.a - y.c, 1 / x.alpha),
            (t1, 1.0, x.c - y.c - t1 / x.beta, 1 / x.beta)
        )

    t1 = x.alpha * (x.c - x.a)
    t2 = y.alpha * (y.c - y.a)

    # For consistency, ensure that t1 <= t2. If not, swap the intervals.
    if t1 > t2:
        x, y = y, x
        t1, t2 = t2, t1

    # Calculate values Ai and Bi for each subinterval [0, t1], [t1, t2], and [t2, 1]
    # Each quartet contains (start, end, Ai, Bi) for the respective subinterval
    ab_values = (
        (0, t1, x.a - y.a, 1 / x.alpha - 1 / y.alpha),
        (t1, t2, x.c - y.a - t1 / x.beta, 1 / x.beta - 1 / y.alpha),
        (t2, 1, x.c - y.c - t1 / x.beta + t2 / y.beta, 1 / x.beta - 1 / y.beta)
    )

    return ab_values


def w1(x: AIN, y: AIN) -> float:
    ab_values = _calculate_AB_intervals(x, y)

    # Calculate the distance using the formula for w1
    distance = 0
    for pi, ri, Ai, Bi in ab_values:
        sigma = (Ai + Bi * pi) * (Ai + Bi * ri)
        if sigma >= 0:
            Ii = abs(Ai * (ri - pi) + Bi/2 * (ri**2 - pi**2))
        else:
            q0 = -Ai / Bi
            Ii = abs(Ai * (q0 - pi) + Bi/2 * (q0**2 - pi**2))\
                + abs(Ai * (ri - q0) + Bi/2 * (ri**2 - q0**2))
        distance += Ii

    return distance


def w2(x: AIN, y: AIN) -> float:
    ab_values = _calculate_AB_intervals(x, y)

    # Calculate the distance using the formula for w2
    distance = 0
    for pi, ri, Ai, Bi in ab_values:
        Ji = Ai**2 * (ri - pi) + Ai * Bi * (ri**2 - pi**2) + Bi**2 / 3 * (ri**3 - pi**3)
        distance += Ji

    return np.sqrt(distance)


def winf(x: AIN, y: AIN) -> float:
    ab_values = _calculate_AB_intervals(x, y)

    # Calculate the distance using the formula for winf
    # D(0) and D(1) are the distances at the endpoints of the interval [0, 1]
    distance = [abs(x.a - y.a), abs(x.b - y.b)]
    # D(t1) and D(t2) are the distances at the points t1 and t2
    # Starting with 1, we have A2 + B2 * t1 and A3 + B3 * t2
    for pi, _, Ai, Bi in ab_values[1:]:
        d = Ai + Bi * pi
        distance.append(abs(d))

    return max(distance)