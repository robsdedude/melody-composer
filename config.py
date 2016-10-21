import math

__author__ = 'rouven'

MIN_LENGTH = "128th triplet"  # also "8th" or "8th triplet" would be possible


# calculating MAX_LENGTH_DENOMINATOR from MIN_LENGTH
def calc_max_length_denominator():
    try:
        ml = MIN_LENGTH.split()
        assert 2 >= len(ml) >= 1
        b = int(ml[0].split("th")[0])
        if b < 4 or math.log(b, 2) % 1 != 0:
            raise ValueError("Note value must be > 4 and a power of 2")
        if len(ml) == 2:
            if ml[1] == "triplet":
                t = 3
            else:
                raise ValueError("Second word must be 'triplet'. Found '%s'."
                                 % ml[1])
        else:
            t = 2
        return b/8*t  # if not triplet b/4 if triplet b/4 / 2*3
    except Exception as e:
        raise ValueError("MIN_LENGTH must be of form '\d+th( triplet)?'", e)

MAX_LENGTH_DENOMINATOR = calc_max_length_denominator()
