# Based on https://stackoverflow.com/questions/470690/how-to-automatically-generate-n-distinct-colors

import colorsys
import itertools
import math
from fractions import Fraction


def zenos_dichotomy():
    """
    http://en.wikipedia.org/wiki/1/2_%2B_1/4_%2B_1/8_%2B_1/16_%2B_%C2%B7_%C2%B7_%C2%B7
    """
    for k in itertools.count():
        yield Fraction(1, 2**k)


def getfracs():
    """
    [Fraction(0, 1), Fraction(1, 2), Fraction(1, 4), Fraction(3, 4), Fraction(1, 8), Fraction(3, 8), Fraction(5, 8), Fraction(7, 8), Fraction(1, 16), Fraction(3, 16), ...]
    [0.0, 0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875, 0.0625, 0.1875, ...]
    """
    yield 0
    for k in zenos_dichotomy():
        i = k.denominator  # [1,2,4,8,16,...]
        for j in range(1, i, 2):
            yield Fraction(j, i)


bias = lambda x: (math.sqrt(x / 3) / Fraction(2, 3) + Fraction(1, 3)) / Fraction(6, 5)


def genhsv(h):
    for s in [Fraction(6, 10)]:  # optionally use range
        for v in [Fraction(8, 10), Fraction(5, 10)]:  # could use range too
            yield (h, s, v)  # use bias for v here if you use range


genrgb = lambda x: colorsys.hsv_to_rgb(*x)

flatten = itertools.chain.from_iterable

gethsvs = lambda: flatten(map(genhsv, getfracs()))

getrgbs = lambda: map(genrgb, gethsvs())


def genhtml(x):
    uint8tuple = map(lambda y: y, x)
    uint8tuple = map(lambda x: round(float(x), 4), uint8tuple)
    return "{} {} {}".format(*uint8tuple)


gethtmlcolors = lambda: map(genhtml, getrgbs())


def get_colors(n):
    return list(itertools.islice(gethtmlcolors(), n))


if __name__ == "__main__":
    print(list(itertools.islice(gethtmlcolors(), 100)))
