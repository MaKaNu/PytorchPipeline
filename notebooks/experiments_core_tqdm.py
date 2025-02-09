import time

from tqdm import tqdm


def rgb2hex(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"


def create_bar(color, total):
    return tqdm(range(total), leave=False, colour=rgb2hex(*color))


active_bars = list(range(10))

N = len(active_bars)
colors = [
    (
        # Red channel
        int(255 * (2 * t if t <= 0.5 else 1)),
        # Green channel
        int(255 * (2 * t if t <= 0.5 else 1 - 2 * (t - 0.5))),
        # Blue channel
        int(255 * (1 if t <= 0.5 else 1 - 2 * (t - 0.5))),
    )
    for i in range(N)
    for t in [i / (N - 1)]
]


a = create_bar(colors[0], 2)
for _ in a:
    b = create_bar(colors[1], 2)
    for _ in b:
        c = create_bar(colors[2], 2)
        for _ in c:
            d = create_bar(colors[3], 2)
            for _ in d:
                e = create_bar(colors[4], 2)
                for _ in e:
                    f = create_bar(colors[5], 2)
                    for _ in f:
                        g = create_bar(colors[6], 2)
                        for _ in g:
                            h = create_bar(colors[7], 2)
                            for _ in h:
                                i = create_bar(colors[8], 2)
                                for _ in i:
                                    j = create_bar(colors[9], 5)
                                    for _ in j:
                                        time.sleep(0.01)
    # for _ in active_bars[2]:
