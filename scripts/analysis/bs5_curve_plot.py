"""Small dependency-free SVG/PNG line plot writer for offline scientific bundles.

Both formats use the same raw coordinates, ±sample SD band and labelled axes.
PNG is an unfiltered RGB raster with a built-in 5x7 ASCII label font.
"""

import html
import math
from pathlib import Path
import struct
import zlib

# Row bitmaps, five columns. Labels are uppercased in both renderings.
FONT = dict(
    zip(
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.-+=:/()_% ",
        [
            "0E11131519110E",
            "040C040404040E",
            "0E11010204081F",
            "1E01010601111E",
            "02060A121F0202",
            "1F10101E01011E",
            "0610101E11110E",
            "1F010204080808",
            "0E11110E11110E",
            "0E11110F01010C",
            "0E11111F111111",
            "1E11111E11111E",
            "0E11101010110E",
            "1E11111111111E",
            "1F10101E10101F",
            "1F10101E101010",
            "0E11101711110F",
            "1111111F111111",
            "0E04040404040E",
            "0702020212120C",
            "11121418141211",
            "1010101010101F",
            "111B1515111111",
            "11191513111111",
            "0E11111111110E",
            "1E11111E101010",
            "0E11111115120D",
            "1E11111E141211",
            "0F10100E01011E",
            "1F040404040404",
            "1111111111110E",
            "11111111110A04",
            "11111115151B11",
            "11110A040A1111",
            "11110A04040404",
            "1F01020408101F",
            "00000000000C0C",
            "0000001F000000",
            "0004041F040400",
            "00001F001F0000",
            "00040000040000",
            "01020204080810",
            "02040808080402",
            "08040202020408",
            "0000000000001F",
            "19190204081313",
            "00000000000000",
        ],
    )
)


def plot(path, series, *, title, ylabel, end, markers, band=None, xlabel=None, footnote=None):
    """Series: label/color/dashed/points; points are (total_step, value)."""
    path = Path(path)
    width, height = 1100, 650
    x0, x1, y0, y1 = 95, 1040, 65, 490
    values = [y for s in series for _, y in s["points"] if y is not None]
    if band:
        values += [v for b in band for _, lo, hi in b["points"] for v in (lo, hi)]
    low, high = min([0] + values), max([0.01] + values)
    high += (high - low) * 0.08
    if low < 0:
        low -= (high - low) * 0.05

    def xy(x, y):
        return x0 + (x / end) * (x1 - x0), y1 - (y - low) / (high - low) * (y1 - y0)

    pixels = bytearray(b"\xff" * (width * height * 3))
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    def rgb(color):
        return tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))

    def pixel(x, y, color):
        x, y = int(x), int(y)
        if 0 <= x < width and 0 <= y < height:
            i = (y * width + x) * 3
            pixels[i : i + 3] = bytes(color)

    def line(a, b, color="#333333", thick=1, dashed=False):
        svg.append(
            f'<line x1="{a[0]:.2f}" y1="{a[1]:.2f}" x2="{b[0]:.2f}" y2="{b[1]:.2f}" stroke="{color}" stroke-width="{thick}"'
            + (' stroke-dasharray="6 5"' if dashed else "")
            + "/>"
        )
        n = max(1, math.ceil(max(abs(b[0] - a[0]), abs(b[1] - a[1]))))
        for j in range(n + 1):
            if dashed and j % 11 >= 6:
                continue
            x, y = a[0] + (b[0] - a[0]) * j / n, a[1] + (b[1] - a[1]) * j / n
            for dx in range(-(thick // 2), thick // 2 + 1):
                for dy in range(-(thick // 2), thick // 2 + 1):
                    pixel(round(x) + dx, round(y) + dy, rgb(color))

    def text(x, y, label, size=2):
        label = label.upper()
        svg.append(
            f'<text x="{x}" y="{y + 7 * size}" font-family="monospace" font-size="{9 * size}" fill="#222">{html.escape(label)}</text>'
        )
        for k, char in enumerate(label):
            rows = FONT.get(char, FONT[" "])
            for j in range(7):
                bits = int(rows[2 * j : 2 * j + 2], 16)
                for i in range(5):
                    if bits & (1 << (4 - i)):
                        for dx in range(size):
                            for dy in range(size):
                                pixel(
                                    x + k * 6 * size + i * size + dx,
                                    y + j * size + dy,
                                    (34, 34, 34),
                                )

    text(95, 15, title)
    text(95, 43, ylabel, 1)
    for i in range(6):
        value = low + (high - low) * i / 5
        _, y = xy(0, value)
        line((x0, y), (x1, y), "#dddddd")
        text(8, int(y) - 5, f"{value:.3f}" if high < 2 else f"{value:.1f}", 1)
    for step in sorted(set([0, end] + list(range(0, end + 1, max(1, end // 10))))):
        x, _ = xy(step, 0)
        line((x, y0), (x, y1), "#eeeeee")
        text(int(x) - 15, y1 + 10, f"{step / 1000:g}K", 1)
    for step, label in markers:
        x, _ = xy(step, 0)
        line((x, y0), (x, y1), "#999999", dashed=True)
        text(int(x) + 3, y0 + 4, label, 1)
    for item in band or []:
        pts = [xy(s, hi) for s, lo, hi in item["points"]]
        pts += [xy(s, lo) for s, lo, hi in reversed(item["points"])]
        pale = tuple(round(c * 0.18 + 255 * 0.82) for c in rgb(item["color"]))
        color = "#" + "".join(f"{c:02x}" for c in pale)
        svg.append(
            '<polygon points="'
            + " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
            + f'" fill="{color}"/>'
        )
        # Same band coordinates, scanline polygon fill.
        if pts:
            for y in range(
                max(y0, int(min(y for _, y in pts))), min(y1, math.ceil(max(y for _, y in pts))) + 1
            ):
                cuts = []
                for a, b in zip(pts, pts[1:] + pts[:1]):
                    if min(a[1], b[1]) <= y < max(a[1], b[1]):
                        cuts.append(a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1]))
                cuts.sort()
                for left, right in zip(cuts[::2], cuts[1::2]):
                    for x in range(int(left), int(right) + 1):
                        pixel(x, y, pale)
    for item in series:
        pts = [xy(s, v) for s, v in item["points"] if v is not None]
        for a, b in zip(pts, pts[1:]):
            line(a, b, item["color"], item.get("width", 2), item.get("dashed", False))
        for x, y in pts:
            line((x - 2, y), (x + 2, y), item["color"], 2)
        k = series.index(item)
        lx, ly = 95 + (k % 3) * 320, 555 + (k // 3) * 27
        line((lx, ly + 5), (lx + 25, ly + 5), item["color"], 2, item.get("dashed", False))
        text(lx + 35, ly, item["label"], 1)
    line((x0, y0), (x0, y1))
    line((x0, y1), (x1, y1))
    text(300, 520, xlabel or "TOTAL ENVIRONMENT STEPS (0 TO " + str(end) + ")", 1)
    text(
        95,
        625,
        footnote or "RAW POINTS; BAND = +/-1 SAMPLE SD OF TRAINING SEEDS, N=2; NO SMOOTHING",
        1,
    )
    svg.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".svg").write_text("\n".join(svg), encoding="utf-8")

    def chunk(kind, data):
        return (
            struct.pack("!I", len(data))
            + kind
            + data
            + struct.pack("!I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + pixels[y * width * 3 : (y + 1) * width * 3] for y in range(height))
    path.with_suffix(".png").write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack("!2I5B", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )
