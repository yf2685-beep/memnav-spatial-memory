"""Render the README's conceptual MemNav walkthrough (Pillow only).

Synthetic route and illustrative retrieval: no model inference or benchmark data.
Run from any directory; output is relative to this script's repository root.
"""
from pathlib import Path
import argparse
import math
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BG, PANEL, LINE = '#0b1220', '#141f31', '#293951'
WHITE, MUTED = '#ecf3ff', '#a5b7ce'
CYAN, GOLD, PURPLE = '#53dccc', '#ffce73', '#b8a2ff'


def font(size, bold=False, supplied=None):
    candidates = [supplied] if supplied else []
    candidates += [
        '/System/Library/Fonts/Supplemental/Arial' + (' Bold' if bold else '') + '.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans' + ('-Bold' if bold else '') + '.ttf',
        'C:/Windows/Fonts/arial' + ('bd' if bold else '') + '.ttf',
    ]
    for p in candidates:
        if p and Path(p).is_file():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


def draw_frame(index, total, font_path=None):
    im = Image.new('RGB', (1000, 640), BG)
    d = ImageDraw.Draw(im)
    f = {s: font(s, s >= 22, font_path) for s in (14, 16, 18, 22, 32)}
    phase = min(3, index // (total // 4))
    q = (index % (total // 4)) / (total // 4 - 1)
    titles = ['01  OBSERVE & REMEMBER', '02  RETRIEVE A PLACE',
              '03  WARM THE GOAL CONTEXT', '04  CONDITION THE PLANNER']
    notes = [
        'A frozen streaming backbone carries visual context between frames.',
        'Goal-image descriptors are matched against candidate history frames.',
        'Replay context near the selected anchor, then append the goal image.',
        'Current geometry + relative pose + visual tokens condition diffusion.',
    ]
    d.text((32, 23), 'MemNav / Spatial Memory', font=f[32], fill=WHITE)
    d.text((33, 66), 'IMAGE-GOAL NAVIGATION  /  STREAMING KV CACHE', font=f[14], fill=CYAN)
    d.rounded_rectangle((749, 27, 968, 61), radius=9, fill=LINE)
    d.text((764, 36), 'CONCEPTUAL ANIMATION', font=f[14], fill=GOLD)

    d.rounded_rectangle((32, 102, 462, 454), radius=16, fill=PANEL)
    d.text((51, 119), 'Synthetic observation route', font=f[18], fill=WHITE)
    # A hand-authored route through a diagram, not an evaluated robot trajectory.
    route = [(75, 398), (75, 313), (162, 313), (162, 205),
             (278, 205), (278, 380), (407, 380), (407, 237)]
    for x in range(58, 441, 24):
        for y in range(170, 438, 24):
            d.ellipse((x, y, x + 1, y + 1), fill=LINE)
    for box in [(103, 172, 130, 276), (195, 249, 245, 400), (314, 171, 365, 327)]:
        d.rounded_rectangle(box, radius=7, fill='#26374a')
    d.line(route, fill=LINE, width=3)
    progress = q if phase == 0 else 1.0
    at = progress * (len(route) - 1)
    seg = min(int(at), len(route) - 2)
    frac = at - seg
    a, b = route[seg:seg + 2]
    pos = (a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac)
    visited = route[:seg + 1] + [pos]
    d.line(visited, fill=CYAN, width=4)
    for x, y in visited[:-1]:
        d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=CYAN)
    px, py = pos
    d.ellipse((px - 9, py - 9, px + 9, py + 9), fill=WHITE)
    d.text((53, 425), 'Route is schematic; no navigation score implied.', font=f[14], fill=MUTED)
    if phase > 0:
        gx, gy = route[3]
        radius = 12 + int(4 * math.sin(index * .3))
        d.ellipse((gx - radius, gy - radius, gx + radius, gy + radius), outline=GOLD, width=3)
        d.text((gx + 21, gy - 10), 'goal / anchor m', font=f[14], fill=GOLD)
        d.text((310, 205), 'current k', font=f[14], fill=WHITE)

    d.rounded_rectangle((482, 102, 968, 454), radius=16, fill=PANEL)
    d.text((504, 119), 'Three memory tiers', font=f[18], fill=WHITE)
    rows = [(159, CYAN, 'Scale frames', '8 frames / full patch KV'),
            (229, PURPLE, 'Older history', '6 special tokens / frame'),
            (299, GOLD, 'Recent window', '32 frames / full patch KV')]
    for row, (y, color, name, desc) in enumerate(rows):
        d.text((504, y), name, font=f[16], fill=WHITE)
        d.text((680, y), desc, font=f[14], fill=MUTED)
        fill_count = int(12 * q) if phase == 0 else 12
        if row == 0:
            fill_count = min(12, int(48 * q)) if phase == 0 else 12
        for j in range(12):
            x = 506 + j * 36
            d.rounded_rectangle((x, y + 25, x + 29, y + 42), radius=4,
                                fill=color if j < fill_count else LINE)
    detail = [
        'Older special-token history still grows over time.',
        'DINO descriptors -> retrieval rank + revisit gate',
        'Default warm replay: up to 64 preceding frames',
        'Diffusion decoder -> 24 local waypoints',
    ][phase]
    d.text((504, 377), detail, font=f[16], fill=WHITE)
    d.text((504, 409), 'Defaults shown: training config; chips are illustrative.', font=f[14], fill=MUTED)

    d.text((33, 477), titles[phase], font=f[22], fill=CYAN)
    d.text((33, 514), notes[phase], font=f[18], fill=WHITE)
    labels = ['RGB stream', 'KV memory', 'Goal retrieval', 'Pose + tokens', 'Waypoints']
    for j, label in enumerate(labels):
        x = 33 + j * 190
        active = (phase == 0 and j < 2) or (phase == 1 and j == 2) or (phase == 2 and j == 3) or (phase == 3 and j >= 3)
        d.rounded_rectangle((x, 552, x + 175, 591), radius=9, fill=LINE if active else PANEL,
                            outline=CYAN if active else LINE, width=2)
        d.text((x + 13, 563), label, font=f[16], fill=WHITE if active else MUTED)
        if j < 4:
            d.text((x + 178, 564), '>', font=f[14], fill=CYAN)
    d.text((33, 610), 'Illustration of code structure, not a model rollout or a latency / memory benchmark.', font=f[14], fill=MUTED)
    return im


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/assets/memory-navigation.gif')
    parser.add_argument('--font', help='Optional TrueType font for portable rendering')
    args = parser.parse_args()
    frames = [draw_frame(i, 100, args.font) for i in range(100)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:],
                   duration=110, loop=0, optimize=True, disposal=2)
    print(f'Wrote {args.output} ({args.output.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
