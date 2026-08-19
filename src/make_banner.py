import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

W, H = 1600, 900

# ---- palette (dark theme) ----
BG_TOP = (10, 14, 23)        # near-black navy
BG_BOTTOM = (17, 24, 39)     # slightly lighter navy
INK = (241, 245, 249)        # near-white
SUBINK = (148, 163, 184)     # muted slate
ACCENT = (248, 113, 113)     # warm red, used sparingly
GOOD = (74, 222, 128)        # green for "correct"
LINE = (51, 65, 85)          # divider on dark bg
BADGE_BG = (30, 41, 59)


def load_font(size, weight="regular"):
    paths = {
        "bold": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "regular": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    }
    path = paths.get(weight, paths["regular"])
    if os.path.exists(path):
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def vgrad(w, h, top, bottom):
    base = Image.new("RGB", (1, h), color=0)
    for y in range(h):
        t = y / max(h - 1, 1)
        r = int(top[0] + (bottom[0] - top[0]) * t)
        g = int(top[1] + (bottom[1] - top[1]) * t)
        b = int(top[2] + (bottom[2] - top[2]) * t)
        base.putpixel((0, y), (r, g, b))
    return base.resize((w, h))


def crop_first_heatmap_panel(source_path):
    im = Image.open(source_path)
    w, h = im.size
    col_w = w // 4
    heat_top = int(h * 0.09 + (h * 0.91) * 0.5)
    return im.crop((10, heat_top, col_w - 10, h - 5)).convert("RGB")


def rounded_rect(draw, box, radius, fill=None, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


canvas = vgrad(W, H, BG_TOP, BG_BOTTOM)

# subtle red glow behind top-left, for depth
glow = Image.new("L", (W, H), 0)
gdraw = ImageDraw.Draw(glow)
gdraw.ellipse([-300, -400, 700, 500], fill=55)
glow = glow.filter(ImageFilter.GaussianBlur(180))
tint = Image.new("RGB", (W, H), (127, 29, 29))
canvas = Image.composite(tint, canvas, glow)
draw = ImageDraw.Draw(canvas)

MARGIN = 70

# ---- title block (left column) ----
LEFT_W = 900
f_title = load_font(56, "bold")
title_lines = ["Grad-CAM Told Me", "My Model Was Looking", "at the Right Thing."]
y = 110
line_h = 66
for line in title_lines:
    draw.text((MARGIN, y), line, font=f_title, fill=INK)
    y += line_h
draw.text((MARGIN, y), "It Was Lying.", font=f_title, fill=ACCENT)
y += line_h

# ---- subtitle ----
f_sub = load_font(23)
sub_lines = [
    "Two models. Nearly identical accuracy. Nearly identical heatmaps.",
    "Only one of them was actually looking at the shape.",
]
y += 22
for line in sub_lines:
    draw.text((MARGIN, y), line, font=f_sub, fill=SUBINK)
    y += 32

# ---- tags row ----
tags = ["TensorFlow / Keras", "Grad-CAM from scratch", "Synthetic ground truth"]
f_tag = load_font(17, "bold")
tag_y = y + 40
tx = MARGIN
for tag in tags:
    tw = draw.textlength(tag, font=f_tag)
    box = [tx, tag_y, tx + tw + 28, tag_y + 38]
    rounded_rect(draw, box, radius=19, outline=(71, 85, 105), width=1)
    draw.text((tx + 14, tag_y + 9), tag, font=f_tag, fill=(203, 213, 225))
    tx += tw + 28 + 16

# ---- right column: heatmap comparison, vertically centered, fills height ----
clean_path = os.path.join(FIG_DIR, "fig1_gradcam_clean_model.png")
diffuse_path = os.path.join(FIG_DIR, "fig7_gradcam_diffuse_shortcut.png")

panel_w, panel_h = 280, 280
gap = 70
right_col_x0 = MARGIN + LEFT_W
right_col_w = W - MARGIN - right_col_x0
total_w = panel_w * 2 + gap
start_x = right_col_x0 + (right_col_w - total_w) // 2
panel_y = 230

f_caption = load_font(18, "bold")
caption = "SAME HEATMAP METHOD"
cw = draw.textlength(caption, font=f_caption)
mid_x = start_x + panel_w + gap / 2
draw.text((mid_x - cw / 2, panel_y - 46), caption, font=f_caption, fill=(100, 116, 139))

f_label = load_font(18, "bold")
for label, path, x, color in [
    ("CORRECT REASONING", clean_path, start_x, GOOD),
    ("SPURIOUS SHORTCUT", diffuse_path, start_x + panel_w + gap, ACCENT),
]:
    panel = crop_first_heatmap_panel(path).resize((panel_w, panel_h))
    canvas.paste(panel, (x, panel_y))
    draw = ImageDraw.Draw(canvas)
    rounded_rect(draw, [x - 2, panel_y - 2, x + panel_w + 2, panel_y + panel_h + 2],
                 radius=6, outline=color, width=3)
    tw = draw.textlength(label, font=f_label)
    label_y = panel_y + panel_h + 22
    draw.text((x + panel_w / 2 - tw / 2, label_y), label, font=f_label, fill=color)

# equals sign between panels
f_eq = load_font(34, "bold")
draw.text((mid_x - 12, panel_y + panel_h / 2 - 20), "=", font=f_eq, fill=(100, 116, 139))

# ---- full-width stat bar, fills the lower dead space with real numbers ----
stats = [
    ("99.5% vs 94.8%", "test accuracy, both models"),
    ("100% / 0%", "shortcut model reliance: shortcut vs. shape"),
    ("1 of 2", "shortcuts Grad-CAM actually caught"),
]
stat_y = 700
f_stat_num = load_font(30, "bold")
f_stat_cap = load_font(15)
col_w = (W - MARGIN * 2) / len(stats)
for i, (num, cap) in enumerate(stats):
    cx = MARGIN + col_w * i
    draw.text((cx, stat_y), num, font=f_stat_num, fill=INK)
    draw.text((cx, stat_y + 44), cap, font=f_stat_cap, fill=SUBINK)
    if i > 0:
        draw.line([(cx - 30, stat_y - 6), (cx - 30, stat_y + 66)], fill=LINE, width=1)

# ---- footer ----
draw.line([(MARGIN, H - 64), (W - MARGIN, H - 64)], fill=LINE, width=1)
f_foot = load_font(17)
draw.text((MARGIN, H - 46), "zainulabideen.dev  ·  github.com/zain-ul-abideen-5036",
          font=f_foot, fill=(100, 116, 139))

canvas.save(os.path.join(FIG_DIR, "banner.png"))
print("Saved dark-theme professional banner")
