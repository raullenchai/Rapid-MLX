# SPDX-License-Identifier: Apache-2.0
"""Compose a contact sheet and silent 1080p marketing reel from local posters.

The MP4 animates still images with FFmpeg. It is not diffusion-generated video.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def font(size):
    for path in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = json.loads((Path(__file__).parent / "destinations.json").read_text())
    images = [(job, args.input / (job["id"] + ".png")) for job in jobs]
    images = [(job, path) for job, path in images if path.exists()]
    if not images:
        parser.error("No destination PNGs found")
    canvas = Image.new("RGB", (1800, 1480), "#090e19")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (48, 30), "A TRAVEL AGENCY FOR IMPOSSIBLE PLACES", font=font(48), fill="#f6f3e9"
    )
    draw.text(
        (48, 98),
        "ComfyUI + Rapid-MLX  /  Qwen-Image 2.1  /  Generated locally on a Mac",
        font=font(24),
        fill="#a4b4c8",
    )
    for i, (job, path) in enumerate(images):
        x, y = 48 + (i % 3) * 580, 166 + (i // 3) * 620
        with Image.open(path) as source:
            canvas.paste(ImageOps.fit(source.convert("RGB"), (552, 552)), (x, y))
        draw.text((x, y + 565), job["title"], font=font(23), fill="#f6f3e9")
    draw.text(
        (48, 1430),
        "RAPID-MLX  /  YOUR MAC. YOUR CREATIVE ENGINE.",
        font=font(24),
        fill="#86ebce",
    )
    canvas.save(args.output / "contact-sheet.jpg", quality=95)

    # Stage numbered frames in scratch; use argument arrays for paths and filters.
    scratch = Path("/private/tmp/demo-twitter-render")
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="render-", dir=scratch) as tmp:
        tmp = Path(tmp)
        clip_paths = []
        reel_images = sorted(images, key=lambda item: item[0]["id"] != "cloud-ocean")
        for i, (job, path) in enumerate(reel_images):
            frame = Image.new("RGB", (1920, 1080), "#090e19")
            with Image.open(path) as source:
                frame.paste(
                    ImageOps.fit(source.convert("RGB"), (1000, 1000)), (880, 40)
                )
            d = ImageDraw.Draw(frame)
            d.text((80, 115), "RAPID TRAVEL", font=font(30), fill="#86ebce")
            words, lines, line = job["title"].split(), [], ""
            for word in words:
                candidate = (line + " " + word).strip()
                if d.textlength(candidate, font=font(68)) > 740 and line:
                    lines.append(line)
                    line = word
                else:
                    line = candidate
            lines.append(line)
            for j, line in enumerate(lines):
                d.text((80, 280 + j * 85), line, font=font(68), fill="#f6f3e9")
            d.text((80, 570), job["subtitle"], font=font(26), fill="#a4b4c8")
            d.text(
                (80, 780), "Generated locally on a Mac", font=font(30), fill="#f6f3e9"
            )
            d.text((80, 840), "ComfyUI  +  Rapid-MLX", font=font(34), fill="#86ebce")
            d.text(
                (80, 905),
                "Qwen-Image 2.1  /  Batch workflow",
                font=font(25),
                fill="#a4b4c8",
            )
            d.text(
                (80, 1000),
                f"{i + 1:02d} / {len(images):02d}     rapidmlx.com",
                font=font(23),
                fill="#a4b4c8",
            )
            frame_path = tmp / f"frame-{i}.png"
            frame.save(frame_path)
            clip = tmp / f"clip-{i}.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-loop",
                    "1",
                    "-i",
                    str(frame_path),
                    "-vf",
                    "scale=2304:1296,zoompan=z='1+0.0003*on':x='iw/2-iw/zoom/2':y='ih/2-ih/zoom/2':d=90:s=1920x1080:fps=30,fade=t=in:st=0:d=0.25,fade=t=out:st=2.75:d=0.25",
                    "-t",
                    "3",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "19",
                    "-pix_fmt",
                    "yuv420p",
                    str(clip),
                ],
                check=True,
            )
            clip_paths.append(clip)
        outro = Image.new("RGB", (1920, 1080), "#090e19")
        d = ImageDraw.Draw(outro)
        d.text((120, 160), "Rapid-MLX", font=font(110), fill="#86ebce")
        d.text((120, 370), "YOUR MAC.", font=font(90), fill="#f6f3e9")
        d.text((120, 485), "YOUR CREATIVE ENGINE.", font=font(90), fill="#f6f3e9")
        d.text(
            (120, 700),
            "Local inference. Repeatable workflows.",
            font=font(42),
            fill="#a4b4c8",
        )
        d.text((120, 860), "rapidmlx.com", font=font(64), fill="#86ebce")
        outro_path, outro_clip = tmp / "outro.png", tmp / "outro.mp4"
        outro.save(outro_path)
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-loop",
                "1",
                "-i",
                str(outro_path),
                "-vf",
                "fade=t=in:st=0:d=0.25",
                "-t",
                "2",
                "-r",
                "30",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "19",
                "-pix_fmt",
                "yuv420p",
                str(outro_clip),
            ],
            check=True,
        )
        clip_paths.append(outro_clip)
        listing = tmp / "clips.txt"
        listing.write_text("".join(f"file '{clip.name}'\n" for clip in clip_paths))
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(listing),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(args.output / "travel-reel.mp4"),
            ],
            check=True,
        )
    print("Saved contact-sheet.jpg and travel-reel.mp4 (animated stills)")


if __name__ == "__main__":
    main()
