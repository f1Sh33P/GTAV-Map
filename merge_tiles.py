#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
无损合并 map 目录下的 256x256 JPEG 瓦片。

原理：瓦片的量化表、采样方式(4:4:4)、尺寸全部一致，因此可以使用
libjpeg-turbo 的 jpegtran -drop，在量化后的 DCT 系数域把每张瓦片
原样拷入大画布。整个过程不做 DCT / 量化重算，拼接结果与原始瓦片
解码后逐像素完全一致（真正的无损，而不是"高质量重新编码"）。

输出：map-merged/{set}-{z}-{分辨率}.jpg，z = 0..5（分辨率 = 256 * 2^z）
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parent
MAP_DIR = ROOT / "map"
OUT_DIR = ROOT / "map-merged"
SETS = ("atlas", "roadmap", "satellite")
TILE = 256
MAX_LEVEL = 5


# ---------------------------------------------------------------- jpegtran

def find_jpegtran() -> str:
    candidates = [
        Path(r"D:\anaconda3\Library\bin\jpegtran.exe"),
        Path(sys.prefix) / "Library" / "bin" / "jpegtran.exe",
        Path(sys.prefix) / "bin" / "jpegtran.exe",
    ]
    which = shutil.which("jpegtran")
    if which:
        candidates.insert(0, Path(which))
    for p in candidates:
        if p.is_file():
            return str(p)
    raise SystemExit(
        "找不到 jpegtran.exe（需要带 -drop 支持的 libjpeg-turbo 版本）。\n"
        "已尝试: " + ", ".join(str(p) for p in candidates)
    )


JPEGTRAN = find_jpegtran()


def jpegtran(src: Path, dst: Path, *options: str) -> None:
    """调用 jpegtran；src/dst 不能是同一个文件。"""
    cmd = [
        JPEGTRAN,
        "-copy", "all",
        "-maxmemory", "1500000",
        *options,
        str(src),
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "jpegtran 失败 (%s)\n命令: %s\nstderr: %s"
            % (proc.returncode, " ".join(cmd), proc.stderr.strip())
        )


# ---------------------------------------------------------------- 画布

def reference_meta(set_dir: Path) -> dict:
    """读取参考瓦片的量化表与元数据（所有瓦片已确认完全一致）。"""
    with Image.open(set_dir / "0_0_0.jpg") as im:
        im.load()
        meta = {
            "qtables": {k: list(v) for k, v in im.quantization.items()},
            "icc_profile": im.info.get("icc_profile"),
            "exif": im.info.get("exif"),
            "xmp": im.info.get("xmp"),
        }
    return meta


def save_blank(path: Path, size: tuple[int, int], meta: dict) -> None:
    """生成与原瓦片同量化表、同采样(4:4:4)、同元数据的黑色画布。

    画布内容之后会被瓦片完全覆盖，所以颜色本身无所谓，但量化表与采样
    方式必须一致，jpegtran -drop 才不会做 requantize。
    """
    img = Image.new("RGB", size, (0, 0, 0))
    kwargs = {
        "format": "JPEG",
        "qtables": meta["qtables"],
        "subsampling": 0,
        "optimize": False,
    }
    if meta.get("icc_profile"):
        kwargs["icc_profile"] = meta["icc_profile"]
    if meta.get("exif"):
        kwargs["exif"] = meta["exif"]
    if meta.get("xmp"):
        kwargs["xmp"] = meta["xmp"]
    try:
        img.save(path, **kwargs)
    except TypeError:
        # 旧版 Pillow 不支持 xmp 关键字时退化处理
        kwargs.pop("xmp", None)
        img.save(path, **kwargs)

    with Image.open(path) as base:
        if base.size != size:
            raise RuntimeError(f"画布尺寸异常: {path} -> {base.size}")
        if base.quantization != meta["qtables"]:
            raise RuntimeError(f"画布量化表与瓦片不一致: {path}")


# ---------------------------------------------------------------- 合并

def merge_level(set_name: str, level: int, tmp: Path) -> Path:
    set_dir = MAP_DIR / set_name
    n = 1 << level
    size = TILE * n
    out_path = OUT_DIR / f"{set_name}-{level}-{size}.jpg"

    if level == 0:
        shutil.copy2(set_dir / "0_0_0.jpg", out_path)
        return out_path

    meta = reference_meta(set_dir)

    # 1) 逐行拼出行条（先在小图上拼，避免上千次重写大画布）
    strips = []
    for y in range(n):
        blank = tmp / f"strip_{y}_a.jpg"
        save_blank(blank, (size, TILE), meta)
        cur = blank
        for x in range(n):
            nxt = tmp / f"strip_{y}_{'b' if x % 2 == 0 else 'a'}.jpg"
            jpegtran(cur, nxt, "-drop", f"+{x * TILE}+0",
                     str(set_dir / f"{level}_{x}_{y}.jpg"))
            cur = nxt
        strip = tmp / f"strip_{y}.jpg"
        if cur != strip:
            os.replace(cur, strip)
        strips.append(strip)

    # 2) 把行条拼进大画布
    blank = tmp / "big_a.jpg"
    save_blank(blank, (size, size), meta)
    cur = blank
    for y, strip in enumerate(strips):
        nxt = tmp / f"big_{'b' if y % 2 == 0 else 'a'}.jpg"
        jpegtran(cur, nxt, "-drop", f"+0+{y * TILE}", str(strip))
        cur = nxt
    if cur != blank:
        os.replace(cur, tmp / "big_final.jpg")
        cur = tmp / "big_final.jpg"

    # 3) 只优化 Huffman 编码（不改变 DCT 系数，仍然无损），写出最终文件
    jpegtran(cur, out_path, "-optimize")

    # 4) 清理本层的中间文件
    for p in (tmp / "big_a.jpg", tmp / "big_b.jpg", tmp / "big_final.jpg"):
        p.unlink(missing_ok=True)
    for p in tmp.glob("strip_*.jpg"):
        p.unlink(missing_ok=True)

    return out_path


# ---------------------------------------------------------------- 校验

def verify_level(set_name: str, level: int, out_path: Path) -> None:
    """逐块比对：成品中每个 256x256 区域必须与原瓦片解码像素完全一致。"""
    set_dir = MAP_DIR / set_name
    n = 1 << level
    size = TILE * n

    with Image.open(out_path) as im:
        if im.size != (size, size):
            raise AssertionError(f"{out_path.name} 尺寸错误: {im.size} != {(size, size)}")
        with Image.open(set_dir / "0_0_0.jpg") as ref:
            if im.quantization != ref.quantization:
                raise AssertionError(f"{out_path.name} 量化表与原始瓦片不一致")

        if level == 0:
            if (set_dir / "0_0_0.jpg").read_bytes() != out_path.read_bytes():
                raise AssertionError("level 0 不是原始文件的完整拷贝")
            print(f"  {out_path.name}: 字节级复制 OK")
            return

        merged = im.convert("RGB")
        merged.load()  # 一次性解码到内存，之后 crop 很快

        bad = []
        for y in range(n):
            for x in range(n):
                with Image.open(set_dir / f"{level}_{x}_{y}.jpg") as tile:
                    tile = tile.convert("RGB")
                    tile.load()
                region = merged.crop(
                    (x * TILE, y * TILE, (x + 1) * TILE, (y + 1) * TILE)
                )
                if ImageChops.difference(region, tile).getbbox() is not None:
                    bad.append((x, y))
        total = n * n
        if bad:
            raise AssertionError(
                f"{out_path.name}: {len(bad)}/{total} 块与原瓦片不一致，例如 {bad[:5]}"
            )
        print(
            f"  {out_path.name}: {total}/{total} 块逐像素一致 OK  "
            f"({size}x{size}, {out_path.stat().st_size / 1048576:.1f} MB)"
        )




# ---------------------------------------------------------------- main

def main() -> None:
    if not MAP_DIR.is_dir():
        raise SystemExit(f"找不到目录: {MAP_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"jpegtran: {JPEGTRAN}")
    print(f"输出目录: {OUT_DIR}\n")

    tmp = Path(tempfile.mkdtemp(prefix="gta-merge-", dir=OUT_DIR))
    try:
        for set_name in SETS:
            if not (MAP_DIR / set_name / "0_0_0.jpg").is_file():
                raise SystemExit(f"找不到瓦片集: {MAP_DIR / set_name}")
            print(f"[{set_name}]")
            for level in range(MAX_LEVEL + 1):
                t0 = time.time()
                out_path = merge_level(set_name, level, tmp)
                print(f"  已生成 {out_path.name}  ({time.time() - t0:.1f}s)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n校验中...")
    for set_name in SETS:
        print(f"[{set_name}]")
        for level in range(MAX_LEVEL + 1):
            size = TILE * (1 << level)
            verify_level(set_name, level, OUT_DIR / f"{set_name}-{level}-{size}.jpg")

    print("\n全部完成：所有成品均与原瓦片逐像素一致（无损）。")


if __name__ == "__main__":
    main()
