#!/usr/bin/env python
"""将 translation JSON 中嵌入的 mask_raw 导出为独立 PNG 文件。

使用方式示例：

python tools/export_masks.py \
    --input OSQC07470_translations.json \
    --mask-dir-name mask \
    --backup

默认行为：
* 读取 JSON 中的每个图片条目。
* 如果存在 base64 字符串字段 mask_raw，则解码并写入 PNG 文件。
* PNG 存放在 JSON 同级目录下的 {mask-dir-name}/ 目录中。
* JSON 中新增/更新 mask_file 字段，保存相对路径（例如 mask/foo_mask.png），并移除 mask_raw。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Tuple


def _slugify(value: str) -> str:
    value = value.strip().replace("\\", "/")
    if not value:
        return "mask"
    # 仅保留常见安全字符
    value = re.sub(r"[^0-9A-Za-z._-]+", "_", value)
    value = value.strip("._")
    return value or "mask"


def _ensure_mask_filename(base_slug: str, mask_dir: Path) -> Tuple[str, Path]:
    base_name = f"{base_slug}_mask.png"
    candidate = mask_dir / base_name
    if not candidate.exists():
        return base_name, candidate

    # 若已存在，则附加序号
    idx = 1
    while True:
        new_name = f"{base_slug}_mask_{idx:02d}.png"
        candidate = mask_dir / new_name
        if not candidate.exists():
            return new_name, candidate
        idx += 1


def _decode_mask(mask_raw: str) -> bytes:
    try:
        return base64.b64decode(mask_raw)
    except Exception as exc:  # pragma: no cover - 极少发生
        raise RuntimeError(f"无法解码 mask_raw：{exc}") from exc


def process_json_file(
    json_path: Path,
    mask_dir_name: str = "mask",
    force: bool = False,
    dry_run: bool = False,
    backup: bool = False,
) -> Tuple[int, int]:
    """处理单个 JSON 文件，返回 (转换条目数量, 已存在条目数量)。"""

    if not json_path.exists():
        raise FileNotFoundError(f"未找到 JSON 文件: {json_path}")

    with json_path.open("r", encoding="utf-8") as fh:
        try:
            data: Dict[str, Any] = json.load(fh)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"JSON 解析失败: {json_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"JSON 格式非法（顶层应为对象）: {json_path}")

    json_dir = json_path.parent
    mask_dir = json_dir / mask_dir_name
    converted = 0
    skipped = 0

    if not dry_run:
        mask_dir.mkdir(parents=True, exist_ok=True)

    for image_key, image_data in data.items():
        if not isinstance(image_data, dict):
            continue

        mask_raw = image_data.get("mask_raw")
        mask_file = image_data.get("mask_file")

        if not isinstance(mask_raw, str):
            if mask_file:
                skipped += 1
            continue

        if mask_file and not force:
            skipped += 1
            continue

        slug = _slugify(Path(image_key).stem)
        filename, mask_path = _ensure_mask_filename(slug, mask_dir)
        mask_rel_path = f"{mask_dir_name}/{filename}".replace("\\", "/")

        if not dry_run:
            mask_bytes = _decode_mask(mask_raw)
            with mask_path.open("wb") as out_f:
                out_f.write(mask_bytes)

        image_data.pop("mask_raw", None)
        image_data["mask_file"] = mask_rel_path
        converted += 1

    if converted and not dry_run:
        if backup:
            backup_path = json_path.with_suffix(json_path.suffix + ".bak")
            shutil.copy2(json_path, backup_path)
        with json_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=4, ensure_ascii=False)

    return converted, skipped


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 JSON 内嵌的 mask_raw 导出为 PNG 文件")
    parser.add_argument(
        "--input",
        "-i",
        action="append",
        required=True,
        help="需要转换的 JSON 文件路径，可重复指定",
    )
    parser.add_argument(
        "--mask-dir-name",
        default="mask",
        help="在 JSON 同级目录下用于存放 PNG 的子目录名称 (默认: mask)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="已有 mask_file 时仍强制重新生成",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅统计将要转换的条目，不写入文件",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="写回 JSON 前生成 .bak 备份",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    total_converted = 0
    total_skipped = 0

    for input_path in args.input:
        path = Path(input_path)
        try:
            converted, skipped = process_json_file(
                path,
                mask_dir_name=args.mask_dir_name,
                force=args.force,
                dry_run=args.dry_run,
                backup=args.backup,
            )
        except Exception as exc:  # pragma: no cover - CLI 场景
            print(f"[ERROR] {path}: {exc}", file=sys.stderr)
            return 1

        print(f"[OK] {path}: 导出 {converted} 条, 跳过 {skipped} 条")
        total_converted += converted
        total_skipped += skipped

    print(f"完成：共导出 {total_converted} 条，跳过 {total_skipped} 条。")
    if args.dry_run and total_converted > 0:
        print("(dry-run 模式：未写入文件)")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI 入口
    raise SystemExit(main())
