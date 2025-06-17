"""Slot Machine Wallet Tracking Example.

This script extracts the wallet balance and bet amount from a slot machine video
and records the values after each spin. The results are saved to a CSV file.

Usage:
    python slot_machine_wallet_tracker.py <video_path> [output_csv]

Adjust the ROI (region of interest) coordinates for your game layout using the
--wallet-roi and --bet-roi options if needed.
"""

import argparse
import csv
import re
from pathlib import Path

import cv2
import pytesseract


def _parse_amount(text: str) -> float | None:
    """Extract a numeric value from OCR text."""
    cleaned = text.replace(",", "")
    match = re.search(r"([0-9]+\.?[0-9]*)", cleaned)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _extract_value(frame, roi) -> float | None:
    x, y, w, h = roi
    crop = frame[y : y + h, x : x + w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(gray, config="--psm 7")
    return _parse_amount(text)


def analyze_slot_machine(video_path: Path, output_csv: Path, wallet_roi, bet_roi) -> None:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = int(fps) if fps > 0 else 1
    frame_count = 0
    spin_index = 0
    last_wallet = None
    rows: list[dict[str, float | str]] = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            wallet_val = _extract_value(frame, wallet_roi)
            bet_val = _extract_value(frame, bet_roi)

            if wallet_val is not None:
                if last_wallet is None or wallet_val != last_wallet:
                    spin_index += 1
                    rows.append(
                        {
                            "spin": spin_index,
                            "timestamp": f"{timestamp:.2f}",
                            "wallet": wallet_val,
                            "bet": bet_val if bet_val is not None else "",
                        }
                    )
                    last_wallet = wallet_val

        frame_count += 1

    cap.release()

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["spin", "timestamp", "wallet", "bet"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\N{check mark} Results written to {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track wallet balance from a slot machine video")
    parser.add_argument("video_path", type=str, help="Path to video file")
    parser.add_argument(
        "output_csv",
        type=str,
        nargs="?",
        default="wallet_history.csv",
        help="Output CSV path (default: wallet_history.csv)",
    )
    parser.add_argument(
        "--wallet-roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        default=[50, 50, 200, 50],
        help="Wallet balance region: x y w h",
    )
    parser.add_argument(
        "--bet-roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        default=[50, 110, 200, 50],
        help="Bet amount region: x y w h",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video_path)
    output_csv = Path(args.output_csv)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    analyze_slot_machine(video_path, output_csv, tuple(args.wallet_roi), tuple(args.bet_roi))


if __name__ == "__main__":
    main()
