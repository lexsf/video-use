import cv2
import pytesseract
import csv
import re
from pathlib import Path
import argparse


def extract_amount(text: str, keywords: list[str]) -> float | None:
    pattern = re.compile(rf"(?:{'|'.join(keywords)})[:\s\$]*([0-9,.]+)", re.I)
    match = pattern.search(text)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


def analyze_slot_video(video_path: Path, fps: float = 1.0) -> list[dict[str, float]]:
    """Analyze slot machine video and return wallet amounts after each spin."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(int(video_fps / fps), 1)

    results = []
    last_wallet = None
    last_bet = None
    spin = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)

            wallet = extract_amount(text, ["wallet", "balance", "credits"])
            bet = extract_amount(text, ["bet", "stake", "wager"])

            if wallet is not None:
                last_wallet = wallet
            if bet is not None:
                last_bet = bet

            if last_wallet is not None and last_bet is not None:
                if not results or last_wallet != results[-1]["wallet"]:
                    spin += 1
                    results.append(
                        {"spin": spin, "bet": last_bet, "wallet": last_wallet}
                    )
        frame_idx += 1

    cap.release()
    return results


def save_results_csv(results: list[dict[str, float]], output_path: Path) -> None:
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["spin", "bet", "wallet"])
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze slot machine video")
    parser.add_argument("video", type=Path, help="Path to slot machine video")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("slot_results.csv"),
        help="Output CSV file",
    )
    parser.add_argument(
        "--fps", type=float, default=1.0, help="Frames per second to analyze"
    )
    args = parser.parse_args()

    results = analyze_slot_video(args.video, args.fps)
    save_results_csv(results, args.output)
    print(f"Saved {len(results)} spins to {args.output}")


if __name__ == "__main__":
    main()
