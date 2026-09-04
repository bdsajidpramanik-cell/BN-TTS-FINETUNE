"""
ধাপ ১: SLR37 (Bangladeshi Bengali TTS) ডেটা ডাউনলোড + প্রিপ্রসেসিং

এই স্ক্রিপ্ট চালাতে হবে GPU মেশিনে (Colab/RunPod), Claude-এর sandbox-এ না,
কারণ এখানে নেটওয়ার্ক এক্সেস বন্ধ।

আউটপুট স্ট্রাকচার (Coqui TTS / LJSpeech ফরম্যাট):
    <output_dir>/
        wavs/
            0001.wav
            0002.wav
            ...
        metadata.csv   (ফরম্যাট: id|transcript|transcript)
"""

import argparse
import csv
import os
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import soundfile as sf
import librosa

# OpenSLR SLR37 — Bengali (Bangladesh) high quality TTS data
# রেফারেন্স: https://openslr.org/37/
SLR37_URL = "https://openslr.elda.org/resources/37/bn_bd_female.zip"
TARGET_SR = 22050


def download_slr37(dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "bn_bd_female.zip"
    if zip_path.exists():
        print(f"ইতিমধ্যে ডাউনলোড করা আছে: {zip_path}")
        return zip_path

    print(f"ডাউনলোড হচ্ছে: {SLR37_URL}")
    subprocess.run(["wget", "-O", str(zip_path), SLR37_URL], check=True)
    return zip_path


def extract_archive(archive_path: Path, extract_to: Path) -> Path:
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"এক্সট্র্যাক্ট হচ্ছে: {archive_path} -> {extract_to}")
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_to)
    elif archive_path.suffixes[-2:] == [".tar", ".gz"]:
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(extract_to)
    else:
        raise ValueError(f"অজানা আর্কাইভ ফরম্যাট: {archive_path}")
    return extract_to


def find_index_file(raw_dir: Path) -> Path:
    """OpenSLR TTS ডেটাসেটে সাধারণত line_index.tsv বা line_index.txt থাকে,
    ফরম্যাট: <wav_id>\\t<transcript>"""
    for candidate in ["line_index.tsv", "line_index.txt", "index.tsv"]:
        matches = list(raw_dir.rglob(candidate))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"'{raw_dir}' এর ভেতরে line_index.tsv/txt পাওয়া যায়নি। "
        "ম্যানুয়ালি চেক করে --index_file দিয়ে পাথ বলো।"
    )


def resample_and_save(src_wav: Path, dst_wav: Path, target_sr: int = TARGET_SR):
    audio, sr = librosa.load(str(src_wav), sr=None, mono=True)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    sf.write(str(dst_wav), audio, target_sr, subtype="PCM_16")


def build_ljspeech_dataset(
    raw_dir: Path,
    index_file: Path,
    output_dir: Path,
    speaker_id: str = None,
):
    wavs_out = output_dir / "wavs"
    wavs_out.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = 0
    with open(index_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            wav_id, transcript = parts[0], parts[1]

            if speaker_id and speaker_id not in wav_id:
                continue

            # OpenSLR TTS সেটে wav ফাইল সাধারণত wavs/ বা root-এ থাকে
            candidates = list(raw_dir.rglob(f"{wav_id}.wav"))
            if not candidates:
                skipped += 1
                continue

            src_wav = candidates[0]
            dst_wav = wavs_out / f"{wav_id}.wav"
            resample_and_save(src_wav, dst_wav)
            rows.append((wav_id, transcript))

    metadata_path = output_dir / "metadata.csv"
    with open(metadata_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        for wav_id, transcript in rows:
            # LJSpeech ফরম্যাট: id|raw_text|normalized_text
            writer.writerow([wav_id, transcript, transcript])

    print(f"মোট {len(rows)}টা utterance প্রসেস হয়েছে, {skipped}টা স্কিপ হয়েছে (wav পাওয়া যায়নি)।")
    print(f"metadata.csv লেখা হয়েছে: {metadata_path}")
    return metadata_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--raw_dir", type=str, default=None,
                         help="যদি ম্যানুয়ালি ডাউনলোড করা থাকে, raw ফোল্ডারের পাথ দাও")
    parser.add_argument("--skip_download", action="store_true")
    parser.add_argument("--index_file", type=str, default=None)
    parser.add_argument("--speaker_id", type=str, default=None,
                         help="Multi-speaker ডেটাসেট হলে নির্দিষ্ট speaker filter করতে")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    work_dir = output_dir / "_raw"

    if args.raw_dir:
        raw_dir = Path(args.raw_dir)
    elif args.skip_download:
        raise ValueError("--skip_download দিলে --raw_dir বাধ্যতামূলক")
    else:
        archive = download_slr37(work_dir)
        raw_dir = extract_archive(archive, work_dir / "extracted")

    index_file = Path(args.index_file) if args.index_file else find_index_file(raw_dir)

    build_ljspeech_dataset(raw_dir, index_file, output_dir, speaker_id=args.speaker_id)


if __name__ == "__main__":
    main()
