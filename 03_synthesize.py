"""
ধাপ ৩: fine-tuned চেকপয়েন্ট দিয়ে টেস্ট সিন্থেসিস

উদাহরণ:
    python 03_synthesize.py \
        --checkpoint ./runs/bn_bd_vits_finetune-*/best_model.pth \
        --config ./runs/bn_bd_vits_finetune-*/config.json \
        --text "আপনাকে কীভাবে সহযোগিতা করতে পারি?" \
        --output test.wav
"""

import argparse

from TTS.utils.synthesizer import Synthesizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--text", required=True, type=str)
    parser.add_argument("--output", default="test.wav", type=str)
    args = parser.parse_args()

    synthesizer = Synthesizer(
        tts_checkpoint=args.checkpoint,
        tts_config_path=args.config,
        use_cuda=True,
    )

    wav = synthesizer.tts(args.text)
    synthesizer.save_wav(wav, args.output)
    print(f"সিন্থেসিস শেষ, সেভ হয়েছে: {args.output}")
    print("এবার শুনে natural-ness, উচ্চারণ, আর pause যাচাই করো।")


if __name__ == "__main__":
    main()
