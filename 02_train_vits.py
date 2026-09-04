"""
ধাপ ২: VITS fine-tuning (Coqui TTS ব্যবহার করে)

চালাতে হবে GPU মেশিনে। ডেটা আগে থেকে ধাপ ১-এর আউটপুট (LJSpeech ফরম্যাট) হতে হবে:
    <data_path>/
        wavs/*.wav
        metadata.csv
"""

import argparse
import os

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# বাংলা ক্যারেক্টার সেট — স্বরবর্ণ, ব্যঞ্জনবর্ণ, matra, সংখ্যা, যতিচিহ্ন
# প্রয়োজনে এখানে আরও ইউনিকোড ক্যারেক্টার যোগ করা যাবে
BENGALI_CHARACTERS = (
    "ঀঁংঃঅআইঈউঊঋঌএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ"
    "ঽািীুূৃৄেৈোৌ্ৎৗড়ঢ়য়০১২৩৪৫৬৭৮৯"
)
PUNCTUATION = " .,!?।-–—()\"'‘’“”:;০১২৩৪৫৬৭৮৯"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--restore_path", type=str, default=None,
                         help="Pretrained multilingual checkpoint (ঐচ্ছিক কিন্তু recommended)")
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train="metadata.csv",
        path=args.data_path,
    )

    audio_config = VitsAudioConfig(
        sample_rate=22050,
        win_length=1024,
        hop_length=256,
        num_mels=80,
        mel_fmin=0,
        mel_fmax=None,
    )

    config = VitsConfig(
        audio=audio_config,
        run_name="bn_bd_vits_finetune",
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        batch_group_size=5,
        num_loader_workers=4,
        num_eval_loader_workers=2,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=args.num_epochs,
        text_cleaner="basic_cleaners",  # বাংলা-নির্দিষ্ট normalization পরে যোগ করা যাবে
        use_phonemes=False,             # প্রথমে গ্রাফিম-ভিত্তিক, পরে phoneme-ভিত্তিক ট্রাই করা যায়
        compute_input_seq_cache=True,
        print_step=25,
        print_eval=True,
        mixed_precision=True,
        output_path=args.output_path,
        datasets=[dataset_config],
        characters={
            "characters": BENGALI_CHARACTERS,
            "punctuations": PUNCTUATION,
            "pad": "<PAD>",
            "eos": "<EOS>",
            "bos": "<BOS>",
            "blank": "<BLNK>",
        },
        save_step=1000,
        save_n_checkpoints=5,
        save_best_after=1000,
        target_loss="loss_1",
        lr=2e-4,
    )

    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    model = Vits(config, ap, tokenizer, speaker_manager=None)

    trainer = Trainer(
        TrainerArgs(restore_path=args.restore_path),
        config,
        args.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
