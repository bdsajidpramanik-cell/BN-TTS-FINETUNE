# বাংলাদেশি বাংলা TTS Fine-tuning — ধাপে ধাপে গাইড

এই ৩টা স্ক্রিপ্ট একসাথে একটা fine-tuned VITS (Coqui TTS) মডেল তৈরি করবে, যেটা
পরে edge-tts-এর জায়গায় Rimi/Neodesk ব্যাকএন্ডে বসানো যাবে।

**গুরুত্বপূর্ণ:** এই স্ক্রিপ্টগুলো তোমার নিজের GPU মেশিনে (Colab Pro, RunPod,
Lambda Labs, বা নিজের GPU সার্ভারে) চালাতে হবে। এখানে (Claude-এর sandbox-এ)
নেটওয়ার্ক ও GPU নেই, তাই আমি এগুলো এখানে রান করে দেখাতে পারব না — কিন্তু কোডটা
সরাসরি কপি করে ব্যবহারযোগ্য।

---

## ধাপ ০ — এনভায়রনমেন্ট সেটআপ

GPU মেশিনে (Colab/RunPod টার্মিনালে) এই কমান্ডগুলো চালাও:

```bash
python -m venv tts_env
source tts_env/bin/activate

pip install TTS torch torchaudio soundfile librosa tqdm
```

GPU আছে কিনা চেক করো:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
`True` না দেখালে fine-tuning-এর জন্য GPU রানটাইম চালু করো (Colab: Runtime →
Change runtime type → GPU)।

---

## ধাপ ১ — ডেটা ডাউনলোড ও প্রিপ্রসেসিং

```bash
python 01_download_preprocess.py --output_dir ./data/bn_tts
```

এই স্ক্রিপ্ট যা করে:
1. OpenSLR SLR37 (bn-BD high-quality TTS speech) ডাউনলোড করে
2. অডিওগুলোকে 22050Hz, mono, 16-bit WAV-এ রিস্যাম্পল করে
3. LJSpeech ফরম্যাটে `metadata.csv` বানায় (Coqui TTS-এর জন্য প্রয়োজন)

**যদি ডাউনলোড লিংক কাজ না করে:** OpenSLR-এর ওয়েবসাইট (openslr.org) থেকে
ম্যানুয়ালি SLR37 ডাউনলোড করে `--skip_download --raw_dir <path>` ফ্ল্যাগ দিয়ে
শুধু প্রিপ্রসেসিং অংশ চালাও।

**চেক করার বিষয়:** SLR37-এ একাধিক speaker থাকতে পারে। যদি multi-speaker হয়,
`--speaker_id` ফ্ল্যাগ দিয়ে একটাই speaker বেছে filter করো — প্রথম fine-tune-এর
জন্য single speaker দিয়ে শুরু করা সহজ ও স্থিতিশীল।

---

## ধাপ ২ — Fine-tuning চালানো

```bash
python 02_train_vits.py \
  --data_path ./data/bn_tts \
  --output_path ./runs \
  --restore_path ./pretrained/vits_bn.pth \
  --num_epochs 100
```

- `--restore_path` না দিলে scratch থেকে ট্রেনিং শুরু হবে (অনেক বেশি ডেটা ও সময়
  লাগবে) — সম্ভব হলে একটা pretrained multilingual checkpoint দিয়ে শুরু করাই ভালো।
- ছোট ডেটাসেটে (২-১০ ঘণ্টা) `--num_epochs 100-300` দিয়ে শুরু করো, তারপর প্রতি
  কয়েক এপকে চেকপয়েন্ট শুনে (ধাপ ৩) কোয়ালিটি যাচাই করো।
- ট্রেনিং লগ ও চেকপয়েন্ট `./runs/` ফোল্ডারে সেভ হবে। TensorBoard দিয়ে monitor
  করা যায়: `tensorboard --logdir ./runs`

---

## ধাপ ৩ — টেস্ট সিন্থেসিস

```bash
python 03_synthesize.py \
  --checkpoint ./runs/best_model.pth \
  --config ./runs/config.json \
  --text "আপনাকে কীভাবে সহযোগিতা করতে পারি?" \
  --output test.wav
```

`test.wav` শুনে natural-ness, উচ্চারণ, আর ফ্লো যাচাই করো। এই কাজটা প্রতি কয়েক
epoch পরপর করা উচিত (ব্লুপ্রিন্টের ধাপ ৭-এর মতো) — শুধু loss নম্বর দেখে ভরসা
করা যাবে না, কান দিয়ে শোনাটাই আসল টেস্ট।

---

## ধাপ ৪ — Deployment (Rimi-তে বসানো)

edge-tts-এর জায়গায় fine-tuned মডেল বসাতে হলে:

1. Fine-tuned checkpoint + config একটা GPU-সহ inference endpoint-এ হোস্ট করো
   (RunPod Serverless, বা নিজের ছোট GPU VPS) — Render/Netlify-এর ফ্রি টিয়ারে
   এটা চলবে না, ওগুলো CPU-only।
2. Flask ব্যাকএন্ডে edge-tts কল করার জায়গায় এই নতুন endpoint-এ HTTP request
   পাঠাও, WAV/MP3 রেসপন্স হিসেবে ফেরত নাও।
3. Latency মাপো — continuous phone-call experience-এর জন্য first-audio-chunk
   কত দ্রুত আসছে সেটা critical। প্রয়োজনে streaming inference বা ছোট
   sentence-chunking ব্যবহার করো।

---

## সংক্ষেপে ক্রম

```
ধাপ ০ (setup) → ধাপ ১ (ডেটা) → ধাপ ২ (ট্রেনিং) → ধাপ ৩ (শোনা ও যাচাই)
→ (প্রয়োজনে ধাপ ২-এ ফিরে যাও, epoch/ডেটা বাড়াও) → ধাপ ৪ (deploy)
```
