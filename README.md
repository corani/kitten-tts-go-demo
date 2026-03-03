# Go demo (KittenTTS)

> [!warning]
> This is a vibe-coded demo for experimentation and learning. Do not use it in production.

This demo runs the KittenTTS ONNX model from Go using ONNX Runtime and the
Goruut phonemizer.

## Setup

### Expected folder structure

```text
root/
  go-demo/
  goruut/
  kitten-tts-mini-0.8/
  onnxruntime-linux-x64-1.18.0/
```

### 1) Clone this repo

```bash
git clone https://github.com/corani/kitten-tts-go-demo go-demo
```

### 2) Get the KittenTTS model assets

Clone the model repo from Hugging Face (example: mini model):

```bash
git clone https://huggingface.co/KittenML/kitten-tts-mini-0.8
```

### 3) Get ONNX Runtime

```bash
curl -L -o onnxruntime-linux-x64-1.18.0.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz
tar -xzf onnxruntime-linux-x64-1.18.0.tgz
```

### 4) Clone goruut

The demo uses goruut as a module with a local replace. Clone it next to this repo:

```bash
cd ..
git clone https://github.com/neurlang/goruut.git
```

This is needed because the repo contains large zip files that `go get` refuses to fetch.

## Run

From the repo root:

```bash
cd go-demo

go run . \
  --model-dir ../kitten-tts-mini-0.8 \
  --onnxruntime-lib ../onnxruntime-linux-x64-1.18.0/lib/libonnxruntime.so.1.18.0 \
  --voice Bruno \
  --goruut-lang EnglishAmerican \
  --text "One day, a little girl named Lily found a needle in her room." \
  --output output.wav
```

## Notes

- Goruut may log `open weights8.bin.zlib: file does not exist`. This is a known
  warning from its optional AI G2P weights. The demo still runs with dictionary
  phonemization.
- `--goruut-normalize` applies a minimal IPA normalization to better match the
  expected symbol set.
- Goruut should probably be replaced with something much smaller (it adds 600MB
  to the binary).
