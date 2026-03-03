# Go demo (KittenTTS)

This demo runs the KittenTTS ONNX model from Go using ONNX Runtime and the
Goruut phonemizer.

## Setup

From the repository root:

### 1) Get the KittenTTS model assets

Clone the model repo from Hugging Face (example: mini model):

```bash
git clone https://huggingface.co/KittenML/kitten-tts-mini-0.8
```

### 2) Get ONNX Runtime

```bash
mkdir -p onnxruntime-linux-x64-1.18.0
cd onnxruntime-linux-x64-1.18.0
curl -L -o onnxruntime-linux-x64-1.18.0.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz
tar -xzf onnxruntime-linux-x64-1.18.0.tgz --strip-components=1
```

### 3) Get goruut (module dependency)

The demo uses goruut as a module with a local replace. Clone it next to this repo:

```bash
cd ..
git clone https://github.com/neurlang/goruut.git
```

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
