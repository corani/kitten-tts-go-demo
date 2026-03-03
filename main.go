package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode"

	ort "github.com/yalue/onnxruntime_go"
)

type Config struct {
	ModelFile    string             `json:"model_file"`
	Voices       string             `json:"voices"`
	SpeedPriors  map[string]float64 `json:"speed_priors"`
	VoiceAliases map[string]string  `json:"voice_aliases"`
}

type Args struct {
	modelDir       string
	text           string
	voice          string
	output         string
	speed          float64
	cleanText      bool
	espeakPath     string
	ipaMode        string
	onnxRuntimeLib string
}

func parseArgs() Args {
	var args Args
	flag.StringVar(&args.modelDir, "model-dir", "kitten-tts-mini-0.8", "Path to model directory (contains config.json)")
	flag.StringVar(&args.text, "text", "One day, a little girl named Lily found a needle in her room.", "Text to synthesize")
	flag.StringVar(&args.voice, "voice", "Bruno", "Voice name (e.g., Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo)")
	flag.StringVar(&args.output, "output", "output.wav", "Output WAV path")
	flag.Float64Var(&args.speed, "speed", 1.0, "Speech speed factor")
	flag.BoolVar(&args.cleanText, "clean-text", true, "Apply basic text cleanup before synthesis")
	flag.StringVar(&args.espeakPath, "espeak", "", "Path to espeak-ng executable (defaults to espeak-ng in PATH)")
	flag.StringVar(&args.ipaMode, "ipa-mode", "ipa", "espeak-ng IPA mode: ipa, ipa=1, ipa=2, ipa=3")
	flag.StringVar(&args.onnxRuntimeLib, "onnxruntime-lib", "onnxruntime-linux-x64-1.18.0/lib/libonnxruntime.so.1.18.0", "Path to libonnxruntime.so")
	flag.Parse()
	return args
}

func main() {
	args := parseArgs()

	configPath := filepath.Join(args.modelDir, "config.json")
	cfg, err := loadConfig(configPath)
	if err != nil {
		fatal(err)
	}

	modelPath := filepath.Join(args.modelDir, cfg.ModelFile)
	voicesPath := filepath.Join(args.modelDir, cfg.Voices)

	voiceID, ok := resolveVoiceID(cfg, args.voice)
	if !ok {
		available := sortedKeys(cfg.VoiceAliases)
		fatal(fmt.Errorf("unknown voice %q (available: %s)", args.voice, strings.Join(available, ", ")))
	}

	voices, err := LoadVoicesNPZ(voicesPath)
	if err != nil {
		fatal(err)
	}
	voiceArr, ok := voices[voiceID]
	if !ok {
		fatal(fmt.Errorf("voice %q not found in %s", voiceID, voicesPath))
	}

	if args.onnxRuntimeLib != "" {
		ort.SetSharedLibraryPath(args.onnxRuntimeLib)
	}
	if err := ort.InitializeEnvironment(); err != nil {
		fatal(fmt.Errorf("initialize onnxruntime: %w", err))
	}
	defer ort.DestroyEnvironment()

	inputInfo, outputInfo, err := ort.GetInputOutputInfo(modelPath)
	if err != nil {
		fatal(fmt.Errorf("inspect model io: %w", err))
	}
	inputNames := make([]string, 0, len(inputInfo))
	for _, i := range inputInfo {
		inputNames = append(inputNames, i.Name)
	}
	outputNames := make([]string, 0, len(outputInfo))
	for _, o := range outputInfo {
		outputNames = append(outputNames, o.Name)
	}
	if len(outputNames) == 0 {
		fatal(errors.New("model has no outputs"))
	}

	session, err := ort.NewDynamicAdvancedSession(modelPath, inputNames, outputNames, nil)
	if err != nil {
		fatal(fmt.Errorf("create onnx session: %w", err))
	}
	defer session.Destroy()

	text := args.text
	if args.cleanText {
		text = basicTextClean(text)
	}

	cleaner := NewTextCleaner()
	chunks := chunkText(text, 400)
	var audio []float32
	for _, chunk := range chunks {
		chunkAudio, err := synthesizeChunk(session, inputNames, outputNames, cleaner, voiceArr, cfg, voiceID, chunk, args.speed, args.espeakPath, args.ipaMode)
		if err != nil {
			fatal(err)
		}
		audio = append(audio, chunkAudio...)
	}

	if err := writeWavFile(args.output, audio, 24000); err != nil {
		fatal(fmt.Errorf("write wav: %w", err))
	}
	fmt.Printf("Saved audio to %s\n", args.output)
}

func synthesizeChunk(session *ort.DynamicAdvancedSession, inputNames []string, outputNames []string, cleaner *TextCleaner, voiceArr VoiceArray, cfg Config, voiceID string, text string, speed float64, espeakPath string, ipaMode string) ([]float32, error) {
	if text == "" {
		return nil, nil
	}

	phonemes, err := phonemize(text, espeakPath, ipaMode)
	if err != nil {
		return nil, err
	}

	phonemeTokens := basicEnglishTokenize(phonemes)
	textTokens := basicEnglishTokenize(text)
	mergedTokens := mergePhonemesWithPunct(textTokens, phonemeTokens)
	phonemeText := strings.Join(mergedTokens, " ")
	ids := cleaner.Encode(phonemeText)
	ids = append([]int64{0}, ids...)
	ids = append(ids, 10, 0)

	idsShape := ort.NewShape(1, int64(len(ids)))
	idsTensor, err := ort.NewTensor(idsShape, ids)
	if err != nil {
		return nil, fmt.Errorf("create input_ids tensor: %w", err)
	}
	defer idsTensor.Destroy()

	refID := len([]rune(text))
	if refID > voiceArr.Shape[0]-1 {
		refID = voiceArr.Shape[0] - 1
	}
	styleData, styleShape, err := sliceVoice(voiceArr, refID)
	if err != nil {
		return nil, err
	}
	styleTensor, err := ort.NewTensor(ort.Shape(styleShape), styleData)
	if err != nil {
		return nil, fmt.Errorf("create style tensor: %w", err)
	}
	defer styleTensor.Destroy()

	adjustedSpeed := float32(speed)
	if cfg.SpeedPriors != nil {
		if prior, ok := cfg.SpeedPriors[voiceID]; ok {
			adjustedSpeed *= float32(prior)
		}
	}
	speedTensor, err := ort.NewTensor(ort.NewShape(1), []float32{adjustedSpeed})
	if err != nil {
		return nil, fmt.Errorf("create speed tensor: %w", err)
	}
	defer speedTensor.Destroy()

	inputMap := map[string]ort.Value{
		"input_ids": idsTensor,
		"style":     styleTensor,
		"speed":     speedTensor,
	}
	inputs := make([]ort.Value, len(inputNames))
	for i, name := range inputNames {
		value, ok := inputMap[name]
		if !ok {
			return nil, fmt.Errorf("missing input %q", name)
		}
		inputs[i] = value
	}
	outputs := make([]ort.Value, len(outputNames))
	if err := session.Run(inputs, outputs); err != nil {
		return nil, fmt.Errorf("run onnx session: %w", err)
	}

	waveIndex := -1
	for i, name := range outputNames {
		if name == "waveform" {
			waveIndex = i
			break
		}
	}
	if waveIndex == -1 {
		return nil, fmt.Errorf("output 'waveform' not found; got %v", outputNames)
	}

	outTensor, ok := outputs[waveIndex].(*ort.Tensor[float32])
	if !ok {
		return nil, errors.New("unexpected output tensor type")
	}
	defer outTensor.Destroy()

	data := outTensor.GetData()
	if len(data) > 5000 {
		data = data[:len(data)-5000]
	}
	return data, nil
}

func resolveVoiceID(cfg Config, voice string) (string, bool) {
	if cfg.VoiceAliases == nil {
		return voice, true
	}
	if v, ok := cfg.VoiceAliases[voice]; ok {
		return v, true
	}
	for alias, id := range cfg.VoiceAliases {
		if strings.EqualFold(alias, voice) {
			return id, true
		}
	}
	return voice, false
}

func limitStrings(values []string, max int) []string {
	if len(values) <= max {
		return values
	}
	return append(values[:max], "...")
}

func mergePhonemesWithPunct(textTokens []string, phonemeTokens []string) []string {
	merged := make([]string, 0, len(textTokens))
	idx := 0
	for _, tok := range textTokens {
		if isWordToken(tok) {
			if idx < len(phonemeTokens) {
				merged = append(merged, phonemeTokens[idx])
				idx++
				continue
			}
		}
		merged = append(merged, tok)
	}
	// Append any remaining phoneme tokens if text ended without them.
	for idx < len(phonemeTokens) {
		merged = append(merged, phonemeTokens[idx])
		idx++
	}
	return merged
}

func isWordToken(tok string) bool {
	if tok == "" {
		return false
	}
	for _, r := range tok {
		if !(r == '_' || unicode.IsLetter(r) || unicode.IsDigit(r) || unicode.IsMark(r)) {
			return false
		}
	}
	return true
}

func sortedKeys(m map[string]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}
