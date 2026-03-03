package main

import (
	"bytes"
	"errors"
	"fmt"
	"os/exec"
	"regexp"
	"strings"
	"unicode"
)

type TextCleaner struct {
	index map[rune]int64
}

func NewTextCleaner() *TextCleaner {
	pad := "$"
	punctuation := `;:,.!?¡¿—…"«»"" `
	letters := "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
	lettersIPA := `ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ`

	symbols := []rune(pad + punctuation + letters + lettersIPA)
	index := make(map[rune]int64, len(symbols))
	for i, r := range symbols {
		index[r] = int64(i)
	}
	return &TextCleaner{index: index}
}

func (tc *TextCleaner) Encode(text string) []int64 {
	out := make([]int64, 0, len(text))
	for _, r := range text {
		if v, ok := tc.index[r]; ok {
			out = append(out, v)
		}
	}
	return out
}

func basicTextClean(text string) string {
	text = strings.ReplaceAll(text, "\n", " ")
	text = strings.TrimSpace(text)
	text = regexp.MustCompile(`\s+`).ReplaceAllString(text, " ")
	return text
}

func ensurePunctuation(text string) string {
	text = strings.TrimSpace(text)
	if text == "" {
		return text
	}
	last := text[len(text)-1]
	if !strings.ContainsRune(".!?,;:", rune(last)) {
		text += ","
	}
	return text
}

func chunkText(text string, maxLen int) []string {
	if maxLen <= 0 {
		maxLen = 400
	}
	sentences := regexp.MustCompile(`[.!?]+`).Split(text, -1)
	chunks := make([]string, 0, len(sentences))
	for _, sentence := range sentences {
		sentence = strings.TrimSpace(sentence)
		if sentence == "" {
			continue
		}
		if len(sentence) <= maxLen {
			chunks = append(chunks, ensurePunctuation(sentence))
			continue
		}
		words := strings.Fields(sentence)
		var temp strings.Builder
		for _, word := range words {
			if temp.Len() > 0 {
				if temp.Len()+1+len(word) > maxLen {
					chunks = append(chunks, ensurePunctuation(temp.String()))
					temp.Reset()
				} else {
					temp.WriteByte(' ')
				}
			}
			temp.WriteString(word)
		}
		if temp.Len() > 0 {
			chunks = append(chunks, ensurePunctuation(temp.String()))
		}
	}
	if len(chunks) == 0 {
		return []string{""}
	}
	return chunks
}

func phonemize(text string, espeakPath string, ipaMode string) (string, error) {
	path := strings.TrimSpace(espeakPath)
	if path == "" {
		path = "espeak-ng"
	}
	mode := strings.TrimSpace(ipaMode)
	if mode == "" {
		mode = "ipa"
	}
	cmd := exec.Command(path, "-q", "--"+mode, "-v", "en-us", text)
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("espeak-ng failed: %w: %s", err, strings.TrimSpace(stderr.String()))
	}
	out := strings.TrimSpace(stdout.String())
	if out == "" {
		return "", errors.New("espeak-ng returned empty phonemes")
	}
	out = regexp.MustCompile(`\s+`).ReplaceAllString(out, " ")
	return strings.TrimSpace(out), nil
}

func basicEnglishTokenize(text string) []string {
	var tokens []string
	var current strings.Builder
	flush := func() {
		if current.Len() > 0 {
			tokens = append(tokens, current.String())
			current.Reset()
		}
	}
	for _, r := range text {
		if unicode.IsSpace(r) {
			flush()
			continue
		}
		if isWordRune(r) {
			current.WriteRune(r)
			continue
		}
		flush()
		tokens = append(tokens, string(r))
	}
	flush()
	return tokens
}

func isWordRune(r rune) bool {
	return r == '_' || unicode.IsLetter(r) || unicode.IsDigit(r) || unicode.IsMark(r)
}
