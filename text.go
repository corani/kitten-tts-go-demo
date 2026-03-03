package main

import (
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
