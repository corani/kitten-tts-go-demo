package main

import (
	"math"
	"os"

	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
)

func writeWavFile(filename string, audioData []float32, sampleRate int) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	intData := make([]int, len(audioData))
	for i, sample := range audioData {
		clamped := math.Max(-1.0, math.Min(1.0, float64(sample)))
		intData[i] = int(clamped * 32767)
	}

	encoder := wav.NewEncoder(file, sampleRate, 16, 1, 1)
	buf := &audio.IntBuffer{
		Data:           intData,
		Format:         &audio.Format{SampleRate: sampleRate, NumChannels: 1},
		SourceBitDepth: 16,
	}
	if err := encoder.Write(buf); err != nil {
		return err
	}
	return encoder.Close()
}
