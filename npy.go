package main

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"regexp"
	"strconv"
	"strings"
)

type VoiceArray struct {
	Shape []int
	Data  []float32
}

func LoadVoicesNPZ(path string) (map[string]VoiceArray, error) {
	zr, err := zip.OpenReader(path)
	if err != nil {
		return nil, fmt.Errorf("open npz: %w", err)
	}
	defer zr.Close()

	voices := make(map[string]VoiceArray)
	for _, f := range zr.File {
		if f.FileInfo().IsDir() {
			continue
		}
		if !strings.HasSuffix(f.Name, ".npy") {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			return nil, fmt.Errorf("read npz file %s: %w", f.Name, err)
		}
		data, err := io.ReadAll(rc)
		rc.Close()
		if err != nil {
			return nil, fmt.Errorf("read npz file %s: %w", f.Name, err)
		}
		arr, err := readNPY(data)
		if err != nil {
			return nil, fmt.Errorf("parse npy %s: %w", f.Name, err)
		}
		key := strings.TrimSuffix(f.Name, ".npy")
		voices[key] = arr
	}
	if len(voices) == 0 {
		return nil, errors.New("no .npy arrays found in npz")
	}
	return voices, nil
}

func readNPY(data []byte) (VoiceArray, error) {
	const magic = "\x93NUMPY"
	if len(data) < 10 || string(data[:6]) != magic {
		return VoiceArray{}, errors.New("invalid npy header")
	}
	major := data[6]
	minor := data[7]
	_ = minor
	var headerLen int
	offset := 8
	switch major {
	case 1:
		if len(data) < offset+2 {
			return VoiceArray{}, errors.New("npy header too short")
		}
		headerLen = int(binary.LittleEndian.Uint16(data[offset : offset+2]))
		offset += 2
	case 2:
		if len(data) < offset+4 {
			return VoiceArray{}, errors.New("npy header too short")
		}
		headerLen = int(binary.LittleEndian.Uint32(data[offset : offset+4]))
		offset += 4
	default:
		return VoiceArray{}, fmt.Errorf("unsupported npy version %d", major)
	}
	if len(data) < offset+headerLen {
		return VoiceArray{}, errors.New("npy header truncated")
	}
	header := string(data[offset : offset+headerLen])
	offset += headerLen

	dtype, shape, err := parseNPYHeader(header)
	if err != nil {
		return VoiceArray{}, err
	}
	if len(shape) < 2 {
		return VoiceArray{}, fmt.Errorf("unexpected shape %v", shape)
	}

	elemCount := 1
	for _, v := range shape {
		elemCount *= v
	}
	payload := data[offset:]

	switch dtype {
	case "<f4", "|f4", "f4":
		if len(payload) < elemCount*4 {
			return VoiceArray{}, errors.New("npy data too short for f4")
		}
		out := make([]float32, elemCount)
		buf := bytes.NewReader(payload)
		if err := binary.Read(buf, binary.LittleEndian, &out); err != nil {
			return VoiceArray{}, fmt.Errorf("read f4 data: %w", err)
		}
		return VoiceArray{Shape: shape, Data: out}, nil
	case "<f8", "|f8", "f8":
		if len(payload) < elemCount*8 {
			return VoiceArray{}, errors.New("npy data too short for f8")
		}
		out := make([]float32, elemCount)
		buf := bytes.NewReader(payload)
		for i := 0; i < elemCount; i++ {
			var v float64
			if err := binary.Read(buf, binary.LittleEndian, &v); err != nil {
				return VoiceArray{}, fmt.Errorf("read f8 data: %w", err)
			}
			out[i] = float32(v)
		}
		return VoiceArray{Shape: shape, Data: out}, nil
	default:
		return VoiceArray{}, fmt.Errorf("unsupported dtype %s", dtype)
	}
}

func parseNPYHeader(header string) (string, []int, error) {
	dtypeRe := regexp.MustCompile(`'descr'\s*:\s*'([^']+)'`)
	shapeRe := regexp.MustCompile(`'shape'\s*:\s*\(([^\)]*)\)`)

	dtypeMatch := dtypeRe.FindStringSubmatch(header)
	if len(dtypeMatch) != 2 {
		return "", nil, errors.New("npy header missing descr")
	}
	shapeMatch := shapeRe.FindStringSubmatch(header)
	if len(shapeMatch) != 2 {
		return "", nil, errors.New("npy header missing shape")
	}

	shapeParts := strings.Split(shapeMatch[1], ",")
	shape := make([]int, 0, len(shapeParts))
	for _, part := range shapeParts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		v, err := strconv.Atoi(part)
		if err != nil {
			return "", nil, fmt.Errorf("invalid shape value %q", part)
		}
		shape = append(shape, v)
	}
	if len(shape) == 0 {
		return "", nil, errors.New("empty shape")
	}

	return dtypeMatch[1], shape, nil
}

func loadConfig(path string) (Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Config{}, fmt.Errorf("read config: %w", err)
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return Config{}, fmt.Errorf("parse config: %w", err)
	}
	return cfg, nil
}

func sliceVoice(arr VoiceArray, row int) ([]float32, []int64, error) {
	if len(arr.Shape) < 2 {
		return nil, nil, fmt.Errorf("voice array shape too small: %v", arr.Shape)
	}
	rows := arr.Shape[0]
	if row < 0 || row >= rows {
		return nil, nil, fmt.Errorf("row %d out of range", row)
	}
	rowSize := 1
	for _, v := range arr.Shape[1:] {
		rowSize *= v
	}
	start := row * rowSize
	end := start + rowSize
	if end > len(arr.Data) {
		return nil, nil, errors.New("voice array data is shorter than expected")
	}
	data := make([]float32, rowSize)
	copy(data, arr.Data[start:end])

	shape := make([]int64, 0, len(arr.Shape))
	shape = append(shape, 1)
	for _, v := range arr.Shape[1:] {
		shape = append(shape, int64(v))
	}
	return data, shape, nil
}
