package providers

import (
	"context"
	"io"
)

type Provider interface {
	Generate(ctx context.Context, inputs Inputs) (string, error)
	Supports(feature Feature) bool
}

type Feature int

const (
	FeatureTextGeneration Feature = iota
	FeatureVision
	FeatureMultiModal
)

type FileInput struct {
	Reader   io.Reader
	Filename string
}

type Inputs struct {
	Prompt string
	Images []FileInput
}

type Config struct {
	APIKey  string
	Timeout int
	Model   string
}
