package providers

import (
	"context"
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
	Data     []byte
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

type ModelLister interface {
	ListModels(ctx context.Context) ([]Model, error)
}

type Model struct {
	ID             string `json:"id"`
	Description    string `json:"description"`
	ContextWindow  int    `json:"context_window"`
	SupportsVision bool   `json:"supports_vision"`
}
