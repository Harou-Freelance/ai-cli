package providers

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"path/filepath"
	"strings"
	"time"
)

/*
=== OpenAI ===
Text Models:
- gpt-4o-mini: General purpose text (128K context)
- gpt-4-turbo: Advanced text processing (128K context)

Vision Models (supports image input via URL/base64):
- gpt-4o: General vision capabilities (128K context)
- gpt-4o-mini: Basic vision processing (128K context)
- gpt-4-turbo: Advanced vision analysis (128K context)

Vision Limitations:
- Max image size: 20MB (PNG/JPEG/WEBP/non-animated GIF)
- Medical images not supported
- Struggles with rotated text, spatial reasoning, and non-Latin characters
- Image costs: 85-170 tokens per 512px tile + base 85 tokens
*/

const (
	openAIBaseURL          = "https://api.openai.com/v1"
	openAIDefaultTimeout   = 30 * time.Second
	openAIDefaultTextModel = "gpt-4"
	openAIVisionModel      = "gpt-4o-mini" //models supporting images as input: o1, gpt-4.5-preview, gpt-4o, gpt-4o-mini, gpt-4-turbo
)

type OpenAI struct {
	config Config
	client *http.Client
}

type openAIError struct {
	Error struct {
		Message string `json:"message"`
	} `json:"error"`
}

func NewOpenAI(config Config) *OpenAI {
	if config.Timeout == 0 {
		config.Timeout = int(openAIDefaultTimeout.Seconds())
	}
	return &OpenAI{
		config: config,
		client: &http.Client{Timeout: openAIDefaultTimeout},
	}
}

func (p *OpenAI) Supports(feature Feature) bool {
	switch feature {
	case FeatureTextGeneration, FeatureVision, FeatureMultiModal:
		return true
	default:
		return false
	}
}

func (p *OpenAI) Generate(ctx context.Context, inputs Inputs) (string, error) {
	if len(inputs.Images) > 0 {
		return p.handleVisionRequest(ctx, inputs)
	}
	return p.handleTextRequest(ctx, inputs.Prompt)
}

func (p *OpenAI) handleTextRequest(ctx context.Context, prompt string) (string, error) {
	payload := map[string]any{
		"model": p.getModel(),
		"messages": []map[string]any{
			{"role": "user", "content": prompt},
		},
		"max_tokens": 1000,
	}

	return p.makeRequest(ctx, payload, "/chat/completions")
}

func (p *OpenAI) handleVisionRequest(ctx context.Context, inputs Inputs) (string, error) {
	content := []any{
		map[string]string{"type": "text", "text": inputs.Prompt},
	}

	for _, img := range inputs.Images {
		// Use the pre-loaded image data
		base64Image := base64.StdEncoding.EncodeToString(img.Data)

		content = append(content, map[string]any{
			"type": "image_url",
			"image_url": map[string]string{
				"url": fmt.Sprintf("data:image/%s;base64,%s",
					getMimeType(img.Filename),
					base64Image,
				),
			},
		})
	}

	payload := map[string]any{
		"model": openAIVisionModel,
		"messages": []map[string]any{
			{"role": "user", "content": content},
		},
		"max_tokens": 1000,
	}

	return p.makeRequest(ctx, payload, "/chat/completions")
}

func (p *OpenAI) getModel() string {
	if p.config.Model != "" {
		return p.config.Model
	}
	return openAIDefaultTextModel
}

func getMimeType(filename string) string {
	ext := filepath.Ext(filename)
	switch ext {
	case ".png":
		return "png"
	case ".jpg", ".jpeg":
		return "jpeg"
	case ".gif":
		return "gif"
	default:
		return "jpeg"
	}
}

func (p *OpenAI) makeRequest(ctx context.Context, payload any, endpoint string) (string, error) {
	jsonData, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("marshal error: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", openAIBaseURL+endpoint, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("request creation failed: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.config.APIKey)

	resp, err := p.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("API request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var apiError openAIError
		if json.Unmarshal(body, &apiError) == nil && apiError.Error.Message != "" {
			return "", fmt.Errorf("API error [%d]: %s", resp.StatusCode, apiError.Error.Message)
		}
		return "", fmt.Errorf("API error [%d]: %s", resp.StatusCode, string(body))
	}

	var response struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return "", fmt.Errorf("response parsing failed: %w", err)
	}

	if len(response.Choices) == 0 {
		return "", fmt.Errorf("no content in response")
	}

	return response.Choices[0].Message.Content, nil
}

type OpenAIModelResponse struct {
	Object string        `json:"object"`
	Data   []OpenAIModel `json:"data"`
}

type OpenAIModel struct {
	ID          string `json:"id"`
	Object      string `json:"object"`
	Created     int64  `json:"created"`
	OwnedBy     string `json:"owned_by"`
	Description string `json:"description,omitempty"`
}

func (p *OpenAI) ListModels(ctx context.Context) ([]Model, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", openAIBaseURL+"/models", nil)
	if err != nil {
		return nil, fmt.Errorf("request creation failed: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+p.config.APIKey)

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("API request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API error [%d]", resp.StatusCode)
	}

	var response OpenAIModelResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("response parsing failed: %w", err)
	}

	models := make([]Model, 0, len(response.Data))
	for _, m := range response.Data {
		models = append(models, Model{
			ID:             m.ID,
			Description:    fmt.Sprintf("%s (%s)", m.ID, m.OwnedBy),
			ContextWindow:  getOpenAIContextWindow(m.ID),
			SupportsVision: isVisionModel(m.ID),
		})
	}

	return models, nil
}

// Helper functions
func getOpenAIContextWindow(modelID string) int {
	switch {
	case strings.Contains(modelID, "128k"):
		return 128000
	case strings.Contains(modelID, "32k"):
		return 32000
	case strings.Contains(modelID, "16k"):
		return 16000
	default:
		return 4096 // Default context window
	}
}

func isVisionModel(modelID string) bool {
	return strings.Contains(modelID, "vision") ||
		strings.Contains(modelID, "gpt-4o") ||
		strings.Contains(modelID, "turbo-vision")
}
