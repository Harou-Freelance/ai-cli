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
	"time"
)

const (
	openAIBaseURL          = "https://api.openai.com/v1"
	openAIDefaultTimeout   = 30 * time.Second
	openAIDefaultTextModel = "gpt-4"
	openAIVisionModel      = "gpt-4-vision-preview"
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
	payload := map[string]interface{}{
		"model": p.getModel(),
		"messages": []map[string]interface{}{
			{"role": "user", "content": prompt},
		},
		"max_tokens": 1000,
	}

	return p.makeRequest(ctx, payload, "/chat/completions")
}

func (p *OpenAI) handleVisionRequest(ctx context.Context, inputs Inputs) (string, error) {
	content := []interface{}{
		map[string]string{"type": "text", "text": inputs.Prompt},
	}

	for _, img := range inputs.Images {
		base64Image, err := encodeImage(img.Reader)
		if err != nil {
			return "", fmt.Errorf("image encoding failed: %w", err)
		}

		content = append(content, map[string]interface{}{
			"type": "image_url",
			"image_url": map[string]string{
				"url": fmt.Sprintf("data:image/%s;base64,%s", getMimeType(img.Filename), base64Image),
			},
		})
	}

	payload := map[string]interface{}{
		"model": openAIVisionModel,
		"messages": []map[string]interface{}{
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

func encodeImage(r io.Reader) (string, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(data), nil
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

func (p *OpenAI) makeRequest(ctx context.Context, payload interface{}, endpoint string) (string, error) {
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
