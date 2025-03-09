package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

const (
	deepseekBaseURL        = "https://api.deepseek.com/v1"
	deepseekDefaultModel   = "deepseek-chat"
	deepseekDefaultTimeout = 30 * time.Second
)

type DeepSeek struct {
	config Config
	client *http.Client
}

func NewDeepSeek(config Config) *DeepSeek {
	if config.Timeout == 0 {
		config.Timeout = int(deepseekDefaultTimeout.Seconds())
	}
	return &DeepSeek{
		config: config,
		client: &http.Client{Timeout: deepseekDefaultTimeout},
	}
}

func (p *DeepSeek) Supports(feature Feature) bool {
	return feature == FeatureTextGeneration
}

func (p *DeepSeek) Generate(ctx context.Context, inputs Inputs) (string, error) {
	if len(inputs.Images) > 0 {
		return "", fmt.Errorf("DeepSeek does not support image analysis")
	}
	return p.handleTextRequest(ctx, inputs.Prompt)
}

func (p *DeepSeek) handleTextRequest(ctx context.Context, prompt string) (string, error) {
	payload := map[string]any{
		"model": p.getModel(),
		"messages": []map[string]any{
			{"role": "user", "content": prompt},
		},
		"max_tokens": 1000,
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("marshal error: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", deepseekBaseURL+"/chat/completions", bytes.NewBuffer(jsonData))
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

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("API error [%d]: %s", resp.StatusCode, string(body))
	}

	var response struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return "", fmt.Errorf("response parsing failed: %w", err)
	}

	if len(response.Choices) == 0 {
		return "", fmt.Errorf("no content in response")
	}

	return response.Choices[0].Message.Content, nil
}

func (p *DeepSeek) getModel() string {
	if p.config.Model != "" {
		return p.config.Model
	}
	return deepseekDefaultModel
}
