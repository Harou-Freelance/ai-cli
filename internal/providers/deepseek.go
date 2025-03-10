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

/*
=== DeepSeek ===
Text Models (no vision support):
- deepseek-chat (DeepSeek-V3): General purpose (64K context)
- deepseek-reasoner (DeepSeek-R1): Advanced reasoning (64K context, 32K CoT tokens)
*/

const (
	deepseekBaseURL        = "https://api.deepseek.com/v1"
	deepseekDefaultModel   = "deepseek-chat"
	deepseekDefaultTimeout = 30 * time.Second
)

type DeepSeek struct {
	config Config
	client *http.Client
}

type deepseekError struct {
	Message string `json:"message"`
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

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var apiError deepseekError
		if json.Unmarshal(body, &apiError) == nil && apiError.Message != "" {
			return "", fmt.Errorf("API error [%d]: %s", resp.StatusCode, apiError.Message)
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

func (p *DeepSeek) getModel() string {
	if p.config.Model != "" {
		return p.config.Model
	}
	return deepseekDefaultModel
}

type DeepSeekModelsResponse struct {
	Data []struct {
		ID      string `json:"id"`
		Details struct {
			Description   string `json:"description"`
			ContextWindow int    `json:"context_length"`
		} `json:"capabilities"`
	} `json:"data"`
}

func (p *DeepSeek) ListModels(ctx context.Context) ([]Model, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", deepseekBaseURL+"/models", nil)
	if err != nil {
		return nil, fmt.Errorf("request creation failed: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+p.config.APIKey)

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("API request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error [%d]: %s", resp.StatusCode, string(body))
	}

	var response DeepSeekModelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("response parsing failed: %w", err)
	}

	var models []Model
	for _, m := range response.Data {
		models = append(models, Model{
			ID:             m.ID,
			Description:    m.Details.Description,
			ContextWindow:  m.Details.ContextWindow,
			SupportsVision: false, // DeepSeek currently has no vision models
		})
	}

	return models, nil
}
