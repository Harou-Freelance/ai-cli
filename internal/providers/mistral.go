package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

/*
=== Mistral ===
- ministral-8b-latest: Fastest, lightweight text generation (32K context, ~300 tokens/s)
- mixtral-8x7b-instruct: High-quality text (32K context, ~200 tokens/s)
- mistral-large-latest: Advanced reasoning (128K context, ~150 tokens/s)
*/
const (
	mistralBaseURL        = "https://api.mistral.ai/v1"
	mistralDefaultModel   = "mistral-small-latest"
	mistralDefaultTimeout = 30 * time.Second
	mistralMaxRetries     = 2
	mistralRetryDelay     = 1 * time.Second
)

type Mistral struct {
	config Config
	client *http.Client
}

type mistralError struct {
	Message string `json:"message"`
}

func NewMistral(config Config) *Mistral {
	timeout := mistralDefaultTimeout
	if config.Timeout > 0 && config.Timeout <= 30 {
		timeout = time.Duration(config.Timeout) * time.Second
	}
	return &Mistral{
		config: config,
		client: &http.Client{Timeout: timeout},
	}
}

func (p *Mistral) Supports(feature Feature) bool {
	return feature == FeatureTextGeneration
}

func (p *Mistral) Generate(ctx context.Context, inputs Inputs) (string, error) {
	if len(inputs.Images) > 0 {
		return "", fmt.Errorf("Mistral does not support image analysis")
	}
	return p.handleTextRequest(ctx, inputs.Prompt)
}

func (p *Mistral) handleTextRequest(ctx context.Context, prompt string) (string, error) {
	payload := map[string]interface{}{
		"model":      p.getModel(),
		"messages":   []map[string]interface{}{{"role": "user", "content": prompt}},
		"max_tokens": 1000,
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("marshal error: %w", err)
	}

	var lastErr error
	for attempt := 1; attempt <= mistralMaxRetries; attempt++ {
		start := time.Now()
		req, err := http.NewRequestWithContext(ctx, "POST", mistralBaseURL+"/chat/completions", bytes.NewBuffer(jsonData))
		if err != nil {
			return "", fmt.Errorf("request creation failed: %w", err)
		}

		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "application/json")
		req.Header.Set("Authorization", "Bearer "+p.config.APIKey)

		if p.config.Debug {
			fmt.Printf("[DEBUG] Attempt %d: Sending request to Mistral: URL=%s, Model=%s, APIKey=%s\n",
				attempt, mistralBaseURL+"/chat/completions", p.getModel(), maskAPIKey(p.config.APIKey))
		}

		resp, err := p.client.Do(req)
		if err != nil {
			lastErr = fmt.Errorf("API request failed: %w", err)
			if p.config.Debug {
				fmt.Printf("[DEBUG] Attempt %d failed after %s: %v\n", attempt, time.Since(start), err)
			}
			if attempt < mistralMaxRetries {
				time.Sleep(mistralRetryDelay)
				continue
			}
			return "", lastErr
		}
		defer resp.Body.Close()

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return "", fmt.Errorf("failed to read response body: %w", err)
		}

		if p.config.Debug {
			fmt.Printf("[DEBUG] Attempt %d: Response status=%d, Time=%s, Body=%s\n",
				attempt, resp.StatusCode, time.Since(start), string(body))
		}

		if resp.StatusCode != http.StatusOK {
			var apiError mistralError
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

		if p.config.Debug {
			fmt.Printf("[DEBUG] Success after %s\n", time.Since(start))
		}
		return response.Choices[0].Message.Content, nil
	}

	return "", lastErr
}

func (p *Mistral) ListModels(ctx context.Context) ([]Model, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", mistralBaseURL+"/models", nil)
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
		return nil, fmt.Errorf("API error [%d]: %s", resp.StatusCode, string(body))
	}

	var response struct {
		Data []struct {
			ID      string `json:"id"`
			Created int64  `json:"created"`
			Object  string `json:"object"`
			OwnedBy string `json:"owned_by"`
		} `json:"data"`
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("response parsing failed: %w", err)
	}

	models := make([]Model, 0, len(response.Data))
	for _, m := range response.Data {
		models = append(models, Model{
			ID:             m.ID,
			Description:    fmt.Sprintf("Mistral model: %s", m.ID),
			ContextWindow:  getMistralContextWindow(m.ID),
			SupportsVision: false,
		})
	}

	return models, nil
}

func (p *Mistral) getModel() string {
	if p.config.Model != "" {
		return p.config.Model
	}
	return mistralDefaultModel
}

func getMistralContextWindow(modelID string) int {
	switch {
	case strings.Contains(modelID, "large"):
		return 128000
	case strings.Contains(modelID, "8x22b"), strings.Contains(modelID, "8x7b"), strings.Contains(modelID, "ministral-8b"):
		return 32000
	default:
		return 32000
	}
}

func maskAPIKey(key string) string {
	if len(key) < 8 {
		return "****"
	}
	return key[:4] + "..." + key[len(key)-4:]
}
