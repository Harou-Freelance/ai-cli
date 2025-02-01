/*
Copyright © 2024 NAME HERE <EMAIL ADDRESS>
*/
package cmd

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"

	"github.com/joho/godotenv"
	"github.com/spf13/cobra"
)

var (
	userPrompt string
	apiKey     string
	promptFile string
	imagePath  string
)

// UnifiedResponse handles responses from both OpenAI and DeepSeek
type UnifiedResponse struct {
	ID      string `json:"id,omitempty"`
	Object  string `json:"object,omitempty"`
	Created int64  `json:"created,omitempty"`
	Model   string `json:"model,omitempty"`
	Choices []struct {
		Index   int `json:"index,omitempty"`
		Message struct {
			Role    string `json:"role,omitempty"`
			Content any    `json:"content,omitempty"`
		} `json:"message,omitempty"`
		FinishReason string `json:"finish_reason,omitempty"`
	} `json:"choices,omitempty"`
	Usage interface{} `json:"usage,omitempty"`
}

// analyzeCmd remains OpenAI-only (vision)
var analyzeCmd = &cobra.Command{
	Use:   "analyze",
	Short: "Analyze an image with GPT-4 Vision capabilities",
	Run: func(cmd *cobra.Command, args []string) {
		_ = godotenv.Load()
		key, err := loadAPIKey("openai", apiKey)
		if err != nil {
			log.Println(err)
			return
		}

		if imagePath == "" {
			log.Println("Image path is required.")
			return
		}

		finalPrompt, err := getFinalPrompt(userPrompt, promptFile)
		if err != nil {
			log.Println("Error getting prompt:", err)
			return
		}

		base64Image, err := encodeImageToBase64(imagePath)
		if err != nil {
			log.Println("Error encoding image:", err)
			return
		}

		responseJSON, err := callVisionAPI(base64Image, finalPrompt, key)
		if err != nil {
			log.Println("Error from API:", err)
			return
		}

		fmt.Println(responseJSON)
	},
}

// promptCmd now supports both providers
var promptCmd = &cobra.Command{
	Use:   "prompt",
	Short: "Interact with OpenAI or DeepSeek using text prompts",
	Run: func(cmd *cobra.Command, args []string) {
		_ = godotenv.Load()

		provider, _ := cmd.Flags().GetString("provider")
		key, err := loadAPIKey(provider, apiKey)
		if err != nil {
			log.Println(err)
			return
		}

		finalPrompt, err := getFinalPrompt(userPrompt, promptFile)
		if err != nil {
			log.Println("Error getting prompt:", err)
			return
		}

		responseJSON, err := callTextAPI(finalPrompt, key, provider)
		if err != nil {
			log.Println("Error from API:", err)
			return
		}

		fmt.Println(responseJSON)
	},
}

func init() {
	rootCmd.AddCommand(analyzeCmd)
	rootCmd.AddCommand(promptCmd)

	// Analyze command flags
	analyzeCmd.Flags().StringVarP(&imagePath, "image", "i", "", "Path to image file")
	analyzeCmd.Flags().StringVarP(&userPrompt, "prompt", "p", "", "Text prompt")
	analyzeCmd.Flags().StringVarP(&apiKey, "apikey", "k", "", "OpenAI API key")
	analyzeCmd.Flags().StringVar(&promptFile, "prompt-file", "", "Prompt file path")

	// Prompt command flags
	promptCmd.Flags().StringVarP(&userPrompt, "prompt", "p", "", "Text prompt")
	promptCmd.Flags().StringVarP(&apiKey, "apikey", "k", "", "API key for selected provider")
	promptCmd.Flags().StringVar(&promptFile, "prompt-file", "", "Prompt file path")
	promptCmd.Flags().String("provider", "openai", "API provider (openai|deepseek)")
}

// Updated API key loading
func loadAPIKey(provider, flagKey string) (string, error) {
	if flagKey != "" {
		return flagKey, nil
	}

	var envKey string
	switch provider {
	case "openai":
		envKey = os.Getenv("OPENAI_API_KEY")
	case "deepseek":
		envKey = os.Getenv("DEEPSEEK_API_KEY")
	default:
		return "", fmt.Errorf("unknown provider: %s", provider)
	}

	if envKey == "" {
		return "", fmt.Errorf("API key required for %s. Set via --apikey or environment variable", provider)
	}

	return envKey, nil
}

// Unified text API caller
func callTextAPI(prompt, apiKey, provider string) (string, error) {
	var (
		url   string
		model string
	)

	switch provider {
	case "openai":
		url = "https://api.openai.com/v1/chat/completions"
		model = "gpt-4"
	case "deepseek":
		url = "https://api.deepseek.com/v1/chat/completions"
		model = "deepseek-chat"
	default:
		return "", fmt.Errorf("unsupported provider: %s", provider)
	}

	payload := map[string]interface{}{
		"model": model,
		"messages": []map[string]interface{}{
			{"role": "user", "content": prompt},
		},
		"max_tokens": 1000,
	}

	jsonData, _ := json.Marshal(payload)
	req, _ := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("API error: %s", string(body))
	}

	var result UnifiedResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return "", err
	}

	formatted, _ := json.MarshalIndent(result, "", "  ")
	return string(formatted), nil
}

// Existing vision functions remain unchanged
func encodeImageToBase64(path string) (string, error) {
	file, _ := os.Open(path)
	defer file.Close()
	data, _ := io.ReadAll(file)
	return base64.StdEncoding.EncodeToString(data), nil
}

func callVisionAPI(image, prompt, apiKey string) (string, error) {
	payload := map[string]interface{}{
		"model": "gpt-4o-mini",
		"messages": []map[string]interface{}{
			{
				"role": "user",
				"content": []map[string]interface{}{
					{"type": "text", "text": prompt},
					{
						"type": "image_url",
						"image_url": map[string]interface{}{
							"url": "data:image/jpeg;base64," + image,
						},
					},
				},
			},
		},
	}
	jsonData, _ := json.Marshal(payload)
	req, _ := http.NewRequest("POST", "https://api.openai.com/v1/chat/completions", bytes.NewBuffer(jsonData))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("API error: %s", string(body))
	}

	var result UnifiedResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return "", err
	}

	formatted, _ := json.MarshalIndent(result, "", "  ")
	return string(formatted), nil
}

// Helper functions remain the same
func getFinalPrompt(prompt, filePath string) (string, error) {
	if filePath != "" {
		data, err := os.ReadFile(filePath)
		return string(data), err
	}
	return prompt, nil
}
