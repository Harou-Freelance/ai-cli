package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/harou24/ai-cli/internal/providers"
	"github.com/joho/godotenv"
	"github.com/spf13/cobra"
)

var (
	modelsProvider []string
	modelsJson     bool
)

var modelsCmd = &cobra.Command{
	Use:   "models",
	Short: "List available models for supported providers",
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx := context.Background()
		_ = godotenv.Load()

		if len(modelsProvider) == 0 {
			modelsProvider = []string{"openai", "deepseek"}
		}

		// Map to group models by provider
		providerModels := make(map[string][]providers.Model)
		var errs []error

		for _, provider := range modelsProvider {
			provider = strings.ToLower(provider)
			key, err := getAPIKeyForProvider(provider)
			if err != nil {
				errs = append(errs, fmt.Errorf("%s: %w", provider, err))
				continue
			}

			lister, err := getModelLister(provider, key)
			if err != nil {
				errs = append(errs, fmt.Errorf("%s: %w", provider, err))
				continue
			}

			models, err := lister.ListModels(ctx)
			if err != nil {
				errs = append(errs, fmt.Errorf("%s: %w", provider, err))
				continue
			}

			providerModels[provider] = models
		}

		if len(errs) > 0 {
			for _, err := range errs {
				log.Printf("Error: %v", err)
			}
		}

		if modelsJson {
			jsonData, _ := json.MarshalIndent(providerModels, "", "  ")
			fmt.Println(string(jsonData))
		} else {
			// Print separate tables for each provider
			for provider, models := range providerModels {
				printProviderTable(provider, models)
				fmt.Println() // Add space between tables
			}
		}
		return nil
	},
}

func printProviderTable(provider string, models []providers.Model) {
	fmt.Printf("\n%s Models:\n", strings.Title(provider))
	if len(models) == 0 {
		fmt.Println("  No models available")
		return
	}

	fmt.Println("┌──────────────────────┬──────────────────────┬──────────────┬─────────────┐")
	fmt.Println("│ Model ID             │ Description          │ Context Size │ Vision      │")
	fmt.Println("├──────────────────────┼──────────────────────┼──────────────┼─────────────┤")
	for _, m := range models {
		fmt.Printf("│ %-20s │ %-20s │ %-12d │ %-11v │\n",
			truncate(m.ID, 20),
			truncate(m.Description, 20),
			m.ContextWindow,
			m.SupportsVision)
	}
	fmt.Println("└──────────────────────┴──────────────────────┴──────────────┴─────────────┘")
}

func init() {
	modelsCmd.Flags().StringSliceVar(&modelsProvider, "provider", []string{}, "Comma-separated list of providers (openai,deepseek)")
	modelsCmd.Flags().BoolVar(&modelsJson, "json", false, "Output in JSON format")
	rootCmd.AddCommand(modelsCmd)
}

func getAPIKeyForProvider(provider string) (string, error) {
	switch provider {
	case "openai":
		key := os.Getenv("OPENAI_API_KEY")
		if key == "" {
			return "", fmt.Errorf("OPENAI_API_KEY not found in environment")
		}
		return key, nil
	case "deepseek":
		key := os.Getenv("DEEPSEEK_API_KEY")
		if key == "" {
			return "", fmt.Errorf("DEEPSEEK_API_KEY not found in environment")
		}
		return key, nil
	default:
		return "", fmt.Errorf("unsupported provider")
	}
}

func getModelLister(provider string, apiKey string) (providers.ModelLister, error) {
	switch provider {
	case "openai":
		return providers.NewOpenAI(providers.Config{APIKey: apiKey}), nil
	case "deepseek":
		return providers.NewDeepSeek(providers.Config{APIKey: apiKey}), nil
	default:
		return nil, fmt.Errorf("unsupported provider")
	}
}

func getProviderName(modelID string) string {
	switch {
	case strings.Contains(modelID, "deepseek"):
		return "DeepSeek"
	case strings.Contains(modelID, "gpt"):
		return "OpenAI"
	default:
		return "Unknown"
	}
}

func truncate(s string, length int) string {
	if len(s) > length {
		return s[:length-3] + "..."
	}
	return s
}
