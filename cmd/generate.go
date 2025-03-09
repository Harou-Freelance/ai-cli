package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/harou24/ai-cli/internal/providers"
	"github.com/joho/godotenv"
	"github.com/spf13/cobra"
)

var (
	promptFlag   string
	imagesFlag   []string
	providerFlag string
	apiKeyFlag   string
	jsonOutput   bool
)

type CLIOutput struct {
	Success  bool     `json:"success"`
	Content  string   `json:"content,omitempty"`
	Error    string   `json:"error,omitempty"`
	Warnings []string `json:"warnings,omitempty"`
}

var generateCmd = &cobra.Command{
	Use:     "generate",
	Aliases: []string{"gen", "ask"},
	Short:   "Generate responses using AI models",
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx := context.Background()
		var warnings []string

		if err := godotenv.Load(); err != nil {
			warnings = append(warnings, "No .env file found")
		}

		inputs, err := parseInputs()
		if err != nil {
			return formatOutput(jsonOutput, "", fmt.Errorf("input validation failed: %w", err), warnings)
		}

		provider, err := getProvider(providerFlag, apiKeyFlag)
		if err != nil {
			return formatOutput(jsonOutput, "", fmt.Errorf("provider setup failed: %w", err), warnings)
		}

		if err := validateCapabilities(provider, inputs); err != nil {
			return formatOutput(jsonOutput, "", err, warnings)
		}

		result, err := provider.Generate(ctx, inputs)
		if err != nil {
			return formatOutput(jsonOutput, "", err, warnings)
		}

		return formatOutput(jsonOutput, result, nil, warnings)
	},
}

func formatOutput(jsonFlag bool, content string, err error, warnings []string) error {
	if jsonFlag {
		output := CLIOutput{
			Success:  err == nil,
			Content:  content,
			Error:    "",
			Warnings: warnings,
		}
		if err != nil {
			output.Error = err.Error()
		}

		jsonData, _ := json.Marshal(output)
		fmt.Println(string(jsonData))
		return nil
	}

	if err != nil {
		return err
	}
	fmt.Println(content)
	return nil
}

func init() {
	generateCmd.Flags().StringVarP(&promptFlag, "prompt", "p", "", "Text prompt (required)")
	generateCmd.Flags().StringSliceVarP(&imagesFlag, "images", "i", []string{}, "Image paths")
	generateCmd.Flags().StringVar(&providerFlag, "provider", "openai", "AI provider (openai|deepseek)")
	generateCmd.Flags().StringVarP(&apiKeyFlag, "apikey", "k", "", "API key (overrides environment variable)")
	generateCmd.Flags().BoolVar(&jsonOutput, "json", false, "Output in JSON format")

	generateCmd.MarkFlagRequired("prompt")
	rootCmd.AddCommand(generateCmd)
}

func parseInputs() (providers.Inputs, error) {
	var imageReaders []providers.FileInput

	for _, imgPath := range imagesFlag {
		file, err := os.Open(imgPath)
		if err != nil {
			return providers.Inputs{}, fmt.Errorf("failed to open image %s: %w", imgPath, err)
		}
		defer file.Close()

		imageReaders = append(imageReaders, providers.FileInput{
			Reader:   file,
			Filename: filepath.Base(imgPath),
		})
	}

	return providers.Inputs{
		Prompt: promptFlag,
		Images: imageReaders,
	}, nil
}

func getProvider(name, flagKey string) (providers.Provider, error) {
	key, err := getAPIKey(name, flagKey)
	if err != nil {
		return nil, err
	}

	config := providers.Config{
		APIKey: key,
	}

	switch name {
	case "openai":
		return providers.NewOpenAI(config), nil
	case "deepseek":
		return providers.NewDeepSeek(config), nil
	default:
		return nil, fmt.Errorf("unsupported provider: %s", name)
	}
}

func getAPIKey(provider, flagKey string) (string, error) {
	if flagKey != "" {
		return flagKey, nil
	}

	var envVar string
	switch provider {
	case "openai":
		envVar = os.Getenv("OPENAI_API_KEY")
	case "deepseek":
		envVar = os.Getenv("DEEPSEEK_API_KEY")
	default:
		return "", fmt.Errorf("unsupported provider: %s", provider)
	}

	if envVar == "" {
		return "", fmt.Errorf("API key required for %s. Set via --apikey or environment variable", provider)
	}

	return envVar, nil
}

func validateCapabilities(p providers.Provider, inputs providers.Inputs) error {
	if len(inputs.Images) > 0 && !p.Supports(providers.FeatureVision) {
		return fmt.Errorf("selected provider doesn't support image analysis")
	}
	return nil
}
