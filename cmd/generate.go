package cmd

import (
	"context"
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
)

var generateCmd = &cobra.Command{
	Use:     "generate",
	Aliases: []string{"gen", "ask"},
	Short:   "Generate responses using AI models",
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx := context.Background()

		// Load environment variables from .env file
		if err := godotenv.Load(); err != nil {
			fmt.Println("Warning: No .env file found")
		}

		inputs, err := parseInputs()
		if err != nil {
			return fmt.Errorf("input validation failed: %w", err)
		}

		provider, err := getProvider(providerFlag, apiKeyFlag)
		if err != nil {
			return fmt.Errorf("provider setup failed: %w", err)
		}

		if err := validateCapabilities(provider, inputs); err != nil {
			return err
		}

		result, err := provider.Generate(ctx, inputs)
		if err != nil {
			return fmt.Errorf("generation failed: %w", err)
		}

		fmt.Println(result)
		return nil
	},
}

func init() {
	generateCmd.Flags().StringVarP(&promptFlag, "prompt", "p", "", "Text prompt (required)")
	generateCmd.Flags().StringSliceVarP(&imagesFlag, "images", "i", []string{}, "Image paths")
	generateCmd.Flags().StringVar(&providerFlag, "provider", "openai", "AI provider (openai|deepseek)")
	generateCmd.Flags().StringVarP(&apiKeyFlag, "apikey", "k", "", "API key (overrides environment variable)")

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
