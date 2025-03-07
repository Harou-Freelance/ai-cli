package cmd

import (
	"os"

	"github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
	Use:   "ai-cli",
	Short: "AI-powered CLI for multimodal generation",
	Long: `Interactive CLI supporting text and image generation through multiple AI providers.

Examples:
  $ ai-cli generate -p "Explain quantum computing"
  $ ai-cli generate -p "Describe this image" -i photo.jpg
  $ ai-cli generate -p "Explain this diagram" -i diagram.png --provider openai`,
}

func Execute() {
	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}
