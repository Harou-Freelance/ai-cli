# AI-CLI

A unified command-line interface for interacting with multiple AI APIs (such as OpenAI, DeepSeek...).

## Installation

```sh
# Clone the repository
git clone https://github.com/harou24/ai-cli.git
cd ai-cli

# Build the CLI
make build

# Run the CLI
./ai-cli --help
```

## Usage

### Generate AI Responses

```sh
./ai-cli generate -p "What is AI?" --provider openai
```

### List Available Models

```sh
./ai-cli models --provider openai --json
```

## Command Reference

### `generate` Command

| Flag              | Description                        | Required |
|------------------|--------------------------------|----------|
| `-p/--prompt`    | Text prompt                     | Yes      |
| `-i/--images`    | Image paths (comma-separated)   | No       |
| `--provider`     | AI provider (openai/deepseek)   | No       |
| `-k/--apikey`    | Override API key                | No       |
| `--json`         | Output in JSON format           | No       |

### `models` Command

| Flag          | Description                             |
|--------------|---------------------------------|
| `--provider` | Filter by provider (openai/deepseek) |
| `--json`     | Output in JSON format               |

## Provider Capabilities

| Provider  | Text Generation | Image Analysis | Model Listing |
|-----------|----------------|----------------|---------------|
| OpenAI    | ✓              | ✓              | ✓             |
| DeepSeek  | ✓              | ✗              | ✗             |

## Environment Variables

| Variable         | Description                   |
|-----------------|-----------------------------|
| `OPENAI_API_KEY` | API key for OpenAI          |
| `DEEPSEEK_API_KEY` | API key for DeepSeek      |

Set them in your `.env` file or export them in your shell:

```sh
export OPENAI_API_KEY=your_openai_key
export DEEPSEEK_API_KEY=your_deepseek_key
```

## License

MIT License.

