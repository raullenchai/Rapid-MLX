# Contributing

We welcome contributions to rapid-mlx!

## Getting Started

```bash
# Clone the repository
git clone https://github.com/raullenchai/Rapid-MLX.git
cd Rapid-MLX

# Install with dev dependencies
pip install -e ".[dev]"
```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_paged_cache.py -v

# Run with coverage
pytest --cov=vllm_mlx tests/
```

### Code Style

```bash
# Lint and format
ruff check .
ruff format --check .
```

### Running Benchmarks

```bash
# LLM benchmark
rapid-mlx-bench --model mlx-community/Qwen3-0.6B-8bit

# Image benchmark
rapid-mlx-bench --model mlx-community/Qwen3-VL-8B-Instruct-4bit

# Video benchmark
rapid-mlx-bench --model mlx-community/Qwen3-VL-8B-Instruct-4bit --video
```

## Areas for Contribution

- **Bug fixes** - Fix issues and improve stability
- **Performance optimizations** - Improve inference speed
- **New features** - Add functionality
- **Documentation** - Improve docs and examples
- **Benchmarks** - Test on different Apple Silicon chips
- **Model support** - Test and add new models

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests to ensure they pass
5. Submit a pull request

## Code Structure

See [Architecture](architecture.md) for details on the codebase structure.

## Testing on Different Hardware

If you have access to different Apple Silicon chips (M1, M2, M3, M4), benchmark results are valuable:

```bash
rapid-mlx-bench --model mlx-community/Qwen3-0.6B-8bit --output results_m4.json
```

## Questions?

Open an issue at [GitHub Issues](https://github.com/raullenchai/Rapid-MLX/issues).
