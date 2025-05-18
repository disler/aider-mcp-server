# Contributing to Aider MCP Server

## Development Setup

1. Clone the repository
2. Install development dependencies with `hatch shell`
3. Install pre-commit hooks: `hatch run dev:install-hooks`

## Code Formatting

To ensure consistency between local development and CI:

### Before Committing

Always run the formatter before committing:

```bash
hatch run dev:format
```

This will:
- Sort imports with `isort`
- Format code with `ruff format`
- Fix linting issues with `ruff check --fix`

### Linting

To run all linters without fixing:

```bash
hatch run dev:lint
```

### Running Everything

To format, lint, and test in one command:

```bash
hatch run dev:all
```

### Manual Commands

If you need to run individual tools:

```bash
# Format imports
hatch -e dev run isort . --profile black

# Format code
hatch -e dev run ruff format .

# Fix linting issues
hatch -e dev run ruff check --fix .

# Run all pre-commit checks
hatch -e dev run pre-commit run --all-files
```

## Writing Tests

- Place test files in the `tests/` directory
- Name test files with `test_` prefix
- Security checks are relaxed for test files (S101, S108)

## Pull Requests

1. Create a feature branch from `development`
2. Make your changes
3. Run `hatch run dev:all` to format, lint, and test
4. Commit your changes
5. Open a PR against the `development` branch

## CI/CD

The CI will:
1. Run pre-commit hooks on all files
2. Fail if any formatting changes are needed
3. Run tests across multiple Python versions

To avoid CI failures, always run `hatch run dev:format` or `hatch run dev:lint` before pushing.
