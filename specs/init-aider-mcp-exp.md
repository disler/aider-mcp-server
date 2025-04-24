# Aider Model Context Protocol (MCP) Experimental Server
> Here we detail how we'll build the experimental ai coding aider mcp server.

## Why?
Claude Code is a new, powerful agentic coding tool that is currently in beta. It's great but it's incredibly expensive.
We can offload some of the work to a simpler ai coding tool: Aider. The original AI Coding Assistant.

By discretely offloading work to Aider, we can not only reduce costs but use Claude Code (and auxillary LLM calls combined with aider) to better create more, reliable code through multiple - focused - LLM calls.

## Resources to ingest
> To understand how we'll build this, READ these files

ai_docs/just-prompt-example-mcp-server.xml
ai_docs/programmable-aider-documentation.md

## Implementation Notes

- We want to mirror the exact structure of the just-prompt codebase as closely as possible. Minus of course the tools that are specific to just-prompt.
- Every atom must be tested in a respective tests/*_test.py file.
- every atom/tools/*.py must only have a single responsibility - one method.
- when we run aider run in no commit mode, we should not commit any changes to the codebase.
- if architect_model is not provided, don't use architect mode.

## Application Structure

- src/
  - aider_mcp_server/
    - __init__.py
    - __main__.py
    - server.py
      - serve(editor_model: str = DEFAULT_EDITOR_MODEL, current_working_dir: str = ".", architect_model: str = None) -> None
    - atoms/
      - __init__.py
      - tools/
        - __init__.py
        - aider_ai_code.py
          - code_with_aider(ai_coding_prompt: str, relative_editable_files: List[str], relative_readonly_files: List[str] = []) -> str
            - runs one shot aider based on ai_docs/programmable-aider-documentation.md
            - outputs 'success' or 'failure'
        - aider_list_models.py
          - list_models(substring: str) -> List[str]
            - calls aider.models.fuzzy_match_models(substr: str) and returns the list of models
      - utils.py
        - DEFAULT_EDITOR_MODEL = "gemini/gemini-2.5-pro-exp-03-25"
        - DEFAULT_ARCHITECT_MODEL = "gemini/gemini-2.5-pro-exp-03-25"
      - data_types.py
    - tests/
      - __init__.py
      - atoms/
        - __init__.py
        - tools/
          - __init__.py
          - test_aider_ai_code.py
            - here create tests for basic 'math' functionality: 'add, 'subtract', 'multiply', 'divide'. Use temp dirs.
          - test_aider_list_models.py
            - here create a real call to list_models(openai) and assert gpt-4o substr in list.

## Commands

- if for whatever reason you need additional python packages use `uv add <package_name>`.

## Validation
- Use `uv run pytest <path_to_test_file.py>` to run tests. Every atom/ must be tested.
- Don't mock any tests - run real LLM calls. Make sure to test for failure paths.
- At the end run `uv run aider-mcp-server --help` to validate the server is working.
