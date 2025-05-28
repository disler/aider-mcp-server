# Just-Prompt with Mode
> This command runs just-prompt with appropriate concise/standard mode directives.

## EXECUTION GUIDE
This command requires at least 1 argument:
1. PROMPT: The prompt to send to the LLM
Additional arguments are passed to just-prompt

Optional flags:
- --force-standard: Ignore concise mode and use standard mode
- --force-concise: Force concise mode even if globally disabled
- --model=MODEL_NAME: Specify LLM model (defaults to appropriate tier based on task)

Example usage:
- `/jp_with_mode "Explain the visitor pattern"`

## TOOLS USED:
- **just-prompt**: For sending prompts to LLMs
- **concise_mode**: For respecting token usage settings

---

## TASK 1: Determine Mode Settings
▶️ TASK STARTING: Checking current mode settings

### Check Current Mode
```bash
# Check if concise mode is enabled
if [ -f ~/.claude/concise_mode ]; then
  CONCISE_MODE=$(cat ~/.claude/concise_mode)
else
  CONCISE_MODE="off"
fi

# Check scope if enabled
if [ -f ~/.claude/concise_mode_scope ]; then
  CONCISE_SCOPE=$(cat ~/.claude/concise_mode_scope)
else
  CONCISE_SCOPE="all"
fi

# Check for override flags
if [[ "$*" == *"--force-standard"* ]]; then
  CONCISE_MODE="off"
  echo "Forcing standard mode"
elif [[ "$*" == *"--force-concise"* ]]; then
  CONCISE_MODE="on"
  CONCISE_SCOPE="all"
  echo "Forcing concise mode"
fi
```

✅ TASK COMPLETED: Mode settings determined

---

## TASK 2: Determine Prompt and Model
▶️ TASK STARTING: Preparing prompt with directives

### Extract User Prompt
```bash
# Extract the primary prompt (first argument)
USER_PROMPT="$1"

# Remove any override flags
CLEANED_ARGS=$(echo "$@" | sed 's/--force-standard//g' | sed 's/--force-concise//g')
```

### Determine Default Model
```bash
# Check if a specific model is requested
if [[ "$*" == *"--model="* ]]; then
  # Model is already specified in arguments
  MODEL_SPECIFIED=true
else
  MODEL_SPECIFIED=false

  # Determine complexity of task
  PROMPT_LOWERCASE=$(echo "$USER_PROMPT" | tr '[:upper:]' '[:lower:]')

  # Check for keywords indicating complexity
  if [[ "$PROMPT_LOWERCASE" == *"architect"* ]] ||
     [[ "$PROMPT_LOWERCASE" == *"design"* ]] ||
     [[ "$PROMPT_LOWERCASE" == *"complex"* ]] ||
     [[ "$PROMPT_LOWERCASE" == *"explain"* ]] ||
     [[ "$PROMPT_LOWERCASE" == *"optimize"* ]]; then
    # Complex task detected, use smart LLM
    DEFAULT_MODEL="gemini:gemini-2.5-pro-preview-03-25"
  else
    # Simple task, use cheap LLM
    DEFAULT_MODEL="gemini:gemini-1.5-pro"
  fi
fi
```

### Prepare Mode-Appropriate Directive
```bash
# Create temporary prompt file
TEMP_PROMPT_FILE=$(mktemp)

# Add appropriate directive based on mode
if [ "$CONCISE_MODE" = "on" ]; then
  if [ "$CONCISE_SCOPE" = "code-only" ] && [[ "$PROMPT_LOWERCASE" == *"code"* ]]; then
    # Add concise code directive
    cat ~/.claude/llm_directives/concise_code.md > "$TEMP_PROMPT_FILE"
    echo -e "\n\n" >> "$TEMP_PROMPT_FILE"
  elif [ "$CONCISE_SCOPE" = "all" ]; then
    # Add concise response directive
    cat ~/.claude/llm_directives/concise_response.md > "$TEMP_PROMPT_FILE"
    echo -e "\n\n" >> "$TEMP_PROMPT_FILE"
  fi
else
  # Standard mode
  if [[ "$PROMPT_LOWERCASE" == *"code"* ]]; then
    # Add standard code directive for code-related prompts
    cat ~/.claude/llm_directives/standard_mode.md > "$TEMP_PROMPT_FILE"
    echo -e "\n\n" >> "$TEMP_PROMPT_FILE"
  fi
fi

# Add user prompt
echo "$USER_PROMPT" >> "$TEMP_PROMPT_FILE"
```

✅ TASK COMPLETED: Prompt prepared with appropriate directive

---

## TASK 3: Execute Just-Prompt
▶️ TASK STARTING: Running just-prompt with prepared prompt

### Run Just-Prompt
```
TOOL: mcp__just-prompt__prompt_from_file
PARAMETERS:
  - file: ${TEMP_PROMPT_FILE}
  - models_prefixed_by_provider: ${MODEL_SPECIFIED ? CLEANED_ARGS : [DEFAULT_MODEL]}
```

### Clean Up
```bash
# Remove temporary file
rm "$TEMP_PROMPT_FILE"
```

✅ TASK COMPLETED: Just-prompt executed with mode-appropriate directive
