# Toggle Concise Mode
> This command toggles "concise mode" which reduces output tokens while maintaining response quality.

## EXECUTION GUIDE
This command takes an optional argument:
- ON/OFF: Turn concise mode on or off (default: toggle current state)
- --code-only: Only apply concise mode to code generation (not explanations)

Example usage: 
- `/concise_mode on` - Turn concise mode on
- `/concise_mode off` - Turn concise mode off
- `/concise_mode` - Toggle current state

## TOOLS USED:
- None (uses internal settings)

---

## TASK 1: Check Current Mode
▶️ TASK STARTING: Checking current concise mode setting

### Read Current Setting
```bash
# Check if the concise mode setting exists
if [ -f ~/.claude/concise_mode ]; then
  current_mode=$(cat ~/.claude/concise_mode)
else
  # Default to off if not set
  current_mode="off"
fi

# Store the current mode
echo "Current concise mode: $current_mode"
```

✅ TASK COMPLETED: Current mode checked

---

## TASK 2: Update Mode Setting
▶️ TASK STARTING: Updating concise mode setting

### Determine New Mode
```bash
# Parse the argument
if [ -z "$1" ]; then
  # No argument provided, toggle current mode
  if [ "$current_mode" = "on" ]; then
    new_mode="off"
  else
    new_mode="on"
  fi
else
  # Use provided argument (convert to lowercase)
  new_mode=$(echo "$1" | tr '[:upper:]' '[:lower:]')
  
  # Validate input
  if [ "$new_mode" != "on" ] && [ "$new_mode" != "off" ]; then
    echo "Error: Invalid argument. Use 'on' or 'off'."
    exit 1
  fi
fi
```

### Save New Setting
```bash
mkdir -p ~/.claude
echo "$new_mode" > ~/.claude/concise_mode

# Check if --code-only flag is set
if [[ "$*" == *"--code-only"* ]]; then
  echo "code-only" > ~/.claude/concise_mode_scope
else
  echo "all" > ~/.claude/concise_mode_scope
fi
```

✅ TASK COMPLETED: Mode setting updated

---

## TASK 3: Apply Concise Mode Settings
▶️ TASK STARTING: Applying concise mode settings

### Define Concise Mode Behavior
```
IF concise_mode is ON:
  - Remove explanatory preambles/postambles in responses
  - Eliminate unnecessary acknowledgments 
  - Omit docstrings in generated code (unless explicitly requested)
  - Skip typing hints in generated code (unless required for functionality)
  - Remove comments in generated code (except critical ones)
  - Avoid repeating information already provided by the user
  - Use abbreviated variable names in temporary code
  - Format output in the most token-efficient manner
  - Focus on direct answers and code solutions
  
IF concise_mode_scope is code-only:
  - Apply concise formatting only to code generation
  - Maintain normal explanation quality
```

### Output New Status
```
OUTPUT: "Concise mode is now ${new_mode}"

IF new_mode is ON:
  IF concise_mode_scope is code-only:
    OUTPUT: "Code will be generated without docstrings, most comments, and type hints"
  ELSE:
    OUTPUT: "All responses will be concise, focusing on solutions with minimal explanation"
ELSE:
  OUTPUT: "Standard response mode with normal level of detail"
```

✅ TASK COMPLETED: Settings applied

---

## TASK 4: Update Prime Context and Create LLM Directives
▶️ TASK STARTING: Creating directives for external LLMs

### Update Context With Mode
```bash
# Get current settings
mode=$(cat ~/.claude/concise_mode 2>/dev/null || echo "off")
scope=$(cat ~/.claude/concise_mode_scope 2>/dev/null || echo "all")

# Add to environment
export CLAUDE_CONCISE_MODE="$mode"
export CLAUDE_CONCISE_SCOPE="$scope"

# Output for session
if [ "$mode" = "on" ]; then
  if [ "$scope" = "code-only" ]; then
    echo "Concise mode is ON (code-only) - Generating minimal code without docstrings/comments/type hints"
  else
    echo "Concise mode is ON (all responses) - Providing direct solutions with minimal explanation"
  fi
fi
```

### Create LLM Directive Files
```bash
# Create directives directory
mkdir -p ~/.claude/llm_directives

# Create concise code directive
cat > ~/.claude/llm_directives/concise_code.md << 'EOL'
# CONCISE CODE DIRECTIVE
Please provide code that is minimal and focused on functionality, with the following characteristics:
- NO docstrings unless explicitly requested
- NO type hints unless they are essential for functionality
- NO explanatory comments
- Minimal variable names
- No explanations about the code implementation
- Focus only on working, efficient code

This directive has been applied to save tokens. If you need more detailed code with documentation, please explicitly request it.
EOL

# Create concise response directive
cat > ~/.claude/llm_directives/concise_response.md << 'EOL'
# CONCISE RESPONSE DIRECTIVE
Please provide extremely concise responses with the following characteristics:
- Answer directly without preamble or pleasantries
- Use minimal words necessary to convey information
- No acknowledgments of understanding the question
- No explanations unless explicitly requested
- No repetition of information from the question
- Single-phrase answers when possible

This directive has been applied to save tokens. If you need more detailed explanations, please explicitly request them.
EOL

# Create standard mode directive
cat > ~/.claude/llm_directives/standard_mode.md << 'EOL'
# STANDARD MODE DIRECTIVE
Please provide well-documented code with the following characteristics:
- Full docstrings in reStructuredText format
- Complete type hints following Python 3.12 conventions
- Explanatory comments for complex sections
- Descriptive variable names
- Follow PEP 8 style guidelines
- Emphasize readability and maintainability

This directive requests fully documented, production-quality code.
EOL
```

✅ TASK COMPLETED: LLM directives created