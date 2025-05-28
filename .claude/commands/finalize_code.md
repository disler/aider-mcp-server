# Finalize Code
> This command takes concise code and produces a fully documented version with proper docstrings, typing, etc.

## EXECUTION GUIDE
This command requires at least 2 arguments:
1. TASK_DESCRIPTION: Description of what the code does or requirements
2. FILE_PATH: Path to the code file to finalize

Optional flags:
- --model=MODEL_NAME: Specify a different AI model (default: gemini:gemini-2.5-pro-preview-03-25)

Example usage: 
`/finalize_code "Fibonacci calculator implementation" src/math/fibonacci.py`

## TOOLS USED:
- **just-prompt**: For generating complete documentation
- **concise_mode**: For temporarily disabling concise mode

---

## TASK 1: Read Input Code
▶️ TASK STARTING: Reading code to finalize

### Verify File Path
```
TOOL: View
PARAMETERS:
  - file_path: ${FILE_PATH}
```

### Store Original Code
```bash
# Create temporary directory
TEMP_DIR=$(mktemp -d)

# Store original code
cp "${FILE_PATH}" "${TEMP_DIR}/original_code.py"
```

✅ TASK COMPLETED: Original code stored

---

## TASK 2: Determine Model and Prepare Prompt
▶️ TASK STARTING: Preparing finalization prompt

### Determine Model
```bash
# Check if a specific model is requested
if [[ "$*" == *"--model="* ]]; then
  # Extract model from arguments
  MODEL=$(echo "$*" | grep -o "\--model=[^ ]*" | cut -d'=' -f2)
else
  # Use default smart model for finalization
  MODEL="gemini:gemini-2.5-pro-preview-03-25"
fi
```

### Prepare Finalization Prompt
```bash
# Create prompt file
cat > "${TEMP_DIR}/finalize_prompt.md" << EOL
# CODE FINALIZATION DIRECTIVE
Please take the following code and transform it into production-ready code with:

1. Complete docstrings in reStructuredText format following PEP 257
2. Full type hints following Python 3.12 conventions (using | for unions, etc.)
3. Clear explanatory comments for complex sections
4. Descriptive variable names
5. PEP 8 style compliance
6. Proper error handling with specific exception types
7. Defensive coding practices

DO NOT change the core functionality or algorithm.
DO NOT introduce new features.
DO optimize and improve the code quality and readability.

## Original Code
\`\`\`python
$(cat "${TEMP_DIR}/original_code.py")
\`\`\`

## Task Description
${TASK_DESCRIPTION}

## Required Output Format
Please provide ONLY the complete finalized code with no explanations.
\`\`\`python
# Your finalized code goes here
\`\`\`
EOL
```

✅ TASK COMPLETED: Finalization prompt prepared

---

## TASK 3: Generate Finalized Code
▶️ TASK STARTING: Generating fully documented code

### Execute LLM Request
```
TOOL: mcp__just-prompt__prompt_from_file
PARAMETERS:
  - file: ${TEMP_DIR}/finalize_prompt.md
  - models_prefixed_by_provider: ["${MODEL}"]
```

### Extract Finalized Code
```bash
# Extract just the code from the LLM response
cat "${TEMP_DIR}/llm_response.txt" | sed -n '/```python/,/```/p' | sed '1d;$d' > "${TEMP_DIR}/finalized_code.py"
```

✅ TASK COMPLETED: Finalized code generated

---

## TASK 4: Review and Apply Changes
▶️ TASK STARTING: Reviewing and applying changes

### Generate Diff for Review
```bash
# Generate diff between original and finalized code
diff -u "${TEMP_DIR}/original_code.py" "${TEMP_DIR}/finalized_code.py" > "${TEMP_DIR}/changes.diff" || true
```

### Display Diff for User Review
```bash
# Output the diff for review
echo "## Changes to be applied:"
cat "${TEMP_DIR}/changes.diff"

# Prompt for confirmation
echo -e "\nDo you want to apply these changes? (y/n)"
read -r CONFIRM
```

### Apply Changes if Confirmed
```bash
if [ "$CONFIRM" = "y" ] || [ "$CONFIRM" = "Y" ]; then
  # Apply the finalized code
  cp "${TEMP_DIR}/finalized_code.py" "${FILE_PATH}"
  echo "✅ Changes applied successfully to ${FILE_PATH}"
else
  echo "❌ Changes not applied"
fi
```

✅ TASK COMPLETED: Changes reviewed and applied if confirmed

---

## TASK 5: Cleanup
▶️ TASK STARTING: Cleaning up temporary files

### Remove Temporary Directory
```bash
rm -rf "${TEMP_DIR}"
```

✅ TASK COMPLETED: Cleanup complete