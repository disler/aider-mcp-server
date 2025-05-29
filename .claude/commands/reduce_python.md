# Python Code Reducer Command
> This command reduces a Python file's content by removing type hints, docstrings, or comments, showing reduction statistics.

## EXECUTION GUIDE
This command requires at least 1 argument: PYTHON_FILE_PATH
Optional flags: --strip-comments, --strip-docstrings, --strip-all (equivalent to both)
Example usage: `/reduce_python path/to/file.py --strip-comments`

## TOOLS USED:
- **code-reducer**: Reduces Python code while preserving functionality
- **wc**: Calculates line and character counts for statistics

## IMPERATIVES:
- Display clear reduction statistics
- Create a preview of the reduced file
- Return both percentage reduction and absolute size changes

---

## INITIALIZATION: Setup Variables and Environment

### Parse Arguments
```
VARIABLES:
  - ${PYTHON_FILE_PATH}: First argument (path to Python file)
  - ${TASK_ID}: Generate a unique task ID (timestamp)
  - ${WORKING_DIR}: _python_reducer_task_${TASK_ID}/
  - ${STRIP_COMMENTS}: false by default, true if --strip-comments flag
  - ${STRIP_DOCSTRINGS}: false by default, true if --strip-docstrings flag
  - ${STRIP_TYPE_HINTS}: true by default
```

VALIDATION:
- IF number of arguments < 1:
  - OUTPUT: "Error: This command requires at least 1 argument: PYTHON_FILE_PATH"
  - EXIT

### Create Working Directory
```bash
mkdir -p ${WORKING_DIR}
```

---

## TASK 1: Verify Input File
▶️ TASK STARTING: Validating Python file path

### Find File Path
```
TOOL: GlobTool
PARAMETERS:
  - pattern: "**/${PYTHON_FILE_PATH}"
```

### Check File Existence and Type
```
TOOL: View
PARAMETERS:
  - file_path: ${PYTHON_FILE_PATH}
```

VALIDATION:
- IF file does not exist:
  - OUTPUT: "Error: Python file not found at ${PYTHON_FILE_PATH}"
  - EXIT
- IF file is not a Python file:
  - OUTPUT: "Error: File does not appear to be a Python file"
  - EXIT

✅ TASK COMPLETED: Python file validated

---

## TASK 2: Process Original File Statistics
▶️ TASK STARTING: Calculating original file statistics

### Count Lines and Characters
```bash
wc -l ${PYTHON_FILE_PATH} > ${WORKING_DIR}/original_stats.txt
wc -c ${PYTHON_FILE_PATH} >> ${WORKING_DIR}/original_stats.txt
```

### Get Original File Content
```
TOOL: View
PARAMETERS:
  - file_path: ${PYTHON_FILE_PATH}
```

✅ TASK COMPLETED: Original file statistics calculated

---

## TASK 3: Apply Code Reduction
▶️ TASK STARTING: Reducing Python code

### Process Python with Code Reducer
```
TOOL: mcp__code-reducer__process_python
PARAMETERS:
  - content: ${ORIGINAL_FILE_CONTENT}
  - file_name: ${PYTHON_FILE_PATH##*/}
  - strip_comments: ${STRIP_COMMENTS}
  - strip_docstrings: ${STRIP_DOCSTRINGS}
  - strip_type_hints: ${STRIP_TYPE_HINTS}
```

### Save Reduced Content to File
```
TOOL: Replace
PARAMETERS:
  - file_path: ${WORKING_DIR}/reduced_file.py
  - content: ${REDUCED_CONTENT}
```

✅ TASK COMPLETED: Code reduction applied

---

## TASK 4: Calculate Reduction Statistics
▶️ TASK STARTING: Calculating reduction statistics

### Count Lines and Characters in Reduced File
```bash
wc -l ${WORKING_DIR}/reduced_file.py > ${WORKING_DIR}/reduced_stats.txt
wc -c ${WORKING_DIR}/reduced_file.py >> ${WORKING_DIR}/reduced_stats.txt
```

### Calculate Reduction Percentages
```bash
original_lines=$(grep -oP '^\s*\d+' ${WORKING_DIR}/original_stats.txt | head -1)
reduced_lines=$(grep -oP '^\s*\d+' ${WORKING_DIR}/reduced_stats.txt | head -1)
original_chars=$(grep -oP '^\s*\d+' ${WORKING_DIR}/original_stats.txt | tail -1)
reduced_chars=$(grep -oP '^\s*\d+' ${WORKING_DIR}/reduced_stats.txt | tail -1)

line_reduction=$((original_lines - reduced_lines))
line_reduction_percent=$(echo "scale=2; ($line_reduction / $original_lines) * 100" | bc)

char_reduction=$((original_chars - reduced_chars))
char_reduction_percent=$(echo "scale=2; ($char_reduction / $original_chars) * 100" | bc)

echo "Original lines: $original_lines" > ${WORKING_DIR}/reduction_summary.txt
echo "Reduced lines: $reduced_lines" >> ${WORKING_DIR}/reduction_summary.txt
echo "Line reduction: $line_reduction ($line_reduction_percent%)" >> ${WORKING_DIR}/reduction_summary.txt
echo "" >> ${WORKING_DIR}/reduction_summary.txt
echo "Original characters: $original_chars" >> ${WORKING_DIR}/reduction_summary.txt
echo "Reduced characters: $reduced_chars" >> ${WORKING_DIR}/reduction_summary.txt
echo "Character reduction: $char_reduction ($char_reduction_percent%)" >> ${WORKING_DIR}/reduction_summary.txt
```

✅ TASK COMPLETED: Reduction statistics calculated

---

## TASK 5: Display Results
▶️ TASK STARTING: Preparing result display

### Generate Head Preview of Reduced Code
```bash
head -n 20 ${WORKING_DIR}/reduced_file.py > ${WORKING_DIR}/preview.txt
```

### Format Output
```
TOOL: mcp__just-prompt__prompt
PARAMETERS:
  - text: |
      # Format Python Code Reduction Results

      Please format the following code reduction summary and preview in a clean, readable format:

      ## Summary
      [REDUCTION_SUMMARY]

      ## Preview of Reduced Code
      ```python
      [CODE_PREVIEW]
      ```

      Format this as a nice markdown output that highlights the key statistics and shows the code preview
      with proper formatting.
```

### Display to User
```
OUTPUT: Formatted summary and preview of reduced code
```

✅ TASK COMPLETED: Results displayed

---

## TASK 6: Cleanup
▶️ TASK STARTING: Cleaning up temporary files

### Copy Reduced File to Current Directory (Optional)
```bash
# Only if user requested saved output
cp ${WORKING_DIR}/reduced_file.py ./${PYTHON_FILE_PATH##*/}.reduced.py
```

### Remove Working Directory
```bash
rm -rf ${WORKING_DIR}
```

✅ TASK COMPLETED: Cleanup complete
