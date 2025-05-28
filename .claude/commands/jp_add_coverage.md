# Test Coverage Command
> This command helps implement comprehensive test coverage for a Python module by leveraging code reduction and multi-LLM analysis.

## EXECUTION GUIDE
This command requires exactly 2 arguments: MODULE_PATH and TEST_FILE_PATH.
Example usage: `/add-coverage path/to/module.py path/to/test_module.py`

## TOOLS USED:
- **code-reducer**: Reduces Python code by removing type hints while preserving structure
- **just-prompt**: Sends prompts to multiple LLMs for diverse solutions
- **pytest with coverage**: Measures test coverage of the module

## IMPERATIVES:
- Always use just-prompt:prompt to request help/information/solution
- Refrain from using your own API calls
- Indicate the task you are executing as feedback to the user
- Follow each task in sequence

---

## INITIALIZATION: Setup Variables and Environment

### Parse Arguments
```
VARIABLES:
  - ${MODULE_PATH}: First argument (path to Python module)
  - ${TEST_FILE_PATH}: Second argument (path to test file)
  - ${TASK_ID}: Generate a unique task ID (timestamp)
  - ${WORKING_DIR}: _test_add_coverage_task_${TASK_ID}/
  - ${PROJECT_ROOT}: To be determined in Task 1
```

VALIDATION:
- IF number of arguments != 2:
  - OUTPUT: "Error: This command requires exactly 2 arguments: MODULE_PATH TEST_FILE_PATH"
  - EXIT

### Create Working Directory
```bash
mkdir -p ${WORKING_DIR}
```

---

## TASK 1: Verify Inputs
▶️ TASK STARTING: Validating module and test file paths

### Find Relevant Paths
```
TOOL: GlobTool
PARAMETERS:
  - pattern: "**/${MODULE_PATH}"
```

```
TOOL: GlobTool
PARAMETERS:
  - pattern: "**/${TEST_FILE_PATH}"
```

```
TOOL: GlobTool
PARAMETERS:
  - pattern: "**/conftest.py"
```

### Identify Project Root
- Determine ${PROJECT_ROOT} based on common parent directory of module and test file
- Verify both files exist before proceeding

✅ TASK COMPLETED: Module and test file validated

---

## TASK 2: Create Initial Coverage Report
▶️ TASK STARTING: Generating initial test coverage report

### Run Coverage Tests
```bash
/home/memento/ClaudeCode/bin/conda run -n ClaudeCode pytest ${TEST_FILE_PATH} -v --cov=${MODULE_PATH} --cov-report=term > ${WORKING_DIR}/coverage_report.txt
```

VALIDATION:
- IF coverage is 100%:
  - OUTPUT: "Coverage is already 100% - no improvements needed"
  - EXIT

✅ TASK COMPLETED: Initial coverage report generated

---

## TASK 3: Prepare LLM Context
▶️ TASK STARTING: Creating context document for LLM analysis

### Create Context Priming File
```
TOOL: Replace
PARAMETERS:
  - file_path: ${WORKING_DIR}/taskcontextpriming.md
  - content: |
    # Code coverage to 100%
    - Review the coverage report
    - Assess the missing coverage from the report and the python module
    - Implement additional test cases to complement the coverage in the testfile
    
    ## Python Guidelines
    - Use pytest as the testing framework
    - Add proper assertions to verify behavior
    - Maintain code style consistency with existing tests
    - Follow single responsibility principle for tests
    - Add comments to explain test purpose
    - Include explicit return type annotations
    - Use modern Python 3.12 type syntax

    ## Coverage report
    ```

### Append Coverage Report
```
TOOL: Bash
PARAMETERS:
  - command: cat ${WORKING_DIR}/coverage_report.txt >> ${WORKING_DIR}/taskcontextpriming.md
```

### Process Module Content with Code Reducer
```
TOOL: View
PARAMETERS:
  - file_path: ${MODULE_PATH}
```

```
TOOL: Bash
PARAMETERS:
  - command: echo -e "\n## Python module (REDUCED)\n\`\`\`python" >> ${WORKING_DIR}/taskcontextpriming.md
```

```
TOOL: mcp__code-reducer__process_python
PARAMETERS:
  - content: ${MODULE_CONTENT_FROM_VIEW}
  - file_name: ${MODULE_PATH##*/}
  - strip_comments: false
  - strip_docstrings: false
  - strip_type_hints: true
```

```
TOOL: Bash
PARAMETERS:
  - command: echo "${REDUCED_MODULE_CONTENT}" >> ${WORKING_DIR}/taskcontextpriming.md
```

```
TOOL: Bash
PARAMETERS:
  - command: echo -e "\n\`\`\`\n" >> ${WORKING_DIR}/taskcontextpriming.md
```

### Process Test File Content with Code Reducer
```
TOOL: View
PARAMETERS:
  - file_path: ${TEST_FILE_PATH}
```

```
TOOL: Bash
PARAMETERS:
  - command: echo -e "\n## Testfile (REDUCED)\n\`\`\`python" >> ${WORKING_DIR}/taskcontextpriming.md
```

```
TOOL: mcp__code-reducer__process_python
PARAMETERS:
  - content: ${TEST_FILE_CONTENT_FROM_VIEW}
  - file_name: ${TEST_FILE_PATH##*/}
  - strip_comments: false
  - strip_docstrings: false
  - strip_type_hints: true
```

```
TOOL: Bash
PARAMETERS:
  - command: echo "${REDUCED_TEST_FILE_CONTENT}" >> ${WORKING_DIR}/taskcontextpriming.md
```

```
TOOL: Bash
PARAMETERS:
  - command: echo -e "\n\`\`\`\n" >> ${WORKING_DIR}/taskcontextpriming.md
```

### Append Original Module Content for Reference
```
TOOL: Bash
PARAMETERS:
  - command: echo -e "\n## Original Python module\n\`\`\`python" >> ${WORKING_DIR}/taskcontextpriming.md
```

```
TOOL: Bash
PARAMETERS:
  - command: cat ${MODULE_PATH} >> ${WORKING_DIR}/taskcontextpriming.md
```

```
TOOL: Bash
PARAMETERS:
  - command: echo -e "\n\`\`\`\n" >> ${WORKING_DIR}/taskcontextpriming.md
```

### Append Any Additional Relevant Files
```
TOOL: Bash
PARAMETERS:
  - command: echo -e "\n## Optional files\n" >> ${WORKING_DIR}/taskcontextpriming.md
```

✅ TASK COMPLETED: Context document prepared

---

## TASK 4: Request Help from Multiple LLMs
▶️ TASK STARTING: Requesting analysis from LLMs

### Create Output Directory
```bash
mkdir -p ${WORKING_DIR}/llm_additions/
```

### Query LLMs for Coverage Improvements
```
TOOL: mcp__just-prompt__prompt_from_file_to_file
PARAMETERS:
  - file: ${WORKING_DIR}/taskcontextpriming.md
  - models_prefixed_by_provider: ["openai:o3-mini", "gemini:gemini-2.0-flash-thinking-exp", "gemini:gemini-2.5-pro-preview-03-25"]
  - output_dir: ${WORKING_DIR}/llm_additions/
```

✅ TASK COMPLETED: LLM analysis received

---

## TASK 5: Synthesize and Implement Solutions
▶️ TASK STARTING: Synthesizing LLM recommendations

### Identify LLM Output Files
```
TOOL: GlobTool
PARAMETERS:
  - pattern: "${WORKING_DIR}/llm_additions/*.txt"
```

### Read and Compare LLM Outputs
```
TOOL: mcp__just-prompt__prompt
PARAMETERS:
  - text: |
      # Test Coverage Synthesis Task
      
      I have received multiple suggestions from different LLMs for improving test coverage 
      of a Python module. Please help me synthesize these into a cohesive solution:
      
      [LLM_OUTPUTS]
      
      ## Task
      - Identify the best test cases from each solution
      - Eliminate any redundancy
      - Ensure consistent style with the existing tests
      - Create a unified solution that maximizes code coverage
      - DO NOT INCLUDE CODE COMMENTS ABOUT THE SYNTHESIS PROCESS IN THE FINAL CODE
      
      ## Output Format
      ```python
      # Test additions go here without any explanatory comments about the synthesis
      ```
```

### Backup Existing Test File
```bash
cp ${TEST_FILE_PATH} ${TEST_FILE_PATH}.backup
```

### Implement Synthesized Solution
```
TOOL: View
PARAMETERS:
  - file_path: ${TEST_FILE_PATH}
```

```
TOOL: Replace
PARAMETERS:
  - file_path: ${TEST_FILE_PATH}
  - content: [SYNTHESIZED_TEST_FILE_WITH_ADDITIONS]
```

✅ TASK COMPLETED: Coverage improvements implemented

---

## TASK 6: Validate Final Coverage
▶️ TASK STARTING: Validating improved test coverage

### Run Updated Coverage Tests
```bash
/home/memento/ClaudeCode/bin/conda run -n ClaudeCode pytest ${TEST_FILE_PATH} -v --cov=${MODULE_PATH} --cov-report=term > ${WORKING_DIR}/updated_coverage_report.txt
```

VALIDATION:
- IF coverage is still not 100%:
  - GOTO: TASK 2 (with updated test file)
- ELSE:
  - OUTPUT: "Coverage successfully improved to 100%"

✅ TASK COMPLETED: Coverage validation complete

---

## TASK 7: Final Cleanup
▶️ TASK STARTING: Finalizing and cleaning up

### Fix Any Failing Tests
```
TOOL: View
PARAMETERS:
  - file_path: ${TEST_FILE_PATH}
```

- IF tests are failing:
  - Mark problematic tests with `@pytest.mark.skip(reason="...")`
  - Ensure all tests pass before proceeding

### Remove Working Directory
```bash
rm -rf ${WORKING_DIR}
```

✅ TASK COMPLETED: Cleanup complete

OUTPUT: "Test coverage improvement completed successfully"