# Prime Working Context: Aider Focus
> This command sets up the ideal working context optimized for aider-based development, prioritizing aider for most code-related tasks.

## EXECUTION GUIDE
This command takes no arguments. It sets up foundational principles focused on aider-based workflows.
Example usage: `/aider_prime_context`

## TOOLS USED:
- **aider**: PRIMARY tool for code implementation, refactoring, and understanding
- **just-prompt**: Supplementary for specific queries when needed
- **code-reducer**: For optimizing code before analysis

---

## TASK 1: Set Tool Priorities for Aider-centric Workflow
▶️ TASK STARTING: Establishing tool priority hierarchy focusing on aider

### Define Primary Tools for Aider-centric Workflow
```
PRIORITIES:
1. USE AIDER EXTENSIVELY FOR CODE TASKS:
   - Aider is your PRIMARY tool for ALL code operations
   - Leverage aider's ability to understand entire codebases
   - Use aider for both exploring/understanding AND modifying code
   - Aider has file context awareness that just-prompt lacks
   - Prefer aider over just-prompt for code-related tasks wherever possible

2. OPTIMIZE AIDER EFFECTIVENESS:
   - Give aider BOTH read-only context files AND editable files
   - Prefer using architect mode with gemini-2.5-pro-preview-05-06 and gemini-2.5-flash-preview-04-17
   - Use readonly files to provide context without risk of unwanted changes
   - Always provide relevant context files to aider for better understanding
   - Prefer showing aider entire files rather than snippets
   - Let aider see test files alongside implementation files

3. ALWAYS verify the mypy/ruff/pytest standard of our project for files created by aider:
   - pixi run -e dev pre-commit run --all-files
   - pixi run -e dev pytest

4. USE just-prompt SELECTIVELY:
   - For conceptual questions (architecture, patterns, approaches)
   - When aider responses need validation
   - For non-code tasks where aider isn't optimal
   - When you need comparative opinions between different LLMs

5. CONSERVE YOUR TOKENS:
   - Focus your tokens on orchestration and file selection
   - Delegate actual code understanding and generation to aider
   - Use your reasoning for selecting which files to show to aider
   - Prefer multiple aider calls over long reasoning chains with your API
```

### Aider Strengths and Best Practices
```
AIDER ADVANTAGES:
1. FULL FILE CONTEXT:
   - Aider sees and understands entire files, not just snippets
   - It maintains awareness of the codebase structure
   - It can make coordinated changes across multiple files
   - It understands file relationships better than isolated LLM calls

2. INTERACTIVE EDITING:
   - Aider can directly apply changes to files
   - It understands code structure and maintains syntax correctness
   - It can make precise, targeted modifications
   - It can verify changes work within the broader codebase

3. PROJECT MEMORY:
   - Aider maintains context between calls about the same files
   - It can build on previous code discussions
   - It has a better understanding of project-specific patterns

4. MODEL VERSATILITY:
   - Supports multiple AI models (OpenAI, Gemini, Anthropic/Claude)
   - Default model is "gemini/gemini-2.5-pro-preview-05-06"
   - Can customize model by passing model parameter
```

### Environment Paths
```
ENVIRONMENT:
- ClaudeCode started within a conda envitonment that has hatch, pixi, uv, python
- Conda: /home/memento/ClaudeCode/bin/conda
- Package managers (always run through conda):
  * Hatch: hatch run <command>
  * Pixi: pixi run <command>
  * UV: uv run <command>
```

✅ TASK COMPLETED: Aider-focused tool priorities established

---

## TASK 2: Define Aider-centric Working Methodology
▶️ TASK STARTING: Setting up working methodology optimized for aider

### Aider-centric Implementation Workflow
```
WORKFLOW:
1. ANALYZE task requirements carefully. Use taskmaster-ai if initialized
2. SEARCH for relevant files with dispatch_agent
3. PREPARE GIT ENVIRONMENT (CRITICAL):
   - Create a dedicated branch: `git checkout -b aider/<feature-name>`
   - Ensure clean working state: `git status`
   - Commit any pending changes: `git add . && git commit -m "WIP: Before aider"`

4. PREPARE FILES FOR AIDER:
   - Identify all files aider needs to see
   - aider has token limitations, use medium sized context
   - Separate into:
     * EDITABLE files (will be modified)
     * READONLY files (provide context)

5. RUN AIDER WITH COMPREHENSIVE CONTEXT:
   - Use architect mode
   - Include ALL relevant files for context
   - Use relative_editable_files for files that will change
   - Use relative_readonly_files for context-only files
   - Provide clear, specific instructions to aider

6. REVIEW AND VALIDATE:
   - ALWAYS use BatchTool to show git diff immediately after aider operations:
     ```python
     BatchTool(
       description="Show diff of changes made by Aider",
       invocations=[{
         "tool_name": "Bash",
         "input": {
           "command": "cd /path/to/correct/project/root && git diff -- <files>",
           "description": "Display git diff of modified files from the correct repository root"
         }
       }]
     )
     ```
   - IMPORTANT: Ensure you're running git commands in the correct project repository context
   - BEWARE: aider may not commit changes or write files in wrong folder/file
   - For projects within larger repositories, include the cd command in your git operations
   - Run tests after aider makes changes
   - If tests fail, give aider the test output
   - Let aider fix its own mistakes
   - Commit working changes: `git commit -m "Implement <feature> with aider"`

7. HANDLING PROBLEMATIC CHANGES:
   - For minor issues: Fix manually or give aider specific feedback
   - For major issues: Reset with `git reset --hard HEAD` and try again
   - If completely unsatisfactory: `git checkout <original-branch>` and restart

8. USE just-prompt SELECTIVELY:
   - Only for questions aider can't handle well
   - For second opinions on architecture/design
   - For explaining concepts before going to aider
```

### Decision Framework for Aider vs Just-prompt
```
DECISION TREE:
- IF task involves code creation or modification:
  → ALWAYS use aider with proper context files

- IF task involves understanding existing code:
  → Use aider with readonly files for full context

- IF task involves fixing bugs:
  → Use aider with test files AND implementation files

- IF task involves architecture or design decisions:
  → Use just-prompt first, then implement with aider

- IF task involves improving test coverage:
  → Use aider with both test and implementation files

- IF task involves exploring a large codebase:
  → Use dispatch_agent to identify relevant files
  → THEN use aider with those files in readonly mode
```

### Troubleshooting Guide
```
TROUBLESHOOTING:
1. "No meaningful changes detected" Error:
   - Aider may not assess the cahnges it made correctly, verify other response fields
   - Check that editable files parameter is correct
   - Ensure prompt is specific enough about what changes to make
   - Verify readonly files provide sufficient context
   - Break down the task into smaller, more focused requests

2. Complex Test Scenarios:
   - Include example test files with similar mocking patterns
   - Be explicit about mocking strategy in your prompt
   - Consider implementing fixture templates first, then test cases

3. Authentication Errors:
   - API keys are loaded from .env files in various locations
   - Check that appropriate API keys are set in environment
   - Common keys: OPENAI_API_KEY, GOOGLE_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY
```

### Optimizing for Test Generation
```
TEST GENERATION OPTIMIZATIONS:
1. Incremental Approach
   - Start with basic test structure and fixtures
   - Add test cases in batches of related functionality
   - Finish with edge cases and error conditions

2. Context Enhancement
   - Include similar, well-written test files as readonly context
   - Reference particular test strategies you want replicated
   - Include project-specific test utilities or fixtures

3. Prompt Construction
   - Begin with overview of test objectives
   - List specific methods/functions to test
   - Specify expected fixtures and mocking approach
   - Highlight particular edge cases or error conditions
```

✅ TASK COMPLETED: Aider-centric methodology defined

---

## TASK 3: Configure Local Tools and Modes
▶️ TASK STARTING: Setting up tool access and response modes

### Grant Tool Permissions
```bash
# Try to detect if the local tools directory exists
LOCAL_CMD_DIR="./.claude/commands"
if [ -d "$LOCAL_CMD_DIR" ]; then
  # Copy local command definitions if they exist
  if [ -f "$LOCAL_CMD_DIR/aider_code.md" ]; then
    OUTPUT: "Found local aider_code command"
  else
    OUTPUT: "Note: Local aider_code command not found. Using global definition."
  fi

  if [ -f "$LOCAL_CMD_DIR/reduce_python.md" ]; then
    OUTPUT: "Found local reduce_python command"
  else
    OUTPUT: "Note: Local reduce_python command not found. Using global definition."
  fi
fi
```

### Trust Paths
```
TRUSTED_PATHS:
- /home/memento/ClaudeCode/bin/conda
- /home/memento/ClaudeCode/bin/python
```

✅ TASK COMPLETED: Local tools and modes configured

---

## TASK 4: Establish Safeguards
▶️ TASK STARTING: Setting up safeguards

### Git Safety Measures (CRITICAL)
```
GIT SAFETY:
1. ALWAYS create a new git branch before running aider:
   - Use: `git checkout -b aider/<feature-name>` or `git switch -c aider/<feature-name>`
   - NEVER run aider on main/master branch directly

2. COMMIT current work before running aider:
   - Use: `git add . && git commit -m "WIP: Before aider"`
   - This ensures you have a clean recovery point

3. VERIFY branch status before/after:
   - Before: `git status` to ensure clean working tree
   - After: `git diff` to review all changes made by aider
   - IMPORTANT: Run git commands from the correct project root directory
   - For multi-project repositories, cd to the specific project directory first

4. REPOSITORY CONTEXT AWARENESS:
   - ALWAYS identify the correct repository root for git commands
   - Check `.git` location with: `find . -name ".git" -type d`
   - For projects within larger repositories, change to the project directory:
     ```bash
     cd /path/to/specific/project/
     git status  # Now shows only relevant changes
     git diff    # Only shows diffs in the current project
     ```

5. RECOVERY options if changes are problematic:
   - Selective reversal: `git checkout -- <file>` to revert specific files
   - Complete reversal: `git reset --hard HEAD` to revert all changes
   - Return to pre-aider: `git checkout <original-branch>`

6. ISOLATE aider sessions:
   - Use different branches for different aider tasks
   - Commit meaningful increments during aider development
```

### Code Modification Rules
```
SAFEGUARDS:
1. CAREFULLY SELECT which files aider can edit
2. ALWAYS display the git diff using BatchTool immediately after aider completes
3. ALWAYS run tests after aider modifications
4. FOLLOW existing code style
5. ENSURE aider preserves docstrings and comments
6. MINIMIZE your API token usage - delegate to aider
7. AVOID limiting aider's context - give it full files
```

### Communication Guidelines
```
COMMUNICATION:
1. Be concise and direct in responses
2. SHOW command output directly
3. PROVIDE clear summaries of changes
4. AVOID unnecessarily verbose explanations unless requested
```

✅ TASK COMPLETED: Safeguards established

---

## EXAMPLE USAGE

```python
# Effective prompt structure for test generation
prompt = """
Create a comprehensive test file for git_manager.py with the following:
1. Test fixtures that mock subprocess calls and Path operations
2. Test cases for all public methods: is_git_installed, init_repo, etc.
3. Error condition tests for each method
4. Proper patching for subprocess.run and Path operations

Follow testing patterns established in test_path_resolver.py, using
unittest.mock.patch for Git command mocking.
"""

# Effective file parameters
editable_files = ["tests/utils/test_git_manager.py"]
readonly_files = [
    "src/pytest_analyzer/utils/git_manager.py",
    "tests/utils/test_path_resolver.py",
    "tests/conftest.py"
]
```

## LESSONS LEARNED

1. **Providing Rich Context**: The quality of aider's output directly correlates with the quality of context provided in readonly files.

2. **Specific Prompts**: More specific prompts about implementation strategies yield better results than general requests.

3. **Incremental Implementation**: Breaking implementation into logical phases produces cleaner, more comprehensive code.

4. **Mocking Guidance**: Being explicit about mocking strategies significantly improves test implementation quality.

5. **Error Resolution**: When tests fail after implementation, providing the error messages and stack traces in a follow-up prompt allows aider to effectively fix issues.

---

OUTPUT: "Context priming complete. I will now prioritize:
1. Using aider EXTENSIVELY as the primary tool for code tasks
2. Providing aider with comprehensive file context (both editable and readonly files)
3. ALWAYS showing git diff with BatchTool immediately after aider operations
4. Using the /aider_with_diff custom command when appropriate
5. Using local conda environment for Python execution
6. ENSURING git safety with dedicated branches for each aider task
7. Using just-prompt selectively for conceptual questions
8. Conserving my tokens by focusing on orchestration and file selection
9. Following the established aider-centric workflow and safeguards"
