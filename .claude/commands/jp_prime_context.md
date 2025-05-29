# Prime Working Context
> This command sets up the ideal working context, tool priorities, and configuration for Claude Code to follow, with an emphasis on using just-prompt extensively to leverage external LLMs for most task logic.

## EXECUTION GUIDE
This command takes no arguments. It sets up the foundational principles for Claude Code operation.
Example usage: `/prime_context`

## TOOLS USED:
- **just-prompt**: PRIMARY tool for delegating reasoning and processing to external LLMs
- **aider**: For AI-assisted code implementation when needed
- **code-reducer**: For optimizing code before analysis with LLMs

---

## TASK 1: Set Tool Priorities and Token Conservation
▶️ TASK STARTING: Establishing tool priority hierarchy with focus on token efficiency

### Define Primary Tools and Token Usage Strategy
```
PRIORITIES:
1. CONSERVE YOUR API TOKENS AGGRESSIVELY:
   - Your tokens are EXTREMELY VALUABLE and must be used with extreme parsimony
   - Focus your tokens on orchestration, context-building, and user interaction
   - DELEGATE all complex reasoning, code generation, and analysis to external LLMs
   - It is ALWAYS better to run just-prompt multiple times to refine an answer than to use your own API
   - When a test is failing, request a review from external LLMs rather than analyzing yourself

2. USE just-prompt PROFUSELY for ALL cognitive work:
   - For simple tasks (docstrings, typing):
     * gemini:gemini-1.5-pro
     * claude-3-5-haiku
   - For complex tasks (architecture, algorithms, debugging):
     * gemini:gemini-2.5-pro-preview-03-25
     * openai:gpt-4o
   - Create rich contexts for just-prompt using your file exploration tools (LS, View, GlobTool)
   - Avoid hallucination by comparing responses between multiple LLMs
   - ESCALATE to smarter LLMs rather than attempting complex reasoning yourself

3. ALWAYS use the local conda environment for Python execution:
   - /home/memento/ClaudeCode/bin/conda run -n ClaudeCode python
   - /home/memento/ClaudeCode/bin/conda run -n ClaudeCode pytest

   For projects using package managers, run them through conda:
   - /home/memento/ClaudeCode/bin/conda run -n ClaudeCode pixi run pytest
   - /home/memento/ClaudeCode/bin/conda run -n ClaudeCode hatch run pytest
   - /home/memento/ClaudeCode/bin/conda run -n ClaudeCode uv run pytest

4. PREFER concise tools during exploration:
   - Use /jp_with_mode instead of direct just-prompt calls
   - Use /aider_code command for implementation tasks
   - Use /reduce_python before analyzing complex code

5. DELEGATE code implementation to aider:
   - For implementation tasks
   - For refactoring tasks
   - For bug fixing tasks

6. NEVER use your own API for code generation or complex analysis
```

### LLM Delegation Strategy
```
LLM DELEGATION STRATEGY:
1. Your role is primarily ORCHESTRATION and CONTEXT-BUILDING:
   - You should identify which files to examine
   - Construct appropriate contexts for LLMs by combining code snippets
   - Choose the appropriate LLM based on task complexity
   - Break complex problems into sub-questions for LLMs
   - Refine and iterate on LLM responses when needed

2. External LLMs handle ALL intensive reasoning:
   - Code analysis and understanding
   - Debugging and error diagnosis
   - Architecture and design decisions
   - Algorithm development
   - Code generation and optimization

3. When responses are not satisfactory:
   - Try again with more context
   - Try a more capable LLM model
   - Break down the problem further
   - Do NOT attempt to solve complex issues with your own reasoning
```

### Environment Paths
```
ENVIRONMENT:
- Python: /home/memento/ClaudeCode/bin/conda run -n ClaudeCode python
- Pytest: /home/memento/ClaudeCode/bin/conda run -n ClaudeCode pytest
- Conda: /home/memento/ClaudeCode/bin/conda
- Package managers (always run through conda):
  * Pixi: /home/memento/ClaudeCode/bin/conda run -n ClaudeCode pixi run <command>
  * Hatch: /home/memento/ClaudeCode/bin/conda run -n ClaudeCode hatch run <command>
  * UV: /home/memento/ClaudeCode/bin/conda run -n ClaudeCode uv run <command>
```

✅ TASK COMPLETED: Tool priorities established

---

## TASK 2: Define Working Methodology
▶️ TASK STARTING: Setting up working methodology with extensive LLM delegation

### Implementation Workflow
```
WORKFLOW:
1. ANALYZE task requirements carefully
2. SEARCH for relevant code with dispatch_agent (NOT your own API)
3. BUILD CONTEXTS for just-prompt:
   - Gather code from relevant files
   - Include necessary context from related files
   - Structure information to maximize LLM comprehension

4. ITERATIVELY QUERY LLMs:
   - Start with simpler/cheaper LLMs for basic tasks
   - Escalate to more powerful LLMs for complex problems
   - Refine queries based on initial responses
   - Compare results from multiple LLMs when appropriate

5. IMPLEMENT with external tools:
   - Use aider for code implementation
   - Use just-prompt for code reviews

6. VALIDATE:
   - Run tests after changes
   - If tests fail, use just-prompt to analyze failures

7. COMMUNICATE clearly what was done
```

### Decision Framework
```
DECISION TREE:
- IF task involves browsing/understanding code:
  → Use dispatch_agent to gather information
  → THEN use just-prompt to analyze the collected code

- IF task involves implementing new features/code:
  → Use aider_code command for implementation
  → Use just-prompt for code review

- IF task involves improving test coverage:
  → Use jp_add_coverage command

- IF task involves reducing code size:
  → Use reduce_python command

- IF task requires technical advice/information:
  → Use just-prompt with appropriate model(s)

- IF tests are failing:
  → Use just-prompt to analyze the failure
  → NEVER attempt complex debugging yourself
```

✅ TASK COMPLETED: Working methodology defined

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

  if [ -f "$LOCAL_CMD_DIR/jp_add_coverage.md" ]; then
    OUTPUT: "Found local jp_add_coverage command"
  else
    OUTPUT: "Note: Local jp_add_coverage command not found. Using global definition."
  fi
else
  OUTPUT: "No local .claude/commands directory found. Using global command definitions."
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

### Code Modification Rules
```
SAFEGUARDS:
1. NEVER modify code directly without user confirmation
2. ALWAYS run tests after modifications
3. FOLLOW existing code style
4. PRESERVE docstrings and comments
5. MINIMIZE your API token usage - rely heavily on external tools
6. When in doubt, USE just-prompt instead of your own reasoning
```

### Communication Guidelines
```
COMMUNICATION:
1. Be concise and direct in responses
2. SHOW command output directly
3. REQUEST confirmation before making changes
4. PROVIDE clear summaries of changes
5. AVOID unnecessarily verbose explanations unless requested
```

✅ TASK COMPLETED: Safeguards established

---

OUTPUT: "Context priming complete. I will now prioritize:
1. Using just-prompt EXTENSIVELY to delegate reasoning and code tasks to external LLMs
2. Conserving my tokens by focusing on orchestration and context-building
3. Using local conda environment for Python execution
4. Delegating implementation to aider when appropriate
5. Using code-reducer before analyzing complex code
6. Following the established workflow and safeguards"
