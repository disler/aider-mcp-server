Implementing Architect Mode in Aider MCP Server

  After analyzing the aider SDK implementation and our current MCP server code, here's a comprehensive plan to implement architect mode
   in the Aider MCP Server.

  1. Understanding of Architect Mode

  Architect mode in aider is a two-phase approach to code generation:
  1. A primary "architect" model designs the implementation at a high level
  2. A secondary "editor" model implements the actual code based on the architect's plan

  This approach is particularly useful for complex tasks that benefit from careful planning before implementation.

  2. Current Status in MCP Server

  Our MCP server has partial architect mode support:
  - References in documentation and tests for complex tasks
  - The aider SDK dependency we use has architect mode support
  - Our implementation doesn't have explicit support for architect mode

  3. Implementation Plan

  Step 1: Update the code_with_aider Function

  Modify the function signature to accept architect mode parameters:

  def code_with_aider(
      ai_coding_prompt: str,
      relative_editable_files: List[str],
      working_dir: str,
      relative_readonly_files: List[str] = [],
      model: str = "gemini/gemini-2.5-pro-exp-03-25",
      architect_mode: bool = False,
      editor_model: Optional[str] = None,
      auto_accept_architect: bool = True,
  ) -> str:
      """
      Run Aider to perform AI coding tasks based on the provided prompt and files.
      
      Added architect mode parameters:
      - architect_mode: Enable two-phase code generation with an architect model
      - editor_model: Optional secondary model for code implementation
      - auto_accept_architect: Automatically accept architect suggestions without confirmation
      """

  Step 2: Update Model Configuration Logic

  Add logic to handle architect and editor models:

  # If architect mode is enabled, configure the model differently
  if architect_mode:
      if editor_model:
          # Configure main model as architect, with a separate editor model
          ai_model = Model(model, editor_model=editor_model)
      else:
          # Use the same model for both architect and editor roles
          ai_model = Model(model, editor_model=model)

      # Set edit_format to architect for the Coder instance
      edit_format = "architect"
  else:
      # Standard (non-architect) configuration
      ai_model = Model(model)
      edit_format = None  # Let aider choose the default

  Step 3: Update Coder Creation Logic

  Modify the Coder creation to support architect mode:

  coder = Coder.create(
      main_model=ai_model,
      edit_format="architect" if architect_mode else None,
      io=InputOutput(yes=True, chat_history_file=chat_history_file),
      fnames=abs_editable_files,
      read_only_fnames=abs_readonly_files,
      auto_commits=False,
      suggest_shell_commands=False,
      detect_urls=False,
      use_git=True,
      auto_accept_architect=auto_accept_architect if architect_mode else True,
  )

  Step 4: Add the Same Updates to async_code_with_aider Function

  Make parallel changes to the async version, ensuring consistent behavior.

  Step 5: Update Default Parameters in Utils

  Update the tool schema defaults:

  # In atoms/utils.py
  DEFAULT_EDITOR_MODEL = "gemini/gemini-2.5-pro-exp-03-25"
  DEFAULT_ARCHITECT_MODEL = "gemini/gemini-2.5-pro-exp-03-25"

  Step 6: Update Tools Schema Definitions

  Update the tools schema to include the new parameters:

  # In server.py or wherever tools are defined
  aider_ai_code_schema = {
      "type": "function",
      "function": {
          "name": "aider_ai_code",
          "description": "Run Aider to perform AI coding tasks based on the provided prompt and files",
          "parameters": {
              "type": "object",
              "properties": {
                  "ai_coding_prompt": {"type": "string", "description": "The prompt for the AI to execute"},
                  "relative_editable_files": {"type": "array", "items": {"type": "string"}, "description": "LIST of relative paths to 
  files that can be edited"},
                  "relative_readonly_files": {"type": "array", "items": {"type": "string"}, "description": "LIST of relative paths to 
  files that can be read but not edited, add files that are not editable but useful for context"},
                  "model": {"type": "string", "description": "The primary AI model Aider should use for generating code, leave blank 
  unless model is specified in the request"},
                  "architect_mode": {"type": "boolean", "description": "Enable two-phase code generation with an architect model 
  planning first, then an editor model implementing"},
                  "editor_model": {"type": "string", "description": "The secondary AI model to use for code implementation when 
  architect_mode is enabled"},
                  "auto_accept_architect": {"type": "boolean", "description": "Automatically accept architect suggestions without 
  confirmation"}
              },
              "required": ["ai_coding_prompt", "relative_editable_files"]
          }
      }
  }

  Step 7: Update Tests

  Add tests for architect mode in test_aider_ai_code.py:

  def test_architect_mode(temp_dir):
      """Test that code_with_aider correctly implements complex tasks using architect mode."""
      test_file = os.path.join(temp_dir, "calculator.py")
      with open(test_file, "w") as f:
          f.write("# This file should implement a calculator class\n")

      prompt = """
      Create a Calculator class with the following features:
      1. Basic operations: add, subtract, multiply, divide methods
      2. Memory functions: memory_store, memory_recall, memory_clear
      3. A history feature that keeps track of operations 
      4. A method to show_history
      5. Error handling for division by zero
      
      All methods should be well-documented with docstrings.
      """

      # Run code_with_aider with architect mode
      result = code_with_aider(
          ai_coding_prompt=prompt,
          relative_editable_files=[test_file],
          model="gemini/gemini-2.5-pro-exp-03-25",  # Main architect model
          editor_model="gemini/gemini-2.5-pro-exp-03-25",  # Editor model
          architect_mode=True,
          auto_accept_architect=True,
          working_dir=temp_dir
      )

      # Parse and verify the result...

  Step 8: Update Documentation

  Add information about architect mode to comments and documentation.

  4. Benefits of Architect Mode

  1. Better Complex Task Handling: Produces more comprehensive and well-organized code for complex tasks
  2. More Thorough Planning: The two-phase approach ensures proper design before implementation
  3. Separation of Concerns: Allows different models to specialize in design and implementation
  4. Improved Documentation: Often results in better docstrings and comments

  5. Implementation Timeline

  1. Phase 1 (Day 1): Update function signatures and parameter handling
  2. Phase 2 (Day 1-2): Update model configuration and Coder creation logic
  3. Phase 3 (Day 2): Add tests for architect mode
  4. Phase 4 (Day 2-3): Conduct comprehensive testing
  5. Phase 5 (Day 3): Update documentation

  6. Technical Challenges and Considerations

  1. Model Compatibility: Not all models work well in architect mode
  2. Token Usage: Architect mode uses more tokens, amplifying token limit issues
  3. Error Handling: Need to handle errors in both architectural and implementation phases
  4. Performance: Two-phase process is slower but produces better results for complex tasks

  This plan provides a complete roadmap for implementing architect mode in the Aider MCP Server, enabling more sophisticated code
  generation capabilities while maintaining backward compatibility.
