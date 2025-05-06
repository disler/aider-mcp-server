  Token Limit Handling in Aider MCP Server: Findings and Recommendations

  Current State

  The aider MCP server currently has several mechanisms related to handling large payloads and rate limits:

  1. Rate Limit Fallback System:
    - Detects rate limit errors from providers like OpenAI, Anthropic, and Gemini
    - Implements backoff and retry logic
    - Falls back to smaller models when needed
  2. Diff Cache Mechanism:
    - Implements a caching system for diffs to reduce repeated content
    - Uses memory limits and LRU eviction
    - Offers content reduction by storing only changes
  3. Error Handling:
    - Catches rate limit errors from various providers
    - Reports errors back to users

  However, the current implementation lacks direct token management features:

  1. No token counting or estimation before sending to models
  2. No proactive content truncation or filtering
  3. No intelligent code-aware content reduction
  4. No explicit handling of token limit errors (as opposed to rate limit errors)

  Recommendations

  Based on our analysis and the aider.chat documentation, here are our recommendations for improving token limit handling:

  1. Implement Token Counting and Estimation

  def estimate_tokens(content: str, model: str) -> int:
      """Estimate token count for content based on model."""
      # Simple heuristic: ~4 characters per token for English text/code
      # Different models may need different estimators
      if model.startswith("gpt-"):
          # OpenAI models
          return len(content) // 4
      elif model.startswith("claude-"):
          # Anthropic models
          return len(content) // 4
      else:
          # Default estimation
          return len(content) // 4

  2. Add Token Budget Management

  class TokenBudgetManager:
      """Manage token budget for a single request."""

      def __init__(self, model: str, max_input_tokens: int, max_output_tokens: int):
          self.model = model
          self.max_input_tokens = max_input_tokens
          self.max_output_tokens = max_output_tokens
          self.used_tokens = 0

      def can_add_content(self, content: str) -> bool:
          """Check if content fits within remaining token budget."""
          estimated_tokens = estimate_tokens(content, self.model)
          return self.used_tokens + estimated_tokens <= self.max_input_tokens

      def add_content(self, content: str) -> int:
          """Add content to token usage and return estimated tokens."""
          estimated_tokens = estimate_tokens(content, self.model)
          self.used_tokens += estimated_tokens
          return estimated_tokens

      def get_remaining_budget(self) -> int:
          """Get remaining token budget."""
          return self.max_input_tokens - self.used_tokens

  3. Implement Smart File Selection

  def select_files_by_relevance(
      files: List[str], 
      prompt: str, 
      token_budget: int,
      model: str
  ) -> List[str]:
      """Select most relevant files that fit within token budget."""
      # Calculate relevance scores (could use TF-IDF, embeddings, etc.)
      relevance_scores = [(f, calculate_relevance(f, prompt)) for f in files]
      # Sort by relevance
      sorted_files = sorted(relevance_scores, key=lambda x: x[1], reverse=True)

      selected_files = []
      used_tokens = 0

      for file, _ in sorted_files:
          file_content = read_file(file)
          file_tokens = estimate_tokens(file_content, model)

          if used_tokens + file_tokens <= token_budget:
              selected_files.append(file)
              used_tokens += file_tokens
          else:
              # Consider truncation here
              pass

      return selected_files

  4. Implement Content Reduction for Large Files

  def reduce_file_content(
      file_path: str, 
      max_tokens: int,
      model: str,
      preserve_imports: bool = True
  ) -> str:
      """Reduce file content to fit within token budget."""
      content = read_file(file_path)
      estimated_tokens = estimate_tokens(content, model)

      if estimated_tokens <= max_tokens:
          return content

      # Intelligent reduction strategies:
      # 1. For Python, preserve imports, class/function signatures
      if file_path.endswith(".py") and preserve_imports:
          return reduce_python_file(content, max_tokens, model)

      # 2. For other files, preserve beginning and end, truncate middle
      middle_start = len(content) // 3
      middle_end = 2 * len(content) // 3
      truncate_length = len(content) - (3 * max_tokens // 4)

      if truncate_length <= 0:
          return content

      return content[:middle_start] + f"\n# ... truncated {truncate_length} characters ...\n" + content[middle_end:]

  5. Add Token-Specific Error Handling

  def handle_token_limit_error(error: Exception, model: str, content_size: int) -> Dict[str, Any]:
      """Handle token limit errors with specific guidance."""
      response = {
          "success": False,
          "error": str(error),
          "suggestions": []
      }

      # Check if error is token-related
      error_message = str(error).lower()
      token_patterns = ["token limit", "context length", "maximum context", "too many tokens"]

      if any(pattern in error_message for pattern in token_patterns):
          # Add specific suggestions
          response["error_type"] = "token_limit"
          response["suggestions"] = [
              "Reduce the number of files in the request",
              "Break the task into smaller changes",
              "Try using a model with larger context window",
              "Use /drop command to remove unnecessary files",
              "Use /clear command to clear chat history"
          ]

          # Add specific model suggestions
          if model.startswith("gpt-3.5"):
              response["suggestions"].append("Try upgrading to gpt-4-turbo with 128k context")
          elif model.startswith("gemini-1.0"):
              response["suggestions"].append("Try upgrading to gemini-1.5-pro with larger context")

      return response

  6. Updates to aider_ai_code.py Implementation

  The current code_with_aider function in aider_ai_code.py could be enhanced with token management:

  async def code_with_aider(
      ai_coding_prompt: str,
      relative_editable_files: List[str],
      relative_readonly_files: Optional[List[str]] = None,
      model: str = "gemini/gemini-2.5-flash-preview-04-17",
      working_dir: Optional[str] = None,
      use_diff_cache: bool = True,
      clear_cached_for_unchanged: bool = True,
      max_input_tokens: Optional[int] = None,  # Added parameter
      auto_reduce_content: bool = True,        # Added parameter
  ) -> str:
      """Run Aider with token management."""
      # ... existing code ...

      # Initialize token budget
      if max_input_tokens is None:
          # Set based on model
          if "gpt-4" in model:
              max_input_tokens = 120000  # example for GPT-4 Turbo
          elif "claude-3" in model:
              max_input_tokens = 120000  # example for Claude 3
          else:
              max_input_tokens = 15000   # conservative default

      # Create token budget manager
      token_manager = TokenBudgetManager(model, max_input_tokens, max_output_tokens=4000)

      # Add prompt to token budget
      token_manager.add_content(ai_coding_prompt)

      # Process files with token budget in mind
      if auto_reduce_content:
          # Select most relevant files that fit in budget
          processed_editable_files = []
          processed_readonly_files = []

          for file in relative_editable_files:
              full_path = os.path.join(working_dir, file) if working_dir else file
              content = read_file(full_path)

              if token_manager.can_add_content(content):
                  token_manager.add_content(content)
                  processed_editable_files.append(file)
              else:
                  # Try to reduce content
                  reduced_content = reduce_file_content(
                      full_path,
                      token_manager.get_remaining_budget() // 2,  # Use half remaining budget
                      model
                  )
                  if token_manager.can_add_content(reduced_content):
                      # Use reduced content through temporary file
                      token_manager.add_content(reduced_content)
                      processed_editable_files.append(file)
                      logger.info(f"Reduced content for {file} to fit token budget")
                  else:
                      logger.warning(f"Skipping {file} as it doesn't fit in token budget")

          # Similar processing for readonly files
          # ...

          # Update the file lists
          relative_editable_files = processed_editable_files
          if relative_readonly_files:
              relative_readonly_files = processed_readonly_files

      # Continue with existing implementation
      # ...

  7. Implementation Roadmap

  1. First Phase: Add token estimation and monitoring
    - Implement basic token counters without changing behavior
    - Add logging for token usage
    - Track token-related errors
  2. Second Phase: Add token budget management
    - Implement TokenBudgetManager
    - Add configuration options for token limits
    - Add content reduction capabilities
  3. Third Phase: Add intelligent file selection
    - Implement relevance-based file selection
    - Add enhanced error messages with helpful suggestions
    - Implement progressive degradation strategies
  4. Fourth Phase: Integration with UI/UX
    - Add user-facing warnings about token usage
    - Provide configuration options in CLI
    - Add documentation on token management strategies

  Conclusion

  The current implementation has a good foundation with the diff cache and rate limit handling, but it lacks direct token management.
  By implementing the proposed changes, the aider MCP server would gain significantly improved ability to handle large codebases
  without hitting token limits, providing a better user experience and more efficient operation.
