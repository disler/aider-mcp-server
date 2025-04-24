# Aider is a programmable AI coding assistant

Here's how to use it in python to build tools that allow us to offload ai coding tasks to aider.

## Code Examples

```

class AICodeParams(BaseModel):
    architect: bool = True
    prompt: str
    model: str
    editor_model: Optional[str] = None
    editable_context: List[str]
    readonly_context: List[str] = []
    settings: Optional[dict]
    use_git: bool = True


def build_ai_coding_assistant(params: AICodeParams) -> Coder:
    """Create and configure a Coder instance based on provided parameters"""
    settings = params.settings or {}
    auto_commits = settings.get("auto_commits", False)
    suggest_shell_commands = settings.get("suggest_shell_commands", False)
    detect_urls = settings.get("detect_urls", False)

    # Extract budget_tokens setting once for both models
    budget_tokens = settings.get("budget_tokens")

    if params.architect:
        model = Model(model=params.model, editor_model=params.editor_model)
        extra_params = {}

        # Add reasoning_effort if available
        if settings.get("reasoning_effort"):
            extra_params["reasoning_effort"] = settings["reasoning_effort"]

        # Add thinking budget if specified
        if budget_tokens is not None:
            extra_params = add_thinking_budget_to_params(extra_params, budget_tokens)

        model.extra_params = extra_params
        return Coder.create(
            main_model=model,
            edit_format="architect",
            io=InputOutput(yes=True),
            fnames=params.editable_context,
            read_only_fnames=params.readonly_context,
            auto_commits=auto_commits,
            suggest_shell_commands=suggest_shell_commands,
            detect_urls=detect_urls,
            use_git=params.use_git,
        )
    else:
        model = Model(params.model)
        extra_params = {}

        # Add reasoning_effort if available
        if settings.get("reasoning_effort"):
            extra_params["reasoning_effort"] = settings["reasoning_effort"]

        # Add thinking budget if specified (consistent for both modes)
        if budget_tokens is not None:
            extra_params = add_thinking_budget_to_params(extra_params, budget_tokens)

        model.extra_params = extra_params
        return Coder.create(
            main_model=model,
            io=InputOutput(yes=True),
            fnames=params.editable_context,
            read_only_fnames=params.readonly_context,
            auto_commits=auto_commits,
            suggest_shell_commands=suggest_shell_commands,
            detect_urls=detect_urls,
            use_git=params.use_git,
        )


def ai_code(coder: Coder, params: AICodeParams):
    """Execute AI coding using provided coder instance and parameters"""
    # Execute the AI coding with the provided prompt
    coder.run(params.prompt)


```
