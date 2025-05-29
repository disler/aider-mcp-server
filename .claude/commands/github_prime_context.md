# GitHub Context Priming
> This command sets up the GitHub workflow context so Claude knows exactly which repository and PR to work with.

## EXECUTION GUIDE
This command requires 1 argument:
1. REPO: Repository in format "owner/repo" (e.g., "MementoRC/aider-mcp-server")

Optional argument:
2. PR_NUMBER: Specific PR number (if not provided, will auto-detect from current branch)

Example usage:
- `/github_prime_context MementoRC/aider-mcp-server` - Set repo, auto-detect PR
- `/github_prime_context MementoRC/aider-mcp-server 30` - Set repo and specific PR #30

## TOOLS USED:
- **Bash**: For git and GitHub CLI commands

---

## GITHUB WORKFLOW CONTEXT SETUP

### Task: Set GitHub Workflow Parameters

▶️ **TASK STARTING**: Establishing GitHub workflow context

```
GITHUB WORKFLOW PARAMETERS:
1. TARGET REPOSITORY: [REPO argument from user]
2. BASE BRANCH: development (always)
3. CURRENT BRANCH: [detected from git branch --show-current]
4. PR DETECTION: Auto-detect from current branch OR use provided PR number
5. WORKFLOW PATTERN: current_branch → development branch PR

STRICT RULES:
- ALWAYS use --repo flag with provided repository
- ALWAYS PR against development branch  
- NEVER create PRs to other repositories
- NEVER PR against main/master
- ALWAYS detect current worktree branch
- ALWAYS use 120s timeout for CI monitoring
```

### Environment Setup Commands

```bash
# Set workflow constants
GITHUB_REPO="[REPO]"
CURRENT_BRANCH=$(git branch --show-current)
BASE_BRANCH="development"

# Validate repository context
REPO_URL=$(git remote get-url origin)
if [[ "$REPO_URL" != *"$GITHUB_REPO"* ]]; then
  echo "❌ ERROR: Not in $GITHUB_REPO repository"
  exit 1
fi

# Auto-detect or use provided PR
if [ -n "$PR_NUMBER" ]; then
  WORKING_PR="$PR_NUMBER"
  echo "✅ Using specified PR #$WORKING_PR"
else
  WORKING_PR=$(gh pr list --repo $GITHUB_REPO --head $CURRENT_BRANCH --json number --jq '.[0].number // empty')
  if [ -z "$WORKING_PR" ]; then
    echo "📝 No existing PR found for branch: $CURRENT_BRANCH"
    echo "    Use 'create' action to create PR: $CURRENT_BRANCH → $BASE_BRANCH"
  else
    echo "✅ Auto-detected PR #$WORKING_PR for branch: $CURRENT_BRANCH"
  fi
fi
```

### Available Actions After Priming

Once primed, Claude will know to use these exact parameters for:
- **Status Check**: `gh pr checks $WORKING_PR --repo $GITHUB_REPO`
- **Monitor CI**: `timeout 120s gh pr checks $WORKING_PR --repo $GITHUB_REPO --watch`
- **Update PR**: `gh pr edit $WORKING_PR --repo $GITHUB_REPO --body "..."`
- **Create PR**: `gh pr create --repo $GITHUB_REPO --base $BASE_BRANCH --head $CURRENT_BRANCH`

### Context Memory

After running this command, Claude will remember:
- Repository: $GITHUB_REPO
- Current Branch: $CURRENT_BRANCH  
- Working PR: $WORKING_PR
- Base Branch: $BASE_BRANCH (always development)

✅ **TASK COMPLETED**: GitHub workflow context established

---

OUTPUT: "GitHub workflow context primed. I will now use:
- Repository: [REPO]
- Current Branch: [detected branch]
- Working PR: [auto-detected or provided PR]
- Base Branch: development
- All GitHub operations will use --repo flag with correct repository
- CI monitoring will use 120s timeout
- PRs will always target development branch"