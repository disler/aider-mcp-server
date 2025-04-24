# Ultra Diff Review
> Execute each task in the order given to conduct a thorough code review.

## Task 1: Create diff.txt

Create a new file called diff.md.

At the top of the file, add the following markdown:

```md
# Code Review
- Review the diff, report on issues, bugs, and improvements.
- End with a concise markdown table of any issues found, their solutions, and a risk assessment for each issue if applicable.
- Use emojis to convey the severity of each issue.

## Diff

```

## Task 2: git diff and append

Then run git diff and append the output to the file.

## Task 3: just-prompt multi-llm tool call

Then use that file as the input to this just-prompt tool call.

prompts_from_file_to_file(
    from_file = diff.md,
    models = "openai:o3-mini, anthropic:claude-3-7-sonnet-20250219:4k, gemini:gemini-2.0-flash-thinking-exp"
    output_dir = ultra_diff_review/
)

## Task 4: Read the output files and synthesize

Then read the output files and think hard to synthesize the results into a new single file called `ultra_diff_review/fusion_ultra_diff_review.md` following the original instructions plus any additional instructions or callouts you think are needed to create the best possible review.

## Task 5: Present the results

Then let me know which issues you think are worth resolving and we'll proceed from there.
