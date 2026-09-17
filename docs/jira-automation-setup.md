# Jira → GitHub → Claude Code automation

When a Jira ticket is **created**, a Jira Automation rule fires a GitHub
`repository_dispatch` event. That triggers the
[`Jira Ticket to PR (Claude)`](../.github/workflows/jira-to-pr.yml) workflow,
which asks **Claude Code** to implement the ticket and opens a **pull request**
with the changes.

```
Jira ticket created
      │  (Automation rule: "Send web request")
      ▼
POST https://api.github.com/repos/Hemakumargokul/RAGBench/dispatches
      │  event_type: jira-ticket-created  +  client_payload {key, summary, description, url}
      ▼
GitHub Actions workflow (.github/workflows/jira-to-pr.yml)
      │
      ├─ build a branch name + prompt from the ticket
      ├─ run Claude Code (anthropics/claude-code-base-action) → edits files
      └─ peter-evans/create-pull-request → opens PR (labels: jira, claude)
```

---

## 1. GitHub setup

### 1a. Add the Anthropic API key as a repo secret

1. Get an API key from <https://console.anthropic.com/> → **API Keys**.
2. In the repo: **Settings → Secrets and variables → Actions → New repository secret**
   - **Name:** `ANTHROPIC_API_KEY`
   - **Value:** your `sk-ant-...` key

### 1b. Allow Actions to open pull requests

**Settings → Actions → General → Workflow permissions**:

- Select **Read and write permissions**
- Check **Allow GitHub Actions to create and approve pull requests**

(The workflow itself only requests `contents: write` and `pull-requests: write`.)

### 1c. Create a token for Jira to call GitHub

Jira needs a token to POST the dispatch. Use **either**:

- **Fine-grained PAT** (recommended): <https://github.com/settings/tokens?type=beta>
  - Repository access: **Only select repositories** → `RAGBench`
  - Repository permissions: **Contents → Read and write**
    (the `repository_dispatch` endpoint requires the Contents write permission)
- **Classic PAT**: <https://github.com/settings/tokens> with the `repo` scope.

Copy the token — you'll paste it into Jira in step 2. Treat it like a password.

---

## 2. Jira Automation rule

In Jira Cloud (`https://hemakumargokul.atlassian.net`):

**Project settings → Automation → Create rule** (or global **Automation** at
`/jira/settings/automation`).

### Trigger

- **Issue created**

*(Optional) Add a condition* to only run for certain issue types, e.g.
**Issue fields condition → Issue Type → is one of → Task, Story**.

### Action: Send web request

- **Web request URL:**
  ```
  https://api.github.com/repos/Hemakumargokul/RAGBench/dispatches
  ```
- **HTTP method:** `POST`
- **Web request body:** `Custom data`
- **Headers:**

  | Key                    | Value                        |
  | ---------------------- | ---------------------------- |
  | `Accept`               | `application/vnd.github+json` |
  | `Authorization`        | `Bearer YOUR_GITHUB_TOKEN`   |
  | `X-GitHub-Api-Version` | `2022-11-28`                 |
  | `Content-Type`         | `application/json`           |

- **Custom data (body):**

  ```json
  {
    "event_type": "jira-ticket-created",
    "client_payload": {
      "issue_key": "{{issue.key}}",
      "summary": "{{issue.summary.jsonEncode}}",
      "description": "{{issue.description.jsonEncode}}",
      "issue_type": "{{issue.issueType.name.jsonEncode}}",
      "issue_url": "https://hemakumargokul.atlassian.net/browse/{{issue.key}}"
    }
  }
  ```

  > `.jsonEncode` escapes quotes, newlines, and special characters so the JSON
  > payload stays valid regardless of ticket content. Do **not** add it to
  > `issue.key` (it's already a safe identifier).

Enable **Allow the rule to check its own web response** if you want the audit
log to show GitHub's response.

Click **Turn it on**.

---

## 3. Test it

### Option A — manual GitHub run (no Jira needed)

Repo → **Actions → Jira Ticket to PR (Claude) → Run workflow**, fill in
`issue_key`, `summary`, and `description`.

### Option B — simulate the Jira webhook with curl

```bash
curl -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer YOUR_GITHUB_TOKEN" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/Hemakumargokul/RAGBench/dispatches \
  -d '{
    "event_type": "jira-ticket-created",
    "client_payload": {
      "issue_key": "KAN-1",
      "summary": "Add a /health endpoint",
      "description": "Add a GET /health route that returns {\"status\":\"ok\"}.",
      "issue_url": "https://hemakumargokul.atlassian.net/browse/KAN-1"
    }
  }'
```

A `204 No Content` means the dispatch was accepted — check the **Actions** tab.

### Option C — end-to-end

Create a real ticket in Jira. Within a minute you should see a workflow run and,
if Claude made changes, a new PR labeled `jira` + `claude`.

---

---

## 4. Claude PR review bot

[`claude-pr-review.yml`](../.github/workflows/claude-pr-review.yml) makes Claude
review pull requests in the GitHub UI (a summary plus inline comments).

It runs:

- **Automatically** on every PR `opened` / `synchronize` / `reopened`, and
- **On demand** when someone comments **`@claude`** on a PR (or a review comment).

It reuses the same `ANTHROPIC_API_KEY` secret — no extra setup needed.

### Important: reviewing auto-generated PRs

GitHub deliberately does **not** trigger further workflows for events raised by
the built-in `GITHUB_TOKEN`. Because `jira-to-pr.yml` opens PRs with
`GITHUB_TOKEN`, those PRs will **not** auto-trigger the review workflow. Two ways
to review them:

1. **Quick:** comment `@claude` on the PR — the review runs on demand.
2. **Fully automatic:** create a PAT secret and have the PR opened with it so the
   `pull_request` event fires:
   - Add repo secret `GH_PAT` (fine-grained: Contents + Pull requests: read/write).
   - In `jira-to-pr.yml`, change the create-PR step to
     `token: ${{ secrets.GH_PAT }}`.

Human-opened PRs are always auto-reviewed.

## Notes & tuning

- **Model / limits:** adjust `claude_args` in the workflow (e.g. add
  `--model claude-sonnet-4-5` or change `--max-turns`).
- **No changes = no PR:** if Claude decides no code change is needed, the run
  logs a warning and skips PR creation (by design).
- **Pin versions:** `@main` is used for the Claude action for convenience; pin to
  a released tag or commit SHA for reproducible builds.
- **Security:** ticket text is passed to the workflow via environment variables
  (never interpolated into shell), and the prompt tells Claude to treat ticket
  text as data. Still, only enable this on repos/tickets you trust, since Claude
  acts on ticket content.
- **Branch naming:** `jira/<KEY>-<slugified-summary>`.
