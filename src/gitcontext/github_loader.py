import os
import re
from pathlib import Path
from urllib.parse import quote

import requests

from src.gitcontext.ingestion import (
    create_repository_file,
    should_ignore,
)


GITHUB_API = "https://api.github.com"
MAX_BRANCHES = 100
MAX_CONTEXT_RECORDS = 5
MAX_CONTEXT_COMMENTS = 10
MAX_COMMENT_LENGTH = 2000


class GitHubLoader:

    def __init__(self):

        self.token = os.getenv("GITHUB_TOKEN")

        self.session = requests.Session()

        self.session.headers.update({
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        })

        if self.token:
            self.session.headers["Authorization"] = (
                f"Bearer {self.token}"
            )

    def parse_repository_url(self, url):

        match = re.match(
            r"^https?://github\.com/([^/]+)/([^/]+)/?$",
            url.strip()
        )

        if not match:
            raise ValueError(
                "Enter a valid GitHub repository URL."
            )

        owner, repo = match.groups()

        return owner, repo.removesuffix(".git")

    def list_branches(self, repository_url):
        owner, repo = self.parse_repository_url(repository_url)
        repository_response = self.session.get(
            f"{GITHUB_API}/repos/{owner}/{repo}",
            timeout=30,
        )
        repository_response.raise_for_status()
        if repository_response.json().get("private"):
            raise ValueError("Only public GitHub repositories are supported.")
        response = self.session.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/branches",
            params={"per_page": MAX_BRANCHES},
            timeout=30,
        )
        response.raise_for_status()
        return [branch["name"] for branch in response.json()]

    @staticmethod
    def _context_file(path, content, metadata):
        return create_repository_file(
            relative_path=path,
            raw_content=content.encode("utf-8"),
            last_modified=metadata.get("last_modified", ""),
            metadata=metadata,
        )

    @staticmethod
    def _comment_text(comment):
        return (comment.get("body") or "")[:MAX_COMMENT_LENGTH]

    def _load_repository_context(self, owner, repo, branch):
        files = []
        commits_response = self.session.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/commits",
            params={"sha": branch, "per_page": MAX_CONTEXT_RECORDS},
            timeout=30,
        )
        commits_response.raise_for_status()
        for commit in commits_response.json():
            details = commit["commit"]
            sha = commit["sha"]
            author = details.get("author") or {}
            body = (
                f"# Commit {sha[:12]}\n\n"
                f"Author: {author.get('name', 'Unknown')}\n"
                f"Date: {author.get('date', '')}\n\n"
                f"{details.get('message', '')}\n"
            )
            result = self._context_file(
                f"__gitcontext__/commits/{sha[:12]}.md",
                body,
                {
                    "source_type": "commit",
                    "commit_sha": sha,
                    "author": author.get("name", ""),
                    "last_modified": author.get("date", ""),
                },
            )
            if result:
                files.append(result)

        pulls_response = self.session.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/pulls",
            params={
                "state": "all",
                "sort": "updated",
                "direction": "desc",
                "per_page": MAX_CONTEXT_RECORDS,
            },
            timeout=30,
        )
        pulls_response.raise_for_status()
        for pull in pulls_response.json():
            user = pull.get("user") or {}
            body = (
                f"# Pull request #{pull['number']}: {pull.get('title', '')}\n\n"
                f"Author: {user.get('login', 'Unknown')}\n"
                f"State: {pull.get('state', '')}\n"
                f"Branch: {pull.get('head', {}).get('ref', '')} -> "
                f"{pull.get('base', {}).get('ref', '')}\n\n"
                f"{pull.get('body') or ''}\n"
            )
            comments_response = self.session.get(
                f"{GITHUB_API}/repos/{owner}/{repo}/issues/{pull['number']}/comments",
                params={"per_page": MAX_CONTEXT_COMMENTS},
                timeout=30,
            )
            comments_response.raise_for_status()
            for comment in comments_response.json():
                comment_user = comment.get("user") or {}
                body += (
                    f"\n\n## Discussion by {comment_user.get('login', 'Unknown')}\n"
                    f"{self._comment_text(comment)}"
                )
            review_response = self.session.get(
                f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pull['number']}/comments",
                params={"per_page": MAX_CONTEXT_COMMENTS},
                timeout=30,
            )
            review_response.raise_for_status()
            for comment in review_response.json():
                comment_user = comment.get("user") or {}
                body += (
                    f"\n\n## Review comment by {comment_user.get('login', 'Unknown')}"
                    f" on {comment.get('path', 'unknown file')}"
                    f" line {comment.get('line') or comment.get('original_line') or '?'}\n"
                    f"{self._comment_text(comment)}"
                )
            result = self._context_file(
                f"__gitcontext__/pull_requests/pr-{pull['number']}.md",
                body,
                {
                    "source_type": "pull_request",
                    "pull_request_number": pull["number"],
                    "author": user.get("login", ""),
                    "last_modified": pull.get("updated_at", ""),
                },
            )
            if result:
                files.append(result)

        issues_response = self.session.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/issues",
            params={
                "state": "all",
                "sort": "updated",
                "direction": "desc",
                "per_page": MAX_CONTEXT_RECORDS,
            },
            timeout=30,
        )
        issues_response.raise_for_status()
        incident_terms = {"incident", "outage", "production", "bug"}
        for issue in issues_response.json():
            if "pull_request" in issue:
                continue
            labels = [
                label.get("name", "")
                for label in issue.get("labels", [])
            ]
            searchable = " ".join(
                [issue.get("title", ""), *labels]
            ).lower()
            if not any(term in searchable for term in incident_terms):
                continue
            user = issue.get("user") or {}
            body = (
                f"# Issue/incident #{issue['number']}: {issue.get('title', '')}\n\n"
                f"Author: {user.get('login', 'Unknown')}\n"
                f"Labels: {', '.join(labels)}\n"
                f"State: {issue.get('state', '')}\n\n"
                f"{issue.get('body') or ''}\n"
            )
            comments_response = self.session.get(
                f"{GITHUB_API}/repos/{owner}/{repo}/issues/{issue['number']}/comments",
                params={"per_page": MAX_CONTEXT_COMMENTS},
                timeout=30,
            )
            comments_response.raise_for_status()
            for comment in comments_response.json():
                comment_user = comment.get("user") or {}
                body += (
                    f"\n\n## Comment by {comment_user.get('login', 'Unknown')}\n"
                    f"{self._comment_text(comment)}"
                )
            result = self._context_file(
                f"__gitcontext__/incidents/issue-{issue['number']}.md",
                body,
                {
                    "source_type": "incident",
                    "issue_number": issue["number"],
                    "author": user.get("login", ""),
                    "last_modified": issue.get("updated_at", ""),
                },
            )
            if result:
                files.append(result)
        return files

    def load_repository(self, repository_url, branch=None, include_context=True):

        owner, repo = self.parse_repository_url(
            repository_url
        )

        repo_api = (
            f"{GITHUB_API}/repos/{owner}/{repo}"
        )

        response = self.session.get(repo_api, timeout=30)
        response.raise_for_status()

        repo_info = response.json()
        if repo_info.get("private"):
            raise ValueError("Only public GitHub repositories are supported.")

        branch = branch or repo_info["default_branch"]

        branch_url = (
            f"{GITHUB_API}/repos/{owner}/{repo}"
            f"/branches/{quote(branch, safe='')}"
        )

        branch_response = self.session.get(
            branch_url,
            timeout=30
        )

        branch_response.raise_for_status()

        branch_commit = branch_response.json()["commit"]
        commit_sha = branch_commit["sha"]
        tree_sha = branch_commit["commit"]["tree"]["sha"]

        tree_url = (
            f"{GITHUB_API}/repos/{owner}/{repo}"
            f"/git/trees/{tree_sha}?recursive=1"
        )

        tree_response = self.session.get(
            tree_url,
            timeout=30
        )

        tree_response.raise_for_status()

        tree_data = tree_response.json()

        if tree_data.get("truncated"):
            raise RuntimeError(
                "GitHub returned an incomplete repository tree. "
                "Refusing to index an incomplete snapshot."
            )

        files = []

        for item in tree_data.get("tree", []):

            if item["type"] != "blob":
                continue

            relative_path = item["path"]

            if should_ignore(Path(relative_path)):
                continue

            # Check size before downloading.
            if item.get("size", 0) > 1024 * 1024:
                continue

            raw_url = (
                f"https://raw.githubusercontent.com/"
                f"{owner}/{repo}/"
                f"{quote(branch, safe='/')}/"
                f"{quote(relative_path, safe='/')}"
            )

            raw_response = self.session.get(
                raw_url,
                timeout=30
            )

            raw_response.raise_for_status()

            repository_file = create_repository_file(
                relative_path=relative_path,
                raw_content=raw_response.content,
                last_modified="",
                metadata={
                    "source_type": "source",
                    "branch": branch,
                    "commit_sha": commit_sha,
                },
            )

            if repository_file:
                files.append(repository_file)

        if include_context:
            files.extend(self._load_repository_context(owner, repo, branch))

        repository_id = f"github:{owner}/{repo}@{branch}"
        self.last_branch = branch

        print(
            f"[GITHUB] Loaded {len(files)} supported files "
            f"from {owner}/{repo}"
        )

        return repository_id, files