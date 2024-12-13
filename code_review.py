import os
import sys
import httpx
from github import Github
from git import Repo
import json
from typing import List, Dict, Any
from dataclasses import dataclass
import logging
import time
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ReviewComment:
    file_path: str
    line_number: int
    message: str
    severity: str
    suggestion: str = None

class PRReviewer:
    def __init__(self):
        self.flow_client_id = os.environ.get('FLOW_CLIENT_ID')
        self.flow_client_secret = os.environ.get('FLOW_CLIENT_SECRET')
        self.flow_api_url = "https://api.flowai.com/v1/chat/completions"
        self.github_client = Github(os.environ.get('GITHUB_TOKEN'))
        self.repo_name = os.environ.get('REPO_NAME')
        self.pr_number = int(os.environ.get('PR_NUMBER'))
        self.max_chunk_size = 15000
        self.diff_map = {}
        self.http_client = httpx.Client(verify=False)

    def __del__(self):
        self.http_client.close()

    def get_pr_diff(self) -> str:
        """Get the current PR diff"""
        try:
            repo = Repo('.')
            repo.git.fetch('origin', 'main:main')

            base_sha = repo.rev_parse('main')
            head_sha = repo.head.commit

            diff = repo.git.diff(f"{base_sha}...{head_sha}")
            logger.info(f"Retrieved diff with {len(diff.splitlines())} lines")
            return diff

        except Exception as e:
            logger.error(f"Error getting diff: {e}")
            raise

    def parse_diff_and_create_map(self, diff: str) -> Dict[str, str]:
        """Parse diff and create line mapping"""
        files_diff = {}
        current_file = None
        current_content = []
        line_map = {}
        current_line = 0

        for line in diff.splitlines():
            if line.startswith('diff --git'):
                if current_file:
                    files_diff[current_file] = '\n'.join(current_content)
                match = re.search(r'b/(.+)$', line)
                if match:
                    current_file = match.group(1)
                    current_content = [line]
                    line_map[current_file] = {}
                    current_line = 0
            elif line.startswith('@@'):
                match = re.search(r'@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@', line)
                if match:
                    current_line = int(match.group(1)) - 1
            elif current_file:
                current_content.append(line)
                if line.startswith('+'):
                    current_line += 1
                    if not line.startswith('+++'):
                        line_map[current_file][current_line] = line
                elif not line.startswith('-'):
                    current_line += 1

        if current_file and current_content:
            files_diff[current_file] = '\n'.join(current_content)

        self.diff_map = line_map
        return files_diff

    def chunk_files(self, files_diff: Dict[str, str]) -> List[Dict[str, str]]:
        """Split files into smaller chunks"""
        chunks = []
        current_chunk = {}
        current_size = 0

        for file_path, content in files_diff.items():
            if len(content) > self.max_chunk_size:
                lines = content.splitlines()
                chunk_lines = []
                current_size = 0

                for line in lines:
                    line_size = len(line) + 1
                    if current_size + line_size > self.max_chunk_size:
                        if chunk_lines:
                            chunks.append({file_path: '\n'.join(chunk_lines)})
                        chunk_lines = [line]
                        current_size = line_size
                    else:
                        chunk_lines.append(line)
                        current_size += line_size

                if chunk_lines:
                    chunks.append({file_path: '\n'.join(chunk_lines)})
            else:
                if current_size + len(content) > self.max_chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = {file_path: content}
                    current_size = len(content)
                else:
                    current_chunk[file_path] = content
                    current_size += len(content)

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def create_review_prompt(self, diff_chunk: Dict[str, str]) -> str:
        """Create a concise prompt for review"""
        files_content = "\n\n".join(
            f"File: {file_path}\n{content}"
            for file_path, content in diff_chunk.items()
        )

        return f"""Review this code change and provide specific, actionable feedback.
        Focus on critical issues only:
        - Bugs and errors
        - Security vulnerabilities
        - Significant performance issues
        - Major design problems
        - Code quality issues
        - Best practices violations

        Format each issue as a JSON object:
        {{
            "file_path": "file path",
            "line_number": line number (must be a line number from the diff that starts with +),
            "message": "clear issue description",
            "severity": "HIGH|MEDIUM|LOW",
            "suggestion": "specific improvement suggestion"
        }}

        Respond ONLY with a JSON array of issues. No other text.

        Code to review:
        {files_content}
        """

    def get_flow_access_token(self):
        """Obter token de acesso do Flow AI"""
        auth_url = "https://api.flowai.com/v1/oauth/token"
        data = {
            "grant_type": "client_credentials",
            "client_id": self.flow_client_id,
            "client_secret": self.flow_client_secret
        }
        try:
            response = self.http_client.post(auth_url, data=data)
            response.raise_for_status()
            return response.json()["access_token"]
        except httpx.HTTPError as e:
            logger.error(f"Failed to obtain Flow AI access token: {e}")
            raise

    def get_review_from_flow(self, diff_chunk: Dict[str, str], retry_count: int = 0) -> List[ReviewComment]:
        """Send chunk for review with retry and rate limiting"""
        try:
            if retry_count > 0:
                time.sleep(retry_count * 5)

            access_token = self.get_flow_access_token()
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "flow-gpt-4",
                "messages": [
                    {"role": "user", "content": self.create_review_prompt(diff_chunk)}
                ]
            }

            response = self.http_client.post(self.flow_api_url, headers=headers, json=data)
            response.raise_for_status()
            response_json = response.json()

            try:
                response_text = response_json["choices"][0]["message"]["content"].strip()
                if not response_text.startswith('['):
                    match = re.search(r'\[(.*)\]', response_text, re.DOTALL)
                    if match:
                        response_text = match.group(0)
                    else:
                        logger.error("Could not find JSON in response")
                        return []

                review_data = json.loads(response_text)
                if not isinstance(review_data, list):
                    logger.error("Response is not a valid JSON list")
                    return []

                comments = [
                    ReviewComment(
                        file_path=item['file_path'],
                        line_number=item['line_number'],
                        message=item['message'],
                        severity=item['severity'],
                        suggestion=item.get('suggestion')
                    )
                    for item in review_data
                    if self.validate_review_item(item)
                ]

                logger.info(f"Generated {len(comments)} comments for chunk")
                return comments

            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON response: {e}")
                if retry_count < 2:
                    return self.get_review_from_flow(diff_chunk, retry_count + 1)
                return []

        except httpx.HTTPError as e:
            logger.error(f"Error getting review: {e}")
            if retry_count < 2:
                return self.get_review_from_flow(diff_chunk, retry_count + 1)
            return []

    def validate_review_item(self, item: Dict) -> bool:
        """Validate if review item has all required fields and line is in diff"""
        try:
            required_fields = ['file_path', 'line_number', 'message', 'severity']
            if not all(field in item and item[field] is not None for field in required_fields):
                return False

            return self.validate_line_number(item['file_path'], item['line_number'])
        except:
            return False

    def validate_line_number(self, file_path: str, line_number: int) -> bool:
        """Check if line is in diff"""
        return (
            file_path in self.diff_map and
            line_number in self.diff_map[file_path]
        )

    def post_review_comments(self, comments: List[ReviewComment]):
        """Post comments to PR using create_review"""
        try:
            repo = self.github_client.get_repo(self.repo_name)
            pr = repo.get_pull(self.pr_number)

            valid_comments = [
                comment for comment in comments
                if self.validate_line_number(comment.file_path, comment.line_number)
            ]

            if not valid_comments:
                logger.warning("No valid comments to post")
                return

            review_comments = []
            for comment in valid_comments:
                review_comments.append({
                    'path': comment.file_path,
                    'position': comment.line_number,
                    'body': f"""
**{comment.severity} Severity Issue**

{comment.message}

**Improvement Suggestion:**
{comment.suggestion if comment.suggestion else 'N/A'}
"""
                })

            if review_comments:
                unique_files = len(set(c['path'] for c in review_comments))
                total_issues = len(review_comments)
                severity_counts = {
                    'HIGH': len([c for c in valid_comments if c.severity == 'HIGH']),
                    'MEDIUM': len([c for c in valid_comments if c.severity == 'MEDIUM']),
                    'LOW': len([c for c in valid_comments if c.severity == 'LOW'])
                }

                review_message = f"""# Automated Code Review

## 🔍 Technical Analysis
- Focus on best practices
- Code pattern verification
- Security analysis
- Optimization suggestions

## 📊 Summary
- Total files analyzed: {unique_files}
- Issues found: {total_issues}

### Severity Distribution:
- 🔴 High: {severity_counts['HIGH']}
- 🟡 Medium: {severity_counts['MEDIUM']}
- 🟢 Low: {severity_counts['LOW']}"""

                response = pr.create_review(
                    body=review_message,
                    event="COMMENT",
                    comments=review_comments
                )
                logger.info(f"Review created with {len(review_comments)} valid comments")

        except Exception as e:
            logger.error(f"Error posting comments: {e}")
            raise

    def run_review(self):
        """Execute the review process"""
        try:
            logger.info("Starting PR review")

            required_vars = ['FLOW_CLIENT_ID', 'FLOW_CLIENT_SECRET', 'GITHUB_TOKEN', 'REPO_NAME', 'PR_NUMBER']
            missing_vars = [var for var in required_vars if not os.environ.get(var)]

            if missing_vars:
                raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

            diff = self.get_pr_diff()

            if not diff.strip():
                logger.info("No changes found")
                return

            files_diff = self.parse_diff_and_create_map(diff)
            chunks = self.chunk_files(files_diff)
            logger.info(f"Diff split into {len(chunks)} chunks")

            all_comments = []
            for i, chunk in enumerate(chunks, 1):
                logger.info(f"Processing chunk {i} of {len(chunks)}")
                comments = self.get_review_from_flow(chunk)
                if comments:
                    all_comments.extend(comments)
                time.sleep(2)

            if all_comments:
                self.post_review_comments(all_comments)
                logger.info(f"Review completed with {len(all_comments)} comments")
            else:
                logger.info("No issues found in code")

        except Exception as e:
            logger.error(f"Error during review: {e}")
            sys.exit(1)

def main():
    reviewer = PRReviewer()
    reviewer.run_review()

if __name__ == "__main__":
    main()