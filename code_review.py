import os
import sys
import requests
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
        self.flow_api_url = "https://api.flowai.com/v1/chat/completions"  # Ajuste esta URL conforme necessário
        self.github_client = Github(os.environ.get('GITHUB_TOKEN'))
        self.repo_name = os.environ.get('REPO_NAME')
        self.pr_number = int(os.environ.get('PR_NUMBER'))
        self.max_chunk_size = 15000
        self.diff_map = {}

    # ... (mantenha os métodos get_pr_diff, parse_diff_and_create_map, chunk_files, create_review_prompt inalterados)

    def get_flow_access_token(self):
        """Obter token de acesso do Flow AI"""
        auth_url = "https://api.flowai.com/v1/oauth/token"  # Ajuste esta URL conforme necessário
        data = {
            "grant_type": "client_credentials",
            "client_id": self.flow_client_id,
            "client_secret": self.flow_client_secret
        }
        response = requests.post(auth_url, data=data)
        if response.status_code == 200:
            return response.json()["access_token"]
        else:
            raise Exception("Failed to obtain Flow AI access token")

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
                "model": "flow-gpt-4",  # Ajuste o modelo conforme necessário
                "messages": [
                    {"role": "user", "content": self.create_review_prompt(diff_chunk)}
                ]
            }

            response = requests.post(self.flow_api_url, headers=headers, json=data)
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

        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting review: {e}")
            if retry_count < 2:
                return self.get_review_from_flow(diff_chunk, retry_count + 1)
            return []

    # ... (mantenha os métodos validate_review_item, validate_line_number, post_review_comments inalterados)

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