import os
import sys
import anthropic
from github import Github
from git import Repo
import json
from typing import List, Dict, Any
from dataclasses import dataclass
import logging
import time
import re

# Configurar logging
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
        self.anthropic_client = anthropic.Anthropic(
            api_key=os.environ.get('ANTHROPIC_API_KEY')
        )
        self.github_client = Github(os.environ.get('GITHUB_TOKEN'))
        self.repo_name = os.environ.get('REPO_NAME')
        self.pr_number = int(os.environ.get('PR_NUMBER'))
        self.max_chunk_size = 15000  # Reduzido para evitar exceder limites
        self.diff_map = {}  # Mapa para armazenar informações do diff

    def get_pr_diff(self) -> str:
        """Obtém o diff do PR atual"""
        try:
            repo = Repo('.')
            repo.git.fetch('origin', 'main:main')

            base_sha = repo.rev_parse('main')
            head_sha = repo.head.commit

            diff = repo.git.diff(f"{base_sha}...{head_sha}")
            logger.info(f"Obtido diff com {len(diff.splitlines())} linhas")
            return diff

        except Exception as e:
            logger.error(f"Erro ao obter diff: {e}")
            raise

    def parse_diff_and_create_map(self, diff: str) -> Dict[str, str]:
        """Parse o diff e cria um mapa de linhas modificadas"""
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
                # Parse hunk header
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
        """Divide os arquivos em chunks menores"""
        chunks = []
        current_chunk = {}
        current_size = 0

        for file_path, content in files_diff.items():
            if len(content) > self.max_chunk_size:
                # Divide arquivo grande em múltiplos chunks
                lines = content.splitlines()
                chunk_lines = []
                current_size = 0

                for line in lines:
                    line_size = len(line) + 1  # +1 para newline
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
        """Cria um prompt mais conciso para o Claude"""
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

    def get_review_from_claude(self, diff_chunk: Dict[str, str], retry_count: int = 0) -> List[ReviewComment]:
        """Envia chunk para review com retry e rate limiting"""
        try:
            if retry_count > 0:
                time.sleep(retry_count * 5)  # Backoff exponencial

            message = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                messages=[{
                    "role": "user",
                    "content": self.create_review_prompt(diff_chunk)
                }]
            )

            try:
                # Tenta extrair JSON da resposta
                response_text = message.content[0].text.strip()
                if not response_text.startswith('['):
                    # Procura por array JSON na resposta
                    match = re.search(r'\[(.*)\]', response_text, re.DOTALL)
                    if match:
                        response_text = match.group(0)
                    else:
                        logger.error("Não foi possível encontrar JSON na resposta")
                        return []

                review_data = json.loads(response_text)
                if not isinstance(review_data, list):
                    logger.error("Resposta não é uma lista JSON válida")
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

                logger.info(f"Gerados {len(comments)} comentários para o chunk")
                return comments

            except json.JSONDecodeError as e:
                logger.error(f"Erro ao parsear resposta JSON: {e}")
                if retry_count < 2:
                    return self.get_review_from_claude(diff_chunk, retry_count + 1)
                return []

        except anthropic.RateLimitError:
            logger.warning(f"Rate limit atingido, tentativa {retry_count + 1}")
            if retry_count < 3:
                time.sleep(60)  # Espera 1 minuto
                return self.get_review_from_claude(diff_chunk, retry_count + 1)
            return []
        except Exception as e:
            logger.error(f"Erro ao obter review: {e}")
            if retry_count < 2:
                return self.get_review_from_claude(diff_chunk, retry_count + 1)
            return []

    def validate_review_item(self, item: Dict) -> bool:
        """Valida se o item do review tem todos os campos necessários e a linha está no diff"""
        try:
            required_fields = ['file_path', 'line_number', 'message', 'severity']
            if not all(field in item and item[field] is not None for field in required_fields):
                return False

            return self.validate_line_number(item['file_path'], item['line_number'])
        except:
            return False

    def validate_line_number(self, file_path: str, line_number: int) -> bool:
        """Verifica se a linha está no diff"""
        return (
            file_path in self.diff_map and
            line_number in self.diff_map[file_path]
        )

    def post_review_comments(self, comments: List[ReviewComment]):
        """Posta comentários no PR usando create_review"""
        try:
            repo = self.github_client.get_repo(self.repo_name)
            pr = repo.get_pull(self.pr_number)

            # Filtra apenas comentários em linhas que foram modificadas
            valid_comments = [
                comment for comment in comments
                if self.validate_line_number(comment.file_path, comment.line_number)
            ]

            if not valid_comments:
                logger.warning("Nenhum comentário válido para postar")
                return

            # Cria um único review com todos os comentários válidos
            review_comments = []
            for comment in valid_comments:
                review_comments.append({
                    'path': comment.file_path,
                    'position': comment.line_number,
                    'body': f"""
**{comment.severity} Severity Issue**

{comment.message}

**Sugestão de melhoria:**
{comment.suggestion if comment.suggestion else 'N/A'}
"""
                })

            if review_comments:
                # Calcula estatísticas para a mensagem
                unique_files = len(set(c['path'] for c in review_comments))
                total_issues = len(review_comments)
                severity_counts = {
                    'HIGH': len([c for c in valid_comments if c.severity == 'HIGH']),
                    'MEDIUM': len([c for c in valid_comments if c.severity == 'MEDIUM']),
                    'LOW': len([c for c in valid_comments if c.severity == 'LOW'])
                }

                review_message = f"""# Code Review Automatizado

## 🔍 Análise Técnica
- Revisão focada em boas práticas
- Verificação de padrões de código
- Análise de segurança
- Sugestões de otimização

## 📊 Resumo
- Total de arquivos analisados: {unique_files}
- Sugestões encontradas: {total_issues}

### Distribuição por Severidade:
- 🔴 Alta: {severity_counts['HIGH']}
- 🟡 Média: {severity_counts['MEDIUM']}
- 🟢 Baixa: {severity_counts['LOW']}

---
Gerado por Claude-3 | v1.0"""

                response = pr.create_review(
                    body=review_message,
                    event="COMMENT",
                    comments=review_comments
                )
                logger.info(f"Review criado com {len(review_comments)} comentários válidos")

        except Exception as e:
            logger.error(f"Erro ao postar comentários: {e}")
            raise

    def run_review(self):
        """Executa o processo de review"""
        try:
            logger.info("Iniciando review do PR")

            # Verificar variáveis de ambiente
            required_vars = ['ANTHROPIC_API_KEY', 'GITHUB_TOKEN', 'REPO_NAME', 'PR_NUMBER']
            missing_vars = [var for var in required_vars if not os.environ.get(var)]

            if missing_vars:
                raise ValueError(f"Variáveis de ambiente faltando: {', '.join(missing_vars)}")

            # Obter e processar o diff
            diff = self.get_pr_diff()

            if not diff.strip():
                logger.info("Nenhuma alteração encontrada")
                return

            # Criar mapa de diff e dividir em chunks
            files_diff = self.parse_diff_and_create_map(diff)
            chunks = self.chunk_files(files_diff)
            logger.info(f"Diff dividido em {len(chunks)} chunks")

            # Processar chunks
            all_comments = []
            for i, chunk in enumerate(chunks, 1):
                logger.info(f"Processando chunk {i} de {len(chunks)}")
                comments = self.get_review_from_claude(chunk)
                if comments:
                    all_comments.extend(comments)
                time.sleep(2)  # Rate limiting

            # Postar comentários
            if all_comments:
                self.post_review_comments(all_comments)
                logger.info(f"Review concluído com {len(all_comments)} comentários")
            else:
                logger.info("Nenhum problema encontrado no código")

        except Exception as e:
            logger.error(f"Erro durante o review: {e}")
            sys.exit(1)

def main():
    reviewer = PRReviewer()
    reviewer.run_review()

if __name__ == "__main__":
    main()
