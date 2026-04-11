# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.6.x   | :white_check_mark: |
| 0.5.x   | :white_check_mark: |
| < 0.5   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

1. **Do NOT** open a public GitHub issue for security vulnerabilities
2. Open a [private security advisory](https://github.com/luongnv89/claude-codex-local/security/advisories/new) on GitHub
3. Include detailed steps to reproduce the vulnerability
4. Allow up to 48 hours for an initial response

### What to Include

- Type of vulnerability
- Full paths of affected source files
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce
- Proof-of-concept or exploit code (if possible)
- Impact of the issue

### What to Expect

- Acknowledgment of your report within 48 hours
- Regular updates on our progress
- Credit in the security advisory (if desired)
- Notification when the issue is fixed

## Security Best Practices

When contributing to this project:

- Never commit secrets, API keys, or credentials
- Use environment variables for sensitive configuration
- Follow secure coding practices
- The project uses `detect-secrets` and `bandit` in pre-commit hooks — run them locally before pushing
- Report any security concerns immediately
