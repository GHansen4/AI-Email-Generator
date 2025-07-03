# Environment Configuration Guide

## Quick Setup

1. **Copy the example file:**
   `ash
   cp .env.example .env
   `

2. **Generate required keys:**
   `ash
   # Generate SECRET_KEY
   python -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(32))"
   
   # Generate ENCRYPTION_KEY (optional but recommended)
   python -c "from cryptography.fernet import Fernet; print('ENCRYPTION_KEY=' + Fernet.generate_key().decode())"
   `

3. **Configure services:**
   - **Mailgun**: Sign up at https://mailgun.com/ and get API keys
   - **Claude AI**: Get API key from https://console.anthropic.com/
   - **Database**: Use SQLite for development or PostgreSQL for production

## Environment Variables Reference

### Required Variables
| Variable | Description | Example |
|----------|-------------|---------|
| DATABASE_URL | Database connection string | sqlite:///./app.db or postgresql://user:pass@host/db |
| MAILGUN_API_KEY | Mailgun API key | key-1234567890abcdef |
| MAILGUN_DOMAIN | Mailgun domain | mg.your-domain.com |
| MAILGUN_WEBHOOK_SIGNING_KEY | Webhook verification key | whsec_1234567890abcdef |
| CLAUDE_API_KEY | Anthropic Claude API key | sk-ant-api01-... |
| SECRET_KEY | Application secret key | Generate with secrets.token_urlsafe(32) |
| SERVICE_EMAIL | Service email address | i-assistant@your-domain.com |

### Optional Variables
| Variable | Default | Description |
|----------|---------|-------------|
| DEBUG | alse | Enable debug mode |
| LOG_LEVEL | INFO | Logging level |
| ENCRYPTION_KEY | None | Data encryption key |
| BLOCKED_DOMAINS | None | Comma-separated blocked domains |
| MAX_REQUESTS_PER_MINUTE | 30 | Rate limiting |

## Database Setup

### SQLite (Development)
`env
DATABASE_URL=sqlite:///./ai_email_generator.db
`

### PostgreSQL (Production)
`env
DATABASE_URL=postgresql://username:password@localhost:5432/ai_email_generator
`

## Security Best Practices

1. **Never commit .env files** - They're in .gitignore
2. **Use strong SECRET_KEY** - Generate with secrets.token_urlsafe(32)
3. **Enable ENCRYPTION_KEY** - For sensitive data encryption
4. **Configure BLOCKED_DOMAINS** - Block disposable email services
5. **Set appropriate rate limits** - Prevent abuse

## Troubleshooting

- **Configuration errors**: Check all required variables are set
- **Database errors**: Verify DATABASE_URL format and permissions
- **Mailgun errors**: Verify API keys and domain configuration
- **Claude API errors**: Check API key and usage limits
