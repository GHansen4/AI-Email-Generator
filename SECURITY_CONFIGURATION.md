# AI Email Generator - Security Configuration Guide

This document provides comprehensive security configuration instructions for deploying the AI Email Generator in production.

## 🔐 Required Environment Variables

### Core Application Settings
```bash
# Database
DATABASE_URL=postgresql://username:password@localhost:5432/ai_email_generator

# Mailgun (Required for email processing)
MAILGUN_API_KEY=your_mailgun_api_key
MAILGUN_DOMAIN=your_mailgun_domain.com
MAILGUN_WEBHOOK_SIGNING_KEY=your_mailgun_webhook_signing_key

# Claude AI
CLAUDE_API_KEY=your_anthropic_claude_api_key

# Application Security
SECRET_KEY=your_super_secret_key_here_minimum_32_characters
SERVICE_EMAIL=ai-assistant@your-domain.com
```

## 🛡️ Security Settings

### Data Encryption (Recommended for Production)
```bash
# Generate encryption key with:
# python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
ENCRYPTION_KEY=your_base64_encoded_encryption_key
```

### Email Security
```bash
BLOCKED_DOMAINS=tempmail.org,guerrillamail.com,10minutemail.com
MAX_EMAIL_SIZE=1048576  # 1MB in bytes
MAX_EMAILS_PER_HOUR=20  # Per email address
```

### Rate Limiting
```bash
MAX_REQUESTS_PER_MINUTE=30
MAX_FAILED_ATTEMPTS_PER_HOUR=10
API_RATE_LIMIT_ENABLED=true
API_BURST_LIMIT=100
```

### Webhook Security
```bash
WEBHOOK_TIMESTAMP_TOLERANCE=300  # 5 minutes in seconds
```

### Content Security
```bash
CONTENT_SANITIZATION_ENABLED=true
MALICIOUS_CONTENT_DETECTION=true
```

### Application Settings
```bash
DEBUG=false
LOG_LEVEL=INFO
```

## 🔧 Security Features Implemented

### 1. Webhook Signature Verification
- **Purpose**: Prevents unauthorized webhook calls
- **Implementation**: HMAC-SHA256 signature verification
- **Configuration**: Set `MAILGUN_WEBHOOK_SIGNING_KEY`
- **Monitoring**: Failed attempts logged with `security_event=true`

### 2. Email Validation & Rate Limiting
- **Email Format Validation**: RFC-compliant email validation
- **Domain Blocking**: Configurable blocked domains list
- **Rate Limiting**: Per-IP and per-email rate limits
- **Abuse Prevention**: Failed attempt tracking and temporary blocking

### 3. Content Security
- **Content Sanitization**: Removes malicious HTML/JavaScript
- **Size Limits**: Configurable email content size limits
- **Malicious Pattern Detection**: Scans for suspicious content patterns
- **Input Validation**: Comprehensive input sanitization

### 4. Data Protection
- **Encryption**: Sensitive data encrypted at rest using Fernet
- **Secure Headers**: Proper HTTP security headers
- **Request Tracking**: Unique request IDs for audit trails
- **Privacy Protection**: Email masking in logs

### 5. Security Monitoring
- **Security Event Logging**: All security events tagged and logged
- **Failed Attempt Tracking**: Automatic blocking after repeated failures
- **Performance Monitoring**: Request timing and error tracking
- **Audit Trail**: Complete request lifecycle logging

## 🚀 Production Deployment Checklist

### Security Hardening
- [ ] Generate strong `SECRET_KEY` (32+ characters)
- [ ] Set `DEBUG=false`
- [ ] Configure HTTPS for all endpoints
- [ ] Generate and securely store `ENCRYPTION_KEY`
- [ ] Set appropriate `LOG_LEVEL` (INFO or WARNING)
- [ ] Configure comprehensive `BLOCKED_DOMAINS` list
- [ ] Set production-appropriate rate limits
- [ ] Enable webhook signature verification

### Database Security
- [ ] Use PostgreSQL for production
- [ ] Secure database credentials
- [ ] Enable database encryption
- [ ] Set up regular automated backups
- [ ] Configure database access controls

### Mailgun Configuration
- [ ] Add domain to Mailgun account
- [ ] Configure webhook URL: `https://your-domain.com/webhook/mailgun`
- [ ] Enable webhook signature verification
- [ ] Test email delivery and webhook processing
- [ ] Configure SPF/DKIM/DMARC records

### Monitoring & Alerting
- [ ] Set up log aggregation
- [ ] Configure alerts for `security_event=true` logs
- [ ] Monitor rate limiting events
- [ ] Track failed authentication attempts
- [ ] Monitor API performance and errors
- [ ] Set up uptime monitoring

## 🔍 Security Monitoring

### Log Analysis
Search for these patterns in your logs:

```bash
# Security events
security_event=true

# Failed webhook signatures
"webhook_signature_invalid"

# Rate limiting violations
"rate_limit_exceeded"

# Email validation failures
"email_validation_failed"

# Malicious content detection
"Malicious content pattern detected"
```

### Key Metrics to Monitor
- Failed webhook signature attempts
- Rate limit violations per hour
- Email validation failure rate
- Content sanitization triggers
- API response times
- Database query performance

## 🚨 Incident Response

### Suspected Attack
1. Check logs for `security_event=true` entries
2. Identify attack patterns (IP addresses, email domains)
3. Temporarily add domains/IPs to block lists
4. Increase rate limiting if needed
5. Monitor for continued suspicious activity

### Performance Issues
1. Check API response times in logs
2. Monitor database query performance
3. Verify rate limiting settings
4. Check external service status (Mailgun, Claude)

## 📊 Security Best Practices

### Regular Maintenance
- Review and update blocked domains list monthly
- Rotate encryption keys quarterly
- Review security logs weekly
- Update dependencies regularly
- Test backup and recovery procedures

### Access Control
- Limit database access to application only
- Use environment-specific API keys
- Implement principle of least privilege
- Regular access audits

### Data Handling
- Encrypt sensitive data at rest
- Use HTTPS for all communications
- Implement data retention policies
- Regular security assessments

## 🔗 Additional Resources

- [Mailgun Security Documentation](https://documentation.mailgun.com/en/latest/user_manual.html#webhooks-security)
- [Anthropic Claude API Security](https://docs.anthropic.com/claude/docs/api-security)
- [FastAPI Security Best Practices](https://fastapi.tiangolo.com/tutorial/security/)
- [PostgreSQL Security](https://www.postgresql.org/docs/current/security.html)

## 📞 Support

For security-related questions or to report vulnerabilities:
- Review application logs for detailed error information
- Check this configuration guide for common issues
- Ensure all environment variables are properly set
- Verify external service connectivity (Mailgun, Claude API) 