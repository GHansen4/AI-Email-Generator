# AI Email Generator

A sophisticated AI-powered email response system that analyzes users' writing styles and generates indistinguishable email responses using advanced linguistic profiling and Claude AI.

## 🚀 Features

### Advanced Linguistic Analysis
- **16-Dimension Linguistic Fingerprinting**: Comprehensive analysis of writing patterns
- **Syntactic Pattern Analysis**: Sentence complexity, coordination/subordination preferences
- **Lexical Sophistication**: Vocabulary diversity, word complexity, lexical density
- **Pragmatic Competence**: Politeness strategies, speech acts, mitigation patterns
- **Cohesion Analysis**: Text flow, transitions, reference patterns
- **Dependency Patterns**: Active/passive voice preferences, question formation
- **Temporal Deixis**: Time reference patterns and temporal orientation

### AI Response Generation
- **Indistinguishable Style Matching**: Responses that sound exactly like the user wrote them
- **Claude AI Integration**: Powered by Anthropic's Claude for high-quality generation
- **Advanced Prompt Engineering**: Sophisticated style instructions for precise matching
- **Adaptive Learning**: Profiles improve with more email samples

### Email Processing
- **Mailgun Integration**: Webhook-based email processing
- **Profile Building**: Automatic style learning from user emails
- **Response Requests**: Forward emails to get AI-generated draft responses
- **Secure Processing**: Webhook signature verification and email validation

## 🏗️ Architecture

### Core Components

- **WritingProfileAnalyzer**: Advanced linguistic analysis engine
- **AIResponseGenerator**: Claude-powered response generation with style matching
- **EmailParser**: Intelligent parsing of forwarded emails and profile content
- **MailgunClient**: Email delivery and webhook handling
- **Database Models**: User profiles, email logs, and writing patterns

### Technology Stack

- **Backend**: FastAPI (Python)
- **AI**: Anthropic Claude API
- **Email**: Mailgun API
- **Database**: SQLAlchemy (PostgreSQL/SQLite)
- **NLP**: NLTK, spaCy (optional), textstat (optional)
- **Security**: Webhook signature verification

## 📦 Installation

### Prerequisites

- Python 3.9+
- PostgreSQL (or SQLite for development)
- Mailgun account
- Anthropic Claude API key

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/AI-Email-Generator.git
   cd AI-Email-Generator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Configuration**
   
   Create a `.env` file with your configuration:
   ```env
   # Database
   DATABASE_URL=postgresql://user:password@localhost:5432/ai_email_db
   
   # Mailgun
   MAILGUN_API_KEY=your_mailgun_api_key
   MAILGUN_DOMAIN=your_mailgun_domain
   MAILGUN_WEBHOOK_SIGNING_KEY=your_webhook_signing_key
   
   # Claude API
   CLAUDE_API_KEY=your_anthropic_api_key
   
   # Application
   SECRET_KEY=your_secret_key
   SERVICE_EMAIL=your_service_email@domain.com
   DEBUG=False
   
   # Logging
   LOG_LEVEL=INFO
   ```

4. **Database Setup**
   ```bash
   python -m app.database
   ```

5. **Optional NLP Libraries**
   ```bash
   # For enhanced analysis (optional)
   pip install spacy
   python -m spacy download en_core_web_sm
   
   # For readability metrics (optional)
   pip install textstat
   ```

## 🔧 Usage

### Starting the Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Email Processing

#### Building Your Writing Profile

Send emails with subject line starting with "PROFILE:" to build your writing style profile:

```
Subject: PROFILE: My writing style sample
Body: This is how I typically write emails. I usually start with a friendly greeting...
```

#### Getting AI Responses

Forward any email to the service to get an AI-generated response draft:

```
Subject: Fwd: Meeting Request
Body: [Original forwarded email content]
```

### API Endpoints

- `POST /webhook/mailgun` - Mailgun webhook for processing incoming emails
- Health check and monitoring endpoints available

## 🧠 How It Works

### 1. **Profile Building Phase**
- User sends sample emails with "PROFILE:" subject
- System performs 16-dimension linguistic analysis
- Writing fingerprint is generated and stored
- Confirmation email sent to user

### 2. **Response Generation Phase**
- User forwards email needing response
- System parses original email content
- User's writing profile is loaded
- Advanced prompt is generated with style instructions
- Claude AI generates response matching user's style
- Draft response is emailed back to user

### 3. **Continuous Learning**
- Each new profile email improves the analysis
- Writing patterns are merged and refined
- Confidence scores increase with more samples

## 🔬 Advanced Features

### Linguistic Analysis Dimensions

1. **Syntactic Complexity**: Sentence structure preferences
2. **Coordination Preference**: Use of coordinating conjunctions
3. **Subordination Preference**: Use of subordinating conjunctions
4. **Active Voice Preference**: Active vs passive voice ratio
5. **Question Directness**: Question formation patterns
6. **Vocabulary Diversity**: Type-token ratios and lexical variety
7. **Word Sophistication**: Average word length and complexity
8. **Lexical Density**: Content word ratios
9. **Cohesion Strength**: Text coherence patterns
10. **Transition Usage**: Discourse marker frequency
11. **Mitigation Tendency**: Hedging and softening language
12. **Certainty Level**: Confidence expression patterns
13. **Politeness Complexity**: Courtesy strategy sophistication
14. **Formality**: Formal vs casual language markers
15. **Enthusiasm**: Emotional expression patterns
16. **Directness**: Communication directness vs indirectness

### Prompt Engineering

The system generates sophisticated prompts that include:
- Syntactic pattern instructions
- Lexical sophistication guidelines
- Pragmatic competence requirements
- Cohesion and flow specifications
- Specific pattern replication instructions

## 🛡️ Security

### Comprehensive Security Architecture
The AI Email Generator implements enterprise-grade security measures:

#### **Multi-Layer Security**
- **Webhook Signature Verification**: HMAC-SHA256 verification prevents unauthorized requests
- **Email Validation & Filtering**: RFC-compliant validation with configurable domain blocking
- **Rate Limiting**: Per-IP and per-email rate limits with abuse prevention
- **Content Security**: Real-time sanitization and malicious pattern detection
- **Data Encryption**: Fernet encryption for sensitive data at rest

#### **Security Monitoring**
- **Comprehensive Logging**: All security events tagged and tracked
- **Failed Attempt Tracking**: Automatic blocking after repeated failures
- **Request Tracing**: Unique request IDs for complete audit trails
- **Performance Monitoring**: Real-time security metric tracking

#### **Production Security Features**
- **Threat Detection**: Automated malicious content scanning
- **Privacy Protection**: Email address masking in logs
- **Secure Configuration**: Environment-based security settings
- **Compliance Ready**: Comprehensive audit logging and data protection

### Security Configuration
For detailed security setup and production deployment, see our [**Security Configuration Guide**](SECURITY_CONFIGURATION.md).

Key security settings include:
```env
# Webhook Security
MAILGUN_WEBHOOK_SIGNING_KEY=your_webhook_signing_key

# Data Protection  
ENCRYPTION_KEY=your_encryption_key

# Rate Limiting
MAX_REQUESTS_PER_MINUTE=30
MAX_FAILED_ATTEMPTS_PER_HOUR=10

# Content Security
CONTENT_SANITIZATION_ENABLED=true
MALICIOUS_CONTENT_DETECTION=true
```

## 🚀 Deployment

### Production Considerations

- Use PostgreSQL for production database
- Set up proper logging and monitoring
- Configure HTTPS for webhook endpoints
- Set up environment-specific configurations
- Implement rate limiting and error handling

### Environment Variables

Ensure all required environment variables are properly configured for your deployment environment.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Open an issue on GitHub
- Check the documentation
- Review the configuration examples

## 🔮 Future Enhancements

- Multi-language support
- Advanced email threading
- Custom model fine-tuning
- Real-time style adaptation
- Enhanced security features
- Performance optimizations

---

**Built with ❤️ using Python, FastAPI, and Anthropic Claude AI** 