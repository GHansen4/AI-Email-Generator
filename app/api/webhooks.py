import asyncio
from datetime import datetime
from fastapi import APIRouter, Form, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Dict, Any

from app.database import get_db
from app.models.user import User, WritingProfile
from app.models.email import EmailLog, ForwardedEmail
from app.services.email_parser import EmailParser
from app.services.profile_analyzer import WritingProfileAnalyzer
from app.services.ai_generator import AIResponseGenerator
from app.services.mailgun_client import MailgunClient
from app.utils.security import verify_mailgun_webhook, validate_email_address
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Initialize services
email_parser = EmailParser()
profile_analyzer = WritingProfileAnalyzer()
ai_generator = AIResponseGenerator()
mailgun_client = MailgunClient()


@router.post("/webhook/mailgun")
async def mailgun_webhook(
    timestamp: str = Form(...),
    token: str = Form(...),
    signature: str = Form(...),
    sender: str = Form(...),
    recipient: str = Form(...),
    subject: str = Form(...),
    body_plain: str = Form(default=""),
    body_html: str = Form(default=""),
    message_id: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Handle incoming emails from Mailgun webhook.
    
    This endpoint processes both profile emails (subject starts with "PROFILE:")
    and regular forwarded emails that need AI responses.
    """
    try:
        # Verify webhook signature for security
        if not verify_mailgun_webhook(timestamp, token, signature):
            logger.warning("Invalid webhook signature", sender=sender)
            raise HTTPException(status_code=401, detail="Invalid signature")
        
        # Validate email addresses
        if not validate_email_address(sender):
            logger.warning("Invalid sender email", sender=sender)
            raise HTTPException(status_code=400, detail="Invalid sender email")
        
        logger.info("Processing incoming email", 
                   sender=sender, 
                   subject=subject,
                   message_id=message_id)
        
        # Get or create user
        user = get_or_create_user(db, sender)
        
        # Create email log entry and commit immediately to get ID
        email_log = EmailLog(
            user_id=user.id,
            message_id=message_id,
            from_email=sender,
            to_email=recipient,
            subject=subject,
            raw_content=body_plain or body_html,
            processing_status="processing"
        )
        db.add(email_log)
        db.commit()  # Commit early to get the ID
        db.refresh(email_log)  # Refresh to get the assigned ID
        
        # Determine email type and process accordingly
        try:
            if subject.upper().startswith("PROFILE:"):
                await process_profile_email(db, email_log, user, body_plain or body_html)
            else:
                await process_response_request(db, email_log, user, body_plain, body_html)
            
            # Final commit for all changes
            db.commit()
            
        except Exception as process_error:
            # Rollback any changes from processing functions
            db.rollback()
            
            # Update email log with error status
            email_log.processing_status = "failed"  # type: ignore  # SQLAlchemy column assignment
            email_log.error_message = str(process_error)  # type: ignore  # SQLAlchemy column assignment
            db.add(email_log)
            db.commit()
            
            logger.error("Error in email processing", error=str(process_error), email_log_id=email_log.id)
            raise process_error
        
        return {"status": "success", "message": "Email processed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error processing webhook", error=str(e))
        
        # Attempt to rollback any incomplete transactions
        try:
            db.rollback()
        except Exception as rollback_error:
            logger.error("Failed to rollback transaction", error=str(rollback_error))
        
        raise HTTPException(status_code=500, detail="Internal server error")


async def process_profile_email(db: Session, email_log: EmailLog, user: User, content: str):
    """Process profile-building email (subject starts with PROFILE:)."""
    try:
        email_log.email_type = "profile"  # type: ignore  # SQLAlchemy column assignment
        
        # Extract content for profile analysis
        profile_content = email_parser.extract_profile_content(content, str(email_log.subject))
        
        if not profile_content:
            logger.warning("No valid content extracted for profile", user_id=user.id)
            email_log.processing_status = "failed"  # type: ignore  # SQLAlchemy column assignment
            email_log.error_message = "Could not extract valid content for profile analysis"  # type: ignore  # SQLAlchemy column assignment
            return
        
        # Analyze writing style (now comprehensive by default)
        analysis_result = profile_analyzer.analyze_writing_style(profile_content)
        
        # Get existing writing profile or create new one
        writing_profile = db.query(WritingProfile).filter(
            WritingProfile.user_id == user.id
        ).first()
        
        if writing_profile:
            # Merge with existing profile
            merged_profile = profile_analyzer.merge_profiles(
                {
                    'avg_sentence_length': writing_profile.avg_sentence_length,
                    'avg_paragraph_length': writing_profile.avg_paragraph_length,
                    'formality_score': writing_profile.formality_score,
                    'enthusiasm_score': writing_profile.enthusiasm_score,
                    'common_greetings': writing_profile.common_greetings or [],
                    'common_closings': writing_profile.common_closings or [],
                    'common_phrases': writing_profile.common_phrases or [],
                    'vocabulary_level': writing_profile.vocabulary_level
                },
                analysis_result,
                int(writing_profile.sample_count)  # type: ignore[arg-type]  # SQLAlchemy column conversion
            )
            
            # Update existing profile
            writing_profile.avg_sentence_length = merged_profile['avg_sentence_length']
            writing_profile.avg_paragraph_length = merged_profile['avg_paragraph_length']
            writing_profile.formality_score = merged_profile['formality_score']
            writing_profile.enthusiasm_score = merged_profile['enthusiasm_score']
            writing_profile.common_greetings = merged_profile['common_greetings']
            writing_profile.common_closings = merged_profile['common_closings']
            writing_profile.common_phrases = merged_profile['common_phrases']
            writing_profile.vocabulary_level = merged_profile['vocabulary_level']
            writing_profile.sample_count = writing_profile.sample_count + 1  # type: ignore  # SQLAlchemy column operation
            writing_profile.confidence_score = min(1.0, float(writing_profile.sample_count) * 0.1)  # type: ignore[arg-type]  # SQLAlchemy column conversion
            writing_profile.updated_at = datetime.utcnow()  # type: ignore  # SQLAlchemy column assignment
            
        else:
            # Create new profile
            writing_profile = WritingProfile(
                user_id=user.id,
                avg_sentence_length=analysis_result['avg_sentence_length'],
                avg_paragraph_length=analysis_result['avg_paragraph_length'],
                formality_score=analysis_result['formality_score'],
                enthusiasm_score=analysis_result['enthusiasm_score'],
                common_greetings=analysis_result['common_greetings'],
                common_closings=analysis_result['common_closings'],
                common_phrases=analysis_result['common_phrases'],
                vocabulary_level=analysis_result['vocabulary_level'],
                sample_count=1,
                confidence_score=0.1
            )
            db.add(writing_profile)
        
        # Generate confirmation response
        confirmation_message = ai_generator.generate_profile_response(str(user.email))
        
        # Send confirmation email
        send_result = await mailgun_client.send_email(
            to_email=str(user.email),
            subject="Writing Profile Updated - AI Email Assistant",
            content=confirmation_message
        )
        
        if send_result['success']:
            email_log.processing_status = "completed"  # type: ignore  # SQLAlchemy column assignment
            logger.info("Profile email processed successfully", user_id=user.id)
        else:
            email_log.processing_status = "failed"  # type: ignore  # SQLAlchemy column assignment
            email_log.error_message = f"Failed to send confirmation: {send_result.get('error')}"  # type: ignore  # SQLAlchemy column assignment
        
    except Exception as e:
        logger.error("Error processing profile email", error=str(e), user_id=user.id)
        email_log.processing_status = "failed"  # type: ignore  # SQLAlchemy column assignment
        email_log.error_message = str(e)  # type: ignore  # SQLAlchemy column assignment
        raise  # Re-raise to trigger rollback in main handler


async def process_response_request(db: Session, email_log: EmailLog, user: User, 
                                 body_plain: str, body_html: str):
    """Process request for AI-generated email response."""
    try:
        email_log.email_type = "response_request"  # type: ignore  # SQLAlchemy column assignment
        
        # Parse the forwarded email
        content = body_plain or body_html
        is_html = bool(body_html and not body_plain)
        
        parsed_email = email_parser.parse_forwarded_email(content, is_html)
        
        if not parsed_email['success']:
            logger.warning("Failed to parse forwarded email", user_id=user.id)
            email_log.processing_status = "failed"  # type: ignore  # SQLAlchemy column assignment
            email_log.error_message = "Could not parse forwarded email"  # type: ignore  # SQLAlchemy column assignment
            return
        
        # Store forwarded email details - NOW email_log.id is available
        forwarded_email = ForwardedEmail(
            email_log_id=email_log.id,  # This now has a valid ID
            original_from=parsed_email.get('original_from'),
            original_to=parsed_email.get('original_to'),
            original_subject=parsed_email.get('original_subject'),
            original_content=parsed_email.get('original_content'),
            forwarded_by=user.email,
            forward_type=parsed_email.get('forward_type')
        )
        db.add(forwarded_email)
        
        # Get user's writing profile
        writing_profile = db.query(WritingProfile).filter(
            WritingProfile.user_id == user.id
        ).first()
        
        if not writing_profile:
            logger.info("No writing profile found, creating default", user_id=user.id)
            # Create basic profile
            profile_data = profile_analyzer._get_default_profile()
            profile_data['sample_count'] = 0
            profile_data['confidence_score'] = 0.0
        else:
            profile_data = {
                'avg_sentence_length': writing_profile.avg_sentence_length,
                'avg_paragraph_length': writing_profile.avg_paragraph_length,
                'formality_score': writing_profile.formality_score,
                'enthusiasm_score': writing_profile.enthusiasm_score,
                'common_greetings': writing_profile.common_greetings or [],
                'common_closings': writing_profile.common_closings or [],
                'common_phrases': writing_profile.common_phrases or [],
                'vocabulary_level': writing_profile.vocabulary_level,
                'sample_count': writing_profile.sample_count,
                'confidence_score': writing_profile.confidence_score
            }
        
        # Generate AI response
        start_time = datetime.utcnow()
        ai_result = ai_generator.generate_response(
            original_email=parsed_email['original_content'],
            user_profile=profile_data,
            context=f"Original sender: {parsed_email.get('original_from')}"
        )
        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        # Update email log with AI response details
        email_log.ai_response = ai_result.get('response')  # type: ignore  # SQLAlchemy column assignment
        email_log.prompt_used = ai_result.get('prompt_used')  # type: ignore  # SQLAlchemy column assignment
        email_log.openai_model = ai_result.get('model')  # type: ignore  # SQLAlchemy column assignment
        email_log.tokens_used = ai_result.get('tokens_used', 0)  # type: ignore  # SQLAlchemy column assignment
        email_log.response_time_ms = processing_time  # type: ignore  # SQLAlchemy column assignment
        
        if ai_result['success']:
            # Send the AI response back to the user
            response_subject = f"AI Draft Response: {parsed_email.get('original_subject', 'No Subject')}"
            
            send_result = await mailgun_client.send_email(
                to_email=str(user.email),
                subject=response_subject,
                content=ai_result['response']
            )
            
            if send_result['success']:
                email_log.processing_status = "completed"  # type: ignore  # SQLAlchemy column assignment
                logger.info("Response email processed successfully", user_id=user.id)
            else:
                email_log.processing_status = "failed"  # type: ignore  # SQLAlchemy column assignment
                email_log.error_message = f"Failed to send response: {send_result.get('error')}"  # type: ignore  # SQLAlchemy column assignment
        else:
            email_log.processing_status = "failed"  # type: ignore  # SQLAlchemy column assignment
            email_log.error_message = f"AI generation failed: {ai_result.get('error')}"  # type: ignore  # SQLAlchemy column assignment
        
    except Exception as e:
        logger.error("Error processing response request", error=str(e), user_id=user.id)
        email_log.processing_status = "failed"  # type: ignore  # SQLAlchemy column assignment
        email_log.error_message = str(e)  # type: ignore  # SQLAlchemy column assignment
        raise  # Re-raise to trigger rollback in main handler


def get_or_create_user(db: Session, email: str) -> User:
    """Get existing user or create new one."""
    user = db.query(User).filter(User.email == email).first()
    
    if not user:
        user = User(email=email)
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info("Created new user", email=email, user_id=user.id)
    
    return user 