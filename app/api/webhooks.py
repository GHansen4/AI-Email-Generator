from fastapi import APIRouter, Request, HTTPException, Depends, status, Form
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime
from typing import Dict, Any, Optional
import json
import time

from app.database import (
    get_db, User, WritingProfile, EmailLog, SecurityEvent,
    get_user_by_email, create_user, get_or_create_user,
    log_email_processing, log_security_event
)
from app.services.ai_generator import AIResponseGenerator
from app.services.email_parser import EnhancedEmailParser
from app.services.mailgun_client import MailgunClient
from app.services.profile_analyzer import WritingProfileAnalyzer
from app.utils.logging import get_logger, LoggingTimer
from app.utils.security import SecurityManager, verify_webhook_signature, validate_email_request, sanitize_content
from app.config import settings

logger = get_logger(__name__)
router = APIRouter()

# Initialize services with enhanced error handling
try:
    profile_analyzer = WritingProfileAnalyzer()
    email_parser = EnhancedEmailParser()
    ai_generator = AIResponseGenerator()
    mailgun_client = MailgunClient()
    security_manager = SecurityManager()
    
    logger.info("All webhook services initialized successfully")
except Exception as e:
    logger.error("Error initializing webhook services", error=str(e))
    raise RuntimeError(f"Failed to initialize webhook services: {e}")


def validate_webhook_environment() -> Dict[str, bool]:
    """Validate that all required environment settings are available."""
    checks = {
        'database_url': bool(getattr(settings, 'database_url', None)),
        'mailgun_api_key': bool(getattr(settings, 'mailgun_api_key', None)),
        'mailgun_domain': bool(getattr(settings, 'mailgun_domain', None)),
        'service_email': bool(getattr(settings, 'service_email', None)),
    }
    
    # Log any missing configuration
    missing = [key for key, value in checks.items() if not value]
    if missing:
        logger.warning("Missing webhook configuration", missing_settings=missing)
    
    return checks


# Validate environment on module load
env_status = validate_webhook_environment()
if not all(env_status.values()):
    logger.warning("Webhook environment validation issues detected", env_status=env_status)


@router.post("/webhook/mailgun")
async def mailgun_webhook(
    request: Request,
    timestamp: str = Form(...),
    token: str = Form(...),
    signature: str = Form(...),
    sender: str = Form(...),
    recipient: str = Form(...),
    subject: str = Form(...),
    body_plain: str = Form(default=""),
    body_html: str = Form(default=""),
    message_id: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    """Handle incoming emails from Mailgun webhook with comprehensive security."""
    
    # Generate request ID for tracking
    try:
        request_id = security_manager.generate_request_id()
    except Exception as e:
        logger.error("Failed to generate request ID", error=str(e))
        request_id = f"webhook_{int(time.time())}"
    
    client_ip = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
    processing_start = time.time()
    
    logger.info("Webhook received", 
               request_id=request_id,
               sender=sender,
               subject=subject,
               message_id=message_id)

    # Security Layer 1: Webhook Signature Verification
    try:
        if not verify_webhook_signature(timestamp, token, signature):
            await log_security_event(
                db, "webhook_signature_invalid", "high",
                source_ip=client_ip, source_email=sender,
                event_data={"message_id": message_id, "timestamp": timestamp}
            )
            logger.error("Invalid webhook signature", 
                        request_id=request_id,
                        sender=sender,
                        security_event=True)
            raise HTTPException(status_code=401, detail="Invalid webhook signature")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Webhook verification error", error=str(e), request_id=request_id)
        raise HTTPException(status_code=401, detail="Webhook verification failed")

    # Security Layer 2: Email Validation and Rate Limiting
    email_content = body_plain or body_html
    try:
        is_valid, validation_error = validate_email_request(
            email=sender, 
            content=email_content, 
            identifier=client_ip
        )
        if not is_valid:
            await log_security_event(
                db, "email_validation_failed", "medium",
                source_ip=client_ip, source_email=sender,
                event_data={"error": validation_error, "message_id": message_id}
            )
            logger.warning("Email validation failed", 
                          sender=sender,
                          error=validation_error,
                          request_id=request_id)
            raise HTTPException(status_code=400, detail=validation_error)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Email validation error", error=str(e), request_id=request_id)
        raise HTTPException(status_code=400, detail="Email validation failed")

    # Security Layer 3: Content Sanitization
    try:
        if getattr(settings, 'content_sanitization_enabled', True):
            email_content = sanitize_content(email_content)
            subject = sanitize_content(subject)
    except Exception as e:
        logger.error("Content sanitization error", error=str(e), request_id=request_id)

    # Start main processing
    try:
        # Get or create user
        user = await get_or_create_user(db, sender)
        
        # Determine email type
        email_type = "profile" if subject.upper().startswith("PROFILE:") else "response_request"
        
        # Log email processing start
        email_log = await log_email_processing(
            db, sender, subject, email_type, "processing",
            processing_time_ms=0
        )
        
        # Process based on email type
        if email_type == "profile":
            result = await process_profile_email(
                db, user, email_content, subject, request_id
            )
        else:
            result = await process_response_request(
                db, user, email_content, subject, request_id
            )
        
        # Update email log with results
        processing_time_ms = (time.time() - processing_start) * 1000
        email_log.processing_status = "success" if result["success"] else "failed"
        
        # Handle optional fields properly
        error_msg = result.get("error")
        if error_msg:
            email_log.error_message = error_msg
            
        confidence_val = result.get("confidence_score")
        if confidence_val is not None:
            email_log.confidence_score = confidence_val
            
        email_log.processing_time_ms = processing_time_ms
        email_log.processed_at = datetime.utcnow()
        
        await db.commit()
        
        logger.info("Email processing completed", 
                   request_id=request_id,
                   sender=sender,
                   email_type=email_type,
                   success=result["success"],
                   processing_time_ms=processing_time_ms)
        
        return {
            "status": "success" if result["success"] else "error",
            "message": result.get("message", "Email processed"),
            "request_id": request_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # Log the error and attempt to update email log
        try:
            processing_time_ms = (time.time() - processing_start) * 1000
            await log_email_processing(
                db, sender, subject, "unknown", "failed",
                processing_time_ms=processing_time_ms,
                error_message=str(e),
                error_type=type(e).__name__
            )
            await db.commit()
        except Exception as log_error:
            logger.error("Failed to log error", error=str(log_error), request_id=request_id)
        
        await log_security_event(
            db, "webhook_processing_error", "high",
            source_ip=client_ip, source_email=sender,
            event_data={"error": str(e), "message_id": message_id}
        )
        
        logger.error("Webhook processing failed", 
                    error=str(e), 
                    request_id=request_id,
                    security_event=True)
        raise HTTPException(status_code=500, detail="Internal server error")


async def process_profile_email(
    db: AsyncSession, 
    user: User, 
    content: str, 
    subject: str,
    request_id: str
) -> Dict[str, Any]:
    """Process profile-building email with proper database integration."""
    try:
        logger.info("Processing profile email", user_id=user.id, request_id=request_id)
        
        # Input validation
        if not content or not content.strip():
            return {
                "success": False,
                "error": "Email content is empty or invalid"
            }
        
        # Security: Validate content size
        max_size = getattr(settings, 'max_email_size', 1024 * 1024)  # 1MB default
        if len(content) > max_size:
            return {
                "success": False,
                "error": f"Content too large (max {max_size} bytes)"
            }
        
        # Use the enhanced email parser to extract user's writing
        try:
            parsed_result = email_parser.parse_email(content, user.email, subject)
        except Exception as e:
            logger.error("Email parsing failed", user_id=user.id, error=str(e), request_id=request_id)
            return {
                "success": False,
                "error": f"Email parsing failed: {str(e)}"
            }
        
        if parsed_result.email_type != "profile" or not parsed_result.user_content:
            return {
                "success": False,
                "error": "Could not extract user writing content for profile analysis"
            }
        
        user_writing = parsed_result.user_content
        
        if len(user_writing.strip()) < 50:
            return {
                "success": False,
                "error": "Insufficient writing content for analysis (minimum 50 characters)"
            }
        
        # Analyze writing style
        try:
            with LoggingTimer(logger, "writing_analysis", user_id=user.id):
                analysis_result = profile_analyzer.analyze_writing_style(user_writing)
        except Exception as e:
            logger.error("Writing analysis failed", user_id=user.id, error=str(e), request_id=request_id)
            return {
                "success": False,
                "error": f"Writing analysis failed: {str(e)}"
            }
        
        confidence_score = analysis_result.get('confidence_score', 0.1)
        
        # Get existing writing profile
        result = await db.execute(
            select(WritingProfile).where(WritingProfile.user_id == user.id)
        )
        existing_profile = result.scalar_one_or_none()
        
        if existing_profile:
            # Decrypt existing profile data
            try:
                existing_data = json.loads(
                    security_manager.decrypt_sensitive_data(existing_profile.profile_data)
                )
                sample_count = existing_data.get('sample_count', 0)
            except Exception as e:
                logger.warning("Could not decrypt existing profile", 
                             user_id=user.id, error=str(e))
                existing_data = profile_analyzer._get_default_profile()
                sample_count = 0
            
            # Merge with existing profile
            merged_profile = profile_analyzer.merge_profiles(
                existing_data, analysis_result, sample_count
            )
            merged_profile['sample_count'] = sample_count + 1
            merged_profile['confidence_score'] = min(1.0, merged_profile['sample_count'] / 10.0)
            
            # Update existing profile
            existing_profile.profile_data = security_manager.encrypt_sensitive_data(
                json.dumps(merged_profile)
            )
            existing_profile.fingerprint_data = security_manager.encrypt_sensitive_data(
                json.dumps(merged_profile.get('comprehensive_fingerprint', {}))
            )
            existing_profile.version += 1
            existing_profile.updated_at = datetime.utcnow()
            
            final_profile = merged_profile
            
        else:
            # Create new profile
            analysis_result['sample_count'] = 1
            analysis_result['confidence_score'] = 0.1
            
            new_profile = WritingProfile(
                user_id=user.id,
                profile_data=security_manager.encrypt_sensitive_data(
                    json.dumps(analysis_result)
                ),
                fingerprint_data=security_manager.encrypt_sensitive_data(
                    json.dumps(analysis_result.get('comprehensive_fingerprint', {}))
                ),
                version=1
            )
            db.add(new_profile)
            
            final_profile = analysis_result
        
        # Update user statistics
        user.sample_count = final_profile['sample_count']
        user.confidence_score = final_profile['confidence_score']
        user.last_profile_update = datetime.utcnow()
        
        # Send confirmation email
        try:
            confirmation_result = await mailgun_client.send_profile_confirmation(
                user.email, 
                final_profile['sample_count'],
                final_profile['confidence_score']
            )
            
            if not confirmation_result['success']:
                logger.warning("Failed to send profile confirmation", 
                             user_id=user.id,
                             error=confirmation_result.get('error'))
        except Exception as e:
            logger.error("Error sending profile confirmation", 
                        user_id=user.id, error=str(e))
        
        logger.info("Profile updated successfully", 
                   user_id=user.id,
                   sample_count=final_profile['sample_count'],
                   confidence_score=final_profile['confidence_score'],
                   parsing_confidence=parsed_result.confidence_score)
        
        return {
            "success": True,
            "message": "Profile updated successfully",
            "confidence_score": final_profile['confidence_score'],
            "sample_count": final_profile['sample_count']
        }
        
    except Exception as e:
        logger.error("Error processing profile email", 
                    user_id=user.id, error=str(e), request_id=request_id)
        return {
            "success": False,
            "error": f"Profile processing failed: {str(e)}"
        }


async def process_response_request(
    db: AsyncSession, 
    user: User, 
    content: str, 
    subject: str,
    request_id: str
) -> Dict[str, Any]:
    """Process request for AI-generated email response."""
    try:
        logger.info("Processing response request", user_id=user.id, request_id=request_id)
        
        # Security: Validate content size
        max_size = getattr(settings, 'max_email_size', 1024 * 1024)  # 1MB default
        if len(content) > max_size:
            return {
                "success": False,
                "error": f"Content too large (max {max_size} bytes)"
            }
        
        # Use the enhanced email parser to extract original email
        parsed_result = email_parser.parse_email(content, user.email, subject)
        
        if parsed_result.email_type != "response_request" or not parsed_result.original_content:
            return {
                "success": False,
                "error": "Could not parse forwarded email content"
            }
        
        original_content = parsed_result.original_content
        
        # Get user's writing profile
        result = await db.execute(
            select(WritingProfile).where(WritingProfile.user_id == user.id)
        )
        writing_profile = result.scalar_one_or_none()
        
        if writing_profile:
            try:
                # Decrypt profile data
                profile_data = json.loads(
                    security_manager.decrypt_sensitive_data(writing_profile.profile_data)
                )
                confidence_score = profile_data.get('confidence_score', 0.1)
            except Exception as e:
                logger.warning("Could not decrypt profile data", 
                             user_id=user.id, error=str(e))
                profile_data = profile_analyzer._get_default_profile()
                confidence_score = 0.1
        else:
            logger.info("No writing profile found, using default", user_id=user.id)
            profile_data = profile_analyzer._get_default_profile()
            confidence_score = 0.1
        
        # Generate AI response using the AI generator
        with LoggingTimer(logger, "ai_response_generation", user_id=user.id):
            ai_response_result = ai_generator.generate_response(
                original_email=original_content,
                user_profile=profile_data,
                context=f"Original sender: {parsed_result.original_sender or 'Unknown'}"
            )
        
        if not ai_response_result.get('success', False):
            return {
                "success": False,
                "error": f"AI response generation failed: {ai_response_result.get('error', 'Unknown error')}"
            }
            
        ai_response = ai_response_result.get('response', '')
        
        if not ai_response or len(ai_response.strip()) < 10:
            return {
                "success": False,
                "error": "AI response generation failed or produced insufficient content"
            }
        
        # Send response draft to user
        try:
            send_result = await mailgun_client.send_ai_response_draft(
                user.email,
                parsed_result.original_subject or "No Subject",
                ai_response,
                original_content[:200] + "..." if len(original_content) > 200 else original_content,
                confidence_score
            )
            
            if not send_result['success']:
                return {
                    "success": False,
                    "error": f"Failed to send response draft: {send_result.get('error')}"
                }
        except Exception as e:
            logger.error("Error sending response draft", 
                        user_id=user.id, error=str(e))
            return {
                "success": False,
                "error": f"Failed to send response draft: {str(e)}"
            }
        
        logger.info("Response request processed successfully", 
                   user_id=user.id,
                   confidence_score=confidence_score,
                   parsing_confidence=parsed_result.confidence_score,
                   response_length=len(ai_response))
        
        return {
            "success": True,
            "message": "AI response draft sent successfully",
            "confidence_score": confidence_score
        }
        
    except Exception as e:
        logger.error("Error processing response request", 
                    user_id=user.id, error=str(e), request_id=request_id)
        return {
            "success": False,
            "error": f"Response processing failed: {str(e)}"
        } 