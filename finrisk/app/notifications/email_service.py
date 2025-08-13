"""
Email notification service for FinRisk application.
Handles deployment notifications, ML training alerts, and system monitoring.
"""

import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class EmailNotificationService:
    """Email notification service for FinRisk alerts and notifications."""
    
    def __init__(self):
        self.smtp_server = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("EMAIL_SMTP_PORT", "587"))
        self.username = os.getenv("EMAIL_USERNAME", "rfondufe@gmail.com")
        self.password = os.getenv("EMAIL_PASSWORD", "")
        self.recipient = os.getenv("NOTIFICATION_EMAIL", "rfondufe@gmail.com")
        
    def send_email(
        self,
        subject: str,
        body: str,
        recipients: Optional[List[str]] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
        html_body: Optional[str] = None
    ) -> bool:
        """
        Send an email notification.
        
        Args:
            subject: Email subject
            body: Plain text email body
            recipients: List of email recipients (defaults to configured recipient)
            attachments: List of attachment dictionaries with 'filename' and 'content'
            html_body: HTML version of email body
            
        Returns:
            bool: True if email sent successfully, False otherwise
        """
        try:
            if not self.password:
                logger.warning("Email password not configured, skipping email notification")
                return False
                
            recipients = recipients or [self.recipient]
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.username
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[FinRisk] {subject}"
            
            # Add plain text body
            text_part = MIMEText(body, 'plain')
            msg.attach(text_part)
            
            # Add HTML body if provided
            if html_body:
                html_part = MIMEText(html_body, 'html')
                msg.attach(html_part)
            
            # Add attachments
            if attachments:
                for attachment in attachments:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment['content'])
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename= {attachment["filename"]}'
                    )
                    msg.attach(part)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
                
            logger.info(f"Email notification sent successfully to {recipients}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False
    
    def send_deployment_notification(
        self,
        environment: str,
        status: str,
        details: Dict[str, Any]
    ) -> bool:
        """
        Send deployment notification.
        
        Args:
            environment: Deployment environment (staging/production)
            status: Deployment status (success/failed)
            details: Additional deployment details
            
        Returns:
            bool: True if email sent successfully
        """
        subject = f"Deployment {status.title()} - {environment.title()}"
        
        body = f"""
FinRisk Deployment Notification

Environment: {environment.title()}
Status: {status.upper()}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

Details:
- Branch: {details.get('branch', 'N/A')}
- Commit: {details.get('commit', 'N/A')}
- Duration: {details.get('duration', 'N/A')}
- Services: {', '.join(details.get('services', []))}

{details.get('message', '')}

---
This is an automated notification from the FinRisk CI/CD pipeline.
        """
        
        html_body = f"""
        <html>
        <body>
            <h2>FinRisk Deployment Notification</h2>
            <table style="border-collapse: collapse; width: 100%;">
                <tr style="background-color: {'#d4edda' if status == 'success' else '#f8d7da'};">
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>Environment:</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{environment.title()}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>Status:</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd; color: {'green' if status == 'success' else 'red'}; font-weight: bold;">{status.upper()}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>Timestamp:</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>Branch:</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{details.get('branch', 'N/A')}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>Commit:</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{details.get('commit', 'N/A')}</td>
                </tr>
            </table>
            <p><strong>Message:</strong> {details.get('message', '')}</p>
            <hr>
            <p><em>This is an automated notification from the FinRisk CI/CD pipeline.</em></p>
        </body>
        </html>
        """
        
        return self.send_email(subject, body, html_body=html_body)
    
    def send_ml_training_notification(
        self,
        model_type: str,
        status: str,
        metrics: Dict[str, float],
        details: Dict[str, Any]
    ) -> bool:
        """
        Send ML training notification.
        
        Args:
            model_type: Type of model (credit-risk/fraud-detection)
            status: Training status (completed/failed)
            metrics: Model performance metrics
            details: Additional training details
            
        Returns:
            bool: True if email sent successfully
        """
        subject = f"ML Training {status.title()} - {model_type.replace('-', ' ').title()}"
        
        # Format metrics
        metrics_text = "\n".join([f"- {k}: {v:.4f}" for k, v in metrics.items()])
        
        body = f"""
FinRisk ML Training Notification

Model Type: {model_type.replace('-', ' ').title()}
Status: {status.upper()}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

Performance Metrics:
{metrics_text}

Training Details:
- Duration: {details.get('duration', 'N/A')}
- Data Size: {details.get('data_size', 'N/A')}
- Features: {details.get('feature_count', 'N/A')}
- Model Version: {details.get('model_version', 'N/A')}

{details.get('message', '')}

---
This is an automated notification from the FinRisk ML pipeline.
        """
        
        html_body = f"""
        <html>
        <body>
            <h2>FinRisk ML Training Notification</h2>
            <table style="border-collapse: collapse; width: 100%;">
                <tr style="background-color: {'#d4edda' if status == 'completed' else '#f8d7da'};">
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>Model Type:</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{model_type.replace('-', ' ').title()}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>Status:</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd; color: {'green' if status == 'completed' else 'red'}; font-weight: bold;">{status.upper()}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>Timestamp:</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</td>
                </tr>
            </table>
            
            <h3>Performance Metrics:</h3>
            <table style="border-collapse: collapse; width: 100%;">
                {''.join([f'<tr><td style="padding: 5px; border: 1px solid #ddd;">{k}</td><td style="padding: 5px; border: 1px solid #ddd;">{v:.4f}</td></tr>' for k, v in metrics.items()])}
            </table>
            
            <h3>Training Details:</h3>
            <ul>
                <li><strong>Duration:</strong> {details.get('duration', 'N/A')}</li>
                <li><strong>Data Size:</strong> {details.get('data_size', 'N/A')}</li>
                <li><strong>Features:</strong> {details.get('feature_count', 'N/A')}</li>
                <li><strong>Model Version:</strong> {details.get('model_version', 'N/A')}</li>
            </ul>
            
            <p><strong>Message:</strong> {details.get('message', '')}</p>
            <hr>
            <p><em>This is an automated notification from the FinRisk ML pipeline.</em></p>
        </body>
        </html>
        """
        
        return self.send_email(subject, body, html_body=html_body)
    
    def send_system_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        details: Dict[str, Any]
    ) -> bool:
        """
        Send system alert notification.
        
        Args:
            alert_type: Type of alert (performance/security/error)
            severity: Alert severity (low/medium/high/critical)
            message: Alert message
            details: Additional alert details
            
        Returns:
            bool: True if email sent successfully
        """
        subject = f"System Alert [{severity.upper()}] - {alert_type.title()}"
        
        body = f"""
FinRisk System Alert

Alert Type: {alert_type.title()}
Severity: {severity.upper()}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

Message: {message}

Details:
- Component: {details.get('component', 'N/A')}
- Error Code: {details.get('error_code', 'N/A')}
- Stack Trace: {details.get('stack_trace', 'N/A')}

---
This is an automated alert from the FinRisk monitoring system.
        """
        
        # Color coding for severity
        severity_colors = {
            'low': '#fff3cd',
            'medium': '#ffeaa7',
            'high': '#fdcb6e',
            'critical': '#e17055'
        }
        
        html_body = f"""
        <html>
        <body>
            <h2>FinRisk System Alert</h2>
            <table style="border-collapse: collapse; width: 100%;">
                <tr style="background-color: {severity_colors.get(severity.lower(), '#f8d7da')};">
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>Alert Type:</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{alert_type.title()}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>Severity:</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold; color: {'red' if severity.lower() in ['high', 'critical'] else 'orange' if severity.lower() == 'medium' else 'green'};">{severity.upper()}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>Timestamp:</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</td>
                </tr>
            </table>
            
            <h3>Message:</h3>
            <p>{message}</p>
            
            <h3>Details:</h3>
            <ul>
                <li><strong>Component:</strong> {details.get('component', 'N/A')}</li>
                <li><strong>Error Code:</strong> {details.get('error_code', 'N/A')}</li>
            </ul>
            
            {f'<h3>Stack Trace:</h3><pre style="background-color: #f8f9fa; padding: 10px; border-radius: 5px;">{details.get("stack_trace", "N/A")}</pre>' if details.get('stack_trace') else ''}
            
            <hr>
            <p><em>This is an automated alert from the FinRisk monitoring system.</em></p>
        </body>
        </html>
        """
        
        return self.send_email(subject, body, html_body=html_body)


# Global email service instance
email_service = EmailNotificationService()
