import boto3
import logging
from botocore.exceptions import ClientError, NoCredentialsError

from app.config import BEDROCK_REGION, AWS_PROFILE, ASSUME_ROLE_ARN

logger = logging.getLogger(__name__)

def get_bedrock_client():
    """
    Get Bedrock client - works both locally and in ECS
    Priority:
    1. ECS Task Role (automatic in ECS)
    2. Local development with profile + assume role
    3. Local development with profile only
    """
    try:
        # Check if we're running in ECS (no profile specified)
        if not AWS_PROFILE and not ASSUME_ROLE_ARN:
            logger.info("🚀 Using ECS Task Role for Bedrock access")
            return boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
        
        # Local development with assume role
        elif AWS_PROFILE and ASSUME_ROLE_ARN:
            logger.info(f"🔑 Using AWS Profile '{AWS_PROFILE}' with assume role for local development")
            session = boto3.Session(profile_name=AWS_PROFILE)
            sts = session.client("sts")
            
            response = sts.assume_role(
                RoleArn=ASSUME_ROLE_ARN,
                RoleSessionName="RAGServiceLocal"
            )
            
            credentials = response['Credentials']
            
            return boto3.client(
                "bedrock-runtime",
                region_name=BEDROCK_REGION,
                aws_access_key_id=credentials['AccessKeyId'],
                aws_secret_access_key=credentials['SecretAccessKey'],
                aws_session_token=credentials['SessionToken']
            )
        
        # Local development with profile only
        elif AWS_PROFILE:
            logger.info(f"🔑 Using AWS Profile '{AWS_PROFILE}' for local development")
            session = boto3.Session(profile_name=AWS_PROFILE)
            return session.client("bedrock-runtime", region_name=BEDROCK_REGION)
        
        # Default boto3 behavior (environment variables, instance profile, etc.)
        else:
            logger.info("🔑 Using default AWS credentials")
            return boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
            
    except ClientError as e:
        error_msg = f"AWS ClientError: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    except NoCredentialsError as e:
        error_msg = f"No AWS credentials found: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error creating Bedrock client: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

def test_bedrock_connection():
    """Test Bedrock connection"""
    try:
        client = get_bedrock_client()
        # Simple test - just creating client is enough for now
        logger.info("✅ Bedrock client created successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Bedrock connection test failed: {e}")
        return False