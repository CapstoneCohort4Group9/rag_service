import boto3
from botocore.exceptions import BotoCoreError, ClientError
from app.config import settings


def get_bedrock_client_with_sts():
    """
    Get Bedrock client using STS assume role (matching your team's pattern)
    """
    try:
        # Priority: Use AWS_PROFILE if defined, otherwise default profile or env
        profile_name = settings.AWS_PROFILE if hasattr(settings, 'AWS_PROFILE') and settings.AWS_PROFILE else None

        session = boto3.Session(profile_name=profile_name)

        # Optional: validate session identity
        try:
            caller = session.client("sts").get_caller_identity()
            print(f"Using credentials for: {caller['Arn']}")
        except Exception as e:
            print(f"Warning: Could not get caller identity: {e}")

        # Step 1: STS AssumeRole using the session
        sts = session.client("sts")

        response = sts.assume_role(
            RoleArn=settings.ASSUME_ROLE_ARN,
            RoleSessionName="RAGServiceBedrockSession"
        )

        credentials = response['Credentials']

        # Step 2: Create a Bedrock Runtime client with temporary session credentials
        bedrock = boto3.client(
            "bedrock-runtime",
            region_name=settings.BEDROCK_REGION,
            aws_access_key_id=credentials['AccessKeyId'],
            aws_secret_access_key=credentials['SecretAccessKey'],
            aws_session_token=credentials['SessionToken']
        )

        return bedrock

    except ClientError as e:
        raise RuntimeError(f"AWS ClientError: {e}")
    except BotoCoreError as e:
        raise RuntimeError(f"STS AssumeRole failed: {e}")


def get_embeddings_client():
    """Get embeddings client (for future use if needed)"""
    try:
        profile_name = settings.AWS_PROFILE if hasattr(settings, 'AWS_PROFILE') and settings.AWS_PROFILE else None
        session = boto3.Session(profile_name=profile_name)
        
        # Create Bedrock client for embeddings
        bedrock = session.client(
            "bedrock-runtime",
            region_name=settings.BEDROCK_REGION
        )
        
        return bedrock
    except Exception as e:
        raise RuntimeError(f"Failed to create embeddings client: {e}")