#!/usr/bin/env python3
"""
Test script to verify Bedrock connection locally
"""
import os
import json
import boto3
from dotenv import load_dotenv

load_dotenv()

def test_bedrock_connection():
    """Test Bedrock connection with your existing setup"""
    
    BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID")
    BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")
    AWS_PROFILE = os.getenv("AWS_PROFILE")
    ASSUME_ROLE_ARN = os.getenv("ASSUME_ROLE_ARN")
    
    print(f"🔧 Configuration:")
    print(f"   Model ID: {BEDROCK_MODEL_ID}")
    print(f"   Region: {BEDROCK_REGION}")
    print(f"   Profile: {AWS_PROFILE}")
    print(f"   Assume Role: {ASSUME_ROLE_ARN}")
    print()
    
    try:
        # Step 1: Create session with profile
        print("🔑 Creating AWS session...")
        session = boto3.Session(profile_name=AWS_PROFILE)
        
        # Verify identity
        sts = session.client("sts")
        identity = sts.get_caller_identity()
        print(f"✅ Current identity: {identity['Arn']}")
        
        # Step 2: Assume role if specified
        if ASSUME_ROLE_ARN:
            print(f"🔄 Assuming role: {ASSUME_ROLE_ARN}")
            response = sts.assume_role(
                RoleArn=ASSUME_ROLE_ARN,
                RoleSessionName="RAGServiceTest"
            )
            
            credentials = response['Credentials']
            
            # Create Bedrock client with assumed role
            bedrock_client = boto3.client(
                "bedrock-runtime",
                region_name=BEDROCK_REGION,
                aws_access_key_id=credentials['AccessKeyId'],
                aws_secret_access_key=credentials['SecretAccessKey'],
                aws_session_token=credentials['SessionToken']
            )
            print("✅ Bedrock client created with assumed role")
        else:
            # Create Bedrock client with profile
            bedrock_client = session.client("bedrock-runtime", region_name=BEDROCK_REGION)
            print("✅ Bedrock client created with profile")
        
        # Step 3: Test model invocation
        print("🤖 Testing model invocation...")
        
        test_prompt = "Hello, this is a test. Please respond briefly."
        
        payload = {
            "prompt": json.dumps({"messages": [{"role": "user", "content": test_prompt}]}),
            "max_tokens": 50,
            "temperature": 0.5
        }
        
        response = bedrock_client.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps(payload),
            contentType="application/json"
        )
        
        result = json.loads(response["body"].read())
        answer = result["outputs"][0]["text"].strip()
        
        print("✅ Model invocation successful!")
        print(f"📝 Response: {answer}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_database_connection():
    """Test database connection"""
    print("\n🗄️  Testing Database Connection...")
    
    DB_HOST = os.getenv("DB_HOST")
    DB_NAME = os.getenv("DB_NAME")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    
    try:
        import psycopg2
        
        connection = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        
        cursor.close()
        connection.close()
        
        if result[0] == 1:
            print("✅ Database connection successful!")
            return True
        else:
            print("❌ Database connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 RAG Service Connection Test\n")
    
    # Test Bedrock
    bedrock_ok = test_bedrock_connection()
    
    # Test Database
    db_ok = test_database_connection()
    
    print(f"\n📊 Test Results:")
    print(f"   Bedrock: {'✅ OK' if bedrock_ok else '❌ FAIL'}")
    print(f"   Database: {'✅ OK' if db_ok else '❌ FAIL'}")
    
    if bedrock_ok and db_ok:
        print("\n🎉 All tests passed! Your setup is ready.")
    else:
        print("\n⚠️  Some tests failed. Check your configuration.")