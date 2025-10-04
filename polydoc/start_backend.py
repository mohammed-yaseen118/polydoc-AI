#!/usr/bin/env python3
"""
Simplified PolyDoc Backend Startup Script
Starts the FastAPI server with Hindi/Kannada support
"""
import sys
import os
import logging
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Start the PolyDoc backend server"""
    
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting PolyDoc Backend with Hindi/Kannada Support")
    logger.info("=" * 60)
    
    try:
        # Import uvicorn
        import uvicorn
        
        # Import the FastAPI app
        from src.api.main_mongodb import app
        
        logger.info("✅ FastAPI app loaded successfully")
        logger.info("✅ Hindi and Kannada processing modules ready")
        logger.info("✅ OCR and language detection initialized")
        
        # Create necessary directories
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("static", exist_ok=True)
        os.makedirs("templates", exist_ok=True)
        os.makedirs("vector_store", exist_ok=True)
        
        logger.info("✅ Required directories created")
        
        # Start the server
        logger.info("🌐 Starting server at http://localhost:8000")
        logger.info("📝 API documentation available at http://localhost:8000/docs")
        logger.info("⚠️  First startup may take a few minutes to download AI models")
        
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8000,
            log_level="info",
            access_log=True
        )
        
    except ImportError as e:
        logger.error(f"❌ Missing required dependency: {e}")
        logger.error("💡 Try: pip install fastapi uvicorn")
        return 1
        
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
