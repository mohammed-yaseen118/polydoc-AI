#!/usr/bin/env python3
"""
PolyDoc - Main Application Entry Point
Multi-lingual Document Understanding and Layout Preservation System
"""

import os
import sys
import logging
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import the FastAPI application
from src.api.main_mongodb import app

def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('polydoc_ai.log')
        ]
    )

def main():
    """Main application entry point"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting PolyDoc - Multi-lingual Document Understanding System")
    logger.info("=" * 70)
    logger.info("Features:")
    logger.info("• Multi-format document processing (PDF, DOCX, PPTX, Images)")
    logger.info("• Multi-language OCR and text extraction")
    logger.info("• AI-powered document summarization")
    logger.info("• Real-time chat interface with documents")
    logger.info("• Vector-based semantic search")
    logger.info("• Layout preservation and analysis")
    logger.info("• Free and open-source AI models")
    logger.info("=" * 70)
    
    # Create necessary directories
    directories = ['uploads', 'vector_store', 'static', 'templates']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"✓ Directory '{directory}' ready")
    
    logger.info("System ready! Starting web server...")
    logger.info("Access the application at: http://localhost:8000")
    
    # Import and run uvicorn
    import uvicorn
    
    try:
        uvicorn.run(
            "src.api.main_mongodb:app",  # Use string import instead of app object
            host="127.0.0.1",
            port=8000,
            log_level="info",
            access_log=True,
            reload=False  # Disable reload to avoid issues
        )
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
