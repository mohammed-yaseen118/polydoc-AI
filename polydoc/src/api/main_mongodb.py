"""
PolyDoc AI - FastAPI Backend with MongoDB Integration
Provides RESTful API and WebSocket endpoints for document processing and chat
"""

import asyncio
import logging
import uuid
import time
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from pydantic import BaseModel
import uvicorn

# Import our custom modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.document_processor import DocumentProcessor, ProcessedDocument
from src.core.mongodb_store import MongoDBStore, get_mongodb_store
from src.models.ai_models import AIModelManager, DocumentAnalyzer
from src.utils.indian_language_detector import IndianLanguageDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class ChatMessage(BaseModel):
    message: str
    user_id: str  # Added for MongoDB
    document_id: Optional[str] = None
    language: Optional[str] = 'en'

class DocumentUploadRequest(BaseModel):
    user_id: str

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    summary: Optional[str] = None
    statistics: Dict[str, Any]
    processing_time: float

class ChatResponse(BaseModel):
    response: str
    confidence: float
    sources: List[int]  # Page numbers
    processing_time: float
    timestamp: float

class DocumentListResponse(BaseModel):
    documents: List[Dict[str, Any]]
    total_count: int

# Global instances
app = FastAPI(title="PolyDoc AI with MongoDB", version="2.0.0")
ai_models: Optional[AIModelManager] = None
document_processor: Optional[DocumentProcessor] = None
# Store per-user MongoDB instances
user_mongodb_stores: Dict[str, MongoDBStore] = {}
document_analyzer: Optional[DocumentAnalyzer] = None

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")
    
    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)

manager = ConnectionManager()

# CORS configuration - Fixed for proper OPTIONS handling
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002", 
        "http://localhost:3003", 
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3002",
        "http://127.0.0.1:3003",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*"  # Allow all origins for development
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"]  # Expose all headers
)

# Global initialization status
initialization_status = {
    "status": "initializing",
    "progress": 0,
    "message": "Starting initialization...",
    "models_ready": False,
    "mongodb_ready": False,
    "error": None
}

# Background model initialization
async def get_user_mongodb_store(user_id: str) -> MongoDBStore:
    """Get or create user-specific MongoDB store"""
    global user_mongodb_stores, ai_models
    
    if not ai_models:
        raise HTTPException(status_code=500, detail="AI models not initialized")
    
    if user_id not in user_mongodb_stores:
        user_mongodb_stores[user_id] = MongoDBStore(ai_models, user_id=user_id)
        await user_mongodb_stores[user_id].connect()
        logger.info(f"Created MongoDB store for user: {user_id}")
    
    return user_mongodb_stores[user_id]

async def initialize_models_background():
    """Initialize AI models in the background with progress updates"""
    global ai_models, document_processor, document_analyzer, initialization_status
    
    try:
        initialization_status["message"] = "Starting AI model initialization..."
        initialization_status["progress"] = 5
        
        logger.info("🚀 Starting optimized AI model initialization...")
        
        # Initialize AI models with optimized loading
        try:
            initialization_status["message"] = "Loading multilingual AI models..."
            initialization_status["progress"] = 15
            logger.info("🤖 Initializing optimized AI Model Manager...")
            
            # Check available disk space and optimize accordingly
            import shutil
            cache_dir = os.path.expanduser("~/.cache/huggingface")
            if os.path.exists(cache_dir):
                free_space = shutil.disk_usage(cache_dir).free / (1024**3)  # GB
                logger.info(f"💾 Available disk space for models: {free_space:.2f} GB")
                
                if free_space < 3.0:
                    logger.warning("⚠️ Limited disk space - optimizing model loading")
                    os.environ['HF_HUB_CACHE'] = cache_dir
            
            # Load models with progress updates
            initialization_status["message"] = "Loading embedding models (1/4)..."
            initialization_status["progress"] = 25
            ai_models = AIModelManager()
            logger.info("✅ AI Model Manager initialized with optimizations")
        except Exception as e:
            logger.error(f"Failed to initialize AI models: {e}")
            # Continue with limited functionality
            ai_models = None
            logger.info("Continuing with document processing only (no AI features)")
        
        try:
            initialization_status["message"] = "Loading OCR engines (2/4)..."
            initialization_status["progress"] = 50
            logger.info("📝 Initializing Document Processor with Hindi/Kannada OCR...")
            document_processor = DocumentProcessor()
            logger.info("✅ Document Processor with multilingual OCR ready")
        except Exception as e:
            logger.error(f"Failed to initialize document processor: {e}")
            document_processor = None
        
        try:
            initialization_status["message"] = "Setting up database (3/4)..."
            initialization_status["progress"] = 75
            logger.info("🍃 Setting up MongoDB integration...")
            # MongoDB stores will be created per-user as needed
            initialization_status["mongodb_ready"] = True
            logger.info("✅ MongoDB integration ready")
        except Exception as e:
            logger.error(f"Failed to setup MongoDB: {e}")
            raise
        
        try:
            initialization_status["message"] = "Finalizing AI systems (4/4)..."
            initialization_status["progress"] = 90
            logger.info("🗘️ Initializing Document Analyzer...")
            document_analyzer = DocumentAnalyzer(ai_models)
            logger.info("✅ Document Analyzer ready for multilingual processing")
        except Exception as e:
            logger.error(f"Failed to initialize document analyzer: {e}")
            raise
        
        initialization_status["status"] = "ready"
        initialization_status["progress"] = 100
        initialization_status["message"] = "All systems ready! Upload your documents now."
        initialization_status["models_ready"] = True
        
        logger.info("🎉 PolyDoc AI fully loaded - Hindi/Kannada processing ready!")
        
    except Exception as e:
        logger.error(f"Failed to initialize PolyDoc AI: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception details: {str(e)}")
        
        initialization_status["status"] = "error"
        initialization_status["error"] = str(e)
        initialization_status["message"] = f"Initialization failed: {str(e)}"
        
        # Don't let the exception crash the server
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")

# Startup event - non-blocking
@app.on_event("startup")
async def startup_event():
    """Start the application with background model loading"""
    logger.info("Starting PolyDoc AI web server with MongoDB...")
    logger.info("Starting background AI model initialization...")

    # Ensure uploads and static dirs exist
    for d in ["uploads", "static", "templates"]:
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass
    
    # Start model initialization in background
    asyncio.create_task(initialize_models_background())
    
    # Don't await the task - let it run in background
    logger.info("Web server is ready, AI models and MongoDB loading in background...")

# Add explicit OPTIONS handler for all routes
@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    """Handle all OPTIONS requests"""
    return JSONResponse(
        content={"message": "OK"}, 
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "86400"
        }
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "models_loaded": ai_models is not None,
        "mongodb_ready": initialization_status.get("mongodb_ready", False),
        "initialization": initialization_status,
        "model_info": ai_models.get_model_info() if ai_models else {}
    }

# Initialization status endpoint
@app.get("/initialization-status")
async def get_initialization_status():
    """Get current initialization status"""
    return initialization_status

# Document upload endpoint
@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Header(..., alias="user-id", description="Firebase User ID")
):
    """Upload and process a document"""
    start_time = time.time()
    
    try:
        # Log upload attempt
        logger.info(f"Upload request from user {user_id}: {file.filename} ({file.content_type})")
        
        # Validate user_id
        if not user_id or user_id.strip() == "":
            raise HTTPException(status_code=400, detail="User ID is required")
        
        # Check system status
        if not all([ai_models, document_processor, document_analyzer]):
            raise HTTPException(status_code=503, detail="System is still initializing. Please wait a moment and try again.")
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Validate file type
        supported_extensions = {'.pdf', '.docx', '.pptx', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in supported_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_extension}. Supported types: {', '.join(sorted(supported_extensions))}"
            )
        
        # Validate file size (10MB limit)
        file_content = await file.read()
        if len(file_content) > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(
                status_code=400,
                detail="File size must be less than 10MB"
            )
        
        # Create unique document ID
        document_id = str(uuid.uuid4())
        
        # Save uploaded file
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / f"{document_id}_{file.filename}"
        
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)  # Use already read content
        
        # Get estimated processing time
        time_estimate = document_processor.estimate_processing_time(str(file_path))
        logger.info(f"Estimated processing time: {time_estimate['estimated_seconds']}s for {file.filename}")
        
        # Process document
        logger.info(f"Processing document {document_id}: {file.filename}")
        processed_doc = await document_processor.process_document(str(file_path))
        
        # Get user-specific MongoDB store
        user_store = await get_user_mongodb_store(user_id)
        
        # Add to MongoDB (extract email from user_id if it looks like an email)
        user_email = user_id if '@' in user_id else None
        chunks_added = await user_store.add_document(
            document_id=document_id,
            user_id=user_id,
            filename=file.filename,
            elements=processed_doc.elements,
            user_email=user_email
        )
        
        # Generate summary - handle both dict and string returns
        summary_result = await document_analyzer.generate_document_summary(
            processed_doc.elements, 
            summary_length='medium'
        )
        
        # Extract properly formatted summary text
        if isinstance(summary_result, dict):
            # Use English summary if available, otherwise use original
            summary = summary_result.get('english_summary', summary_result.get('summary', 'Summary generation failed'))
        else:
            summary = str(summary_result) if summary_result else "Summary generation failed"
        
        # Get document statistics
        stats = document_processor.get_document_stats(processed_doc)
        stats['chunks_added'] = chunks_added
        
        processing_time = time.time() - start_time
        
        logger.info(f"Document {document_id} processed successfully in {processing_time:.2f}s")
        
        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            status="processed",
            summary=summary,
            statistics=stats,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Document list endpoint
@app.get("/documents", response_model=DocumentListResponse)
async def list_documents(user_id: str = Header(..., alias="user-id", description="Firebase User ID")):
    """Get list of processed documents for user"""
    try:
        if not ai_models:
            return DocumentListResponse(documents=[], total_count=0)
        
        # Get user-specific MongoDB store
        user_store = await get_user_mongodb_store(user_id)
        documents = await user_store.list_user_documents(user_id)
        
        return DocumentListResponse(
            documents=documents,
            total_count=len(documents)
        )
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Chat endpoint (REST API)
@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Process chat message and return response"""
    start_time = time.time()
    
    try:
        if not ai_models:
            raise HTTPException(status_code=500, detail="AI models not initialized")
        
        # Get user-specific MongoDB store
        user_store = await get_user_mongodb_store(message.user_id)
        
        # Debug logging
        logger.info(f"Processing chat request: user_id={message.user_id}, document_id={message.document_id}, message='{message.message[:100]}...'")
        
        # Get relevant context from MongoDB
        context, page_numbers = await user_store.get_context_for_question(
            question=message.message,
            user_id=message.user_id,
            document_id=message.document_id
        )
        
        logger.info(f"Context retrieved: {len(context)} characters, {len(page_numbers)} pages")
        
        if not context:
            # More informative error message with debugging info
            error_msg = "I couldn't find relevant information in the documents to answer your question."
            
            # Check if user has any documents at all
            try:
                user_docs = await user_store.list_user_documents(message.user_id)
                if not user_docs:
                    error_msg += "\n\nIt looks like you haven't uploaded any documents yet. Please upload a document first."
                else:
                    error_msg += f"\n\nI found {len(user_docs)} document(s) in your account, but couldn't extract relevant content for your question. Try rephrasing your question or make sure the document contains the information you're looking for."
            except Exception as doc_check_error:
                logger.error(f"Error checking user documents: {doc_check_error}")
                error_msg += "\n\nThere seems to be an issue accessing your documents. Please try again or contact support."
            
            return ChatResponse(
                response=error_msg,
                confidence=0.0,
                sources=[],
                processing_time=time.time() - start_time,
                timestamp=time.time()
            )
        
        # Generate response using AI model
        logger.info(f"Generating AI response for question with {len(context)} chars of context")
        try:
            response = await ai_models.answer_question(
                question=message.message,
                context=context,
                language=message.language or 'en'
            )
            logger.info(f"AI response generated successfully: {len(response.content)} chars, confidence: {response.confidence}")
        except Exception as ai_error:
            logger.error(f"AI model failed: {ai_error}")
            # Fallback response
            return ChatResponse(
                response=f"I found relevant information but encountered an error processing your question: {str(ai_error)}\n\nHere's what I found:\n\n{context[:500]}...",
                confidence=0.3,
                sources=page_numbers,
                processing_time=time.time() - start_time,
                timestamp=time.time()
            )
        
        # Format response with page references
        formatted_response = response.content
        if page_numbers:
            formatted_response += f"\n\n📄 Sources: Page(s) {', '.join(map(str, page_numbers))}"
        
        logger.info(f"Chat response completed successfully")
        return ChatResponse(
            response=formatted_response,
            confidence=response.confidence,
            sources=page_numbers,
            processing_time=time.time() - start_time,
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time chat
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time chat"""
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Send processing status
            await manager.send_message(client_id, {
                "type": "processing",
                "message": "Processing your question..."
            })
            
            try:
                # Process the message
                chat_message = ChatMessage(**message_data)
                
                # Get context and generate response
                start_time = time.time()
                
                # Get user-specific MongoDB store
                user_store = await get_user_mongodb_store(chat_message.user_id)
                
                logger.info(f"WebSocket processing: user_id={chat_message.user_id}, message='{chat_message.message[:100]}...'")
                
                context, page_numbers = await user_store.get_context_for_question(
                    question=chat_message.message,
                    user_id=chat_message.user_id,
                    document_id=chat_message.document_id
                )
                
                logger.info(f"WebSocket context retrieved: {len(context)} characters, {len(page_numbers)} pages")
                
                if not context:
                    # More informative error message
                    error_response = "I couldn't find relevant information to answer your question."
                    
                    try:
                        user_docs = await user_store.list_user_documents(chat_message.user_id)
                        if not user_docs:
                            error_response += "\n\nNo documents found. Please upload a document first."
                        else:
                            error_response += f"\n\nFound {len(user_docs)} document(s) but couldn't extract relevant content. Try rephrasing your question."
                    except Exception:
                        pass
                    
                    response_msg = {
                        "type": "response",
                        "response": error_response,
                        "confidence": 0.0,
                        "sources": [],
                        "processing_time": time.time() - start_time,
                        "timestamp": time.time()
                    }
                else:
                    # Generate AI response
                    logger.info(f"WebSocket generating AI response with {len(context)} chars of context")
                    try:
                        response = await ai_models.answer_question(
                            question=chat_message.message,
                            context=context,
                            language=chat_message.language or 'en'
                        )
                        logger.info(f"WebSocket AI response generated: {len(response.content)} chars, confidence: {response.confidence}")
                        
                        formatted_response = response.content
                        if page_numbers:
                            formatted_response += f"\n\n📄 Sources: Page(s) {', '.join(map(str, page_numbers))}"
                        
                        response_msg = {
                            "type": "response",
                            "response": formatted_response,
                            "confidence": response.confidence,
                            "sources": page_numbers,
                            "processing_time": time.time() - start_time,
                            "timestamp": time.time()
                        }
                        
                    except Exception as ai_error:
                        logger.error(f"WebSocket AI model failed: {ai_error}")
                        # Fallback response
                        response_msg = {
                            "type": "response",
                            "response": f"I found relevant information but encountered an error: {str(ai_error)}\n\nHere's what I found:\n\n{context[:500]}...",
                            "confidence": 0.3,
                            "sources": page_numbers,
                            "processing_time": time.time() - start_time,
                            "timestamp": time.time()
                        }
                
                await manager.send_message(client_id, response_msg)
                
            except Exception as e:
                # Send error message
                await manager.send_message(client_id, {
                    "type": "error",
                    "message": f"Error processing message: {str(e)}"
                })
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        manager.disconnect(client_id)

# Delete document endpoint
@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    user_id: str = Header(..., alias="user-id", description="Firebase User ID")
):
    """Delete a document from the system"""
    try:
        if not ai_models:
            raise HTTPException(status_code=500, detail="MongoDB not initialized")
        
        # Get user-specific MongoDB store
        user_store = await get_user_mongodb_store(user_id)
        success = await user_store.delete_document(document_id, user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {"message": f"Document {document_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Search endpoint
@app.get("/search")
async def search_documents(
    query: str,
    user_id: str = Header(..., alias="user-id", description="Firebase User ID"),
    document_id: Optional[str] = None,
    top_k: int = 5
):
    """Search for relevant content across user's documents"""
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if not ai_models:
            raise HTTPException(status_code=500, detail="MongoDB not initialized")
        
        # Get user-specific MongoDB store
        user_store = await get_user_mongodb_store(user_id)
        
        results = await user_store.search(
            query=query,
            user_id=user_id,
            top_k=top_k,
            document_id=document_id
        )
        
        search_results = []
        for result in results:
            search_results.append({
                "document_id": result.chunk.document_id,
                "page_number": result.chunk.page_number,
                "text": result.chunk.text,
                "score": result.score,
                "relevance": result.relevance,
                "element_type": result.chunk.element_type,
                "language": result.chunk.language
            })
        
        return {
            "query": query,
            "results": search_results,
            "total_results": len(search_results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Statistics endpoint
@app.get("/stats")
async def get_system_statistics():
    """Get system statistics"""
    try:
        stats = {
            "system_status": "operational",
            "timestamp": time.time(),
            "active_websocket_connections": len(manager.active_connections),
        }
        
        # Get aggregated MongoDB stats from all user stores
        if user_mongodb_stores:
            total_docs = 0
            total_chunks = 0
            for user_store in user_mongodb_stores.values():
                try:
                    user_stats = await user_store.get_statistics()
                    total_docs += user_stats.get('total_documents', 0)
                    total_chunks += user_stats.get('total_chunks', 0)
                except Exception:
                    continue
            stats["mongodb"] = {
                "total_documents": total_docs,
                "total_chunks": total_chunks,
                "total_users": len(user_mongodb_stores)
            }
        
        if ai_models:
            stats["ai_models"] = ai_models.get_model_info()
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# MongoDB data viewer endpoint
@app.get("/admin/mongodb-data")
async def view_mongodb_data():
    """View MongoDB data (admin endpoint)"""
    try:
        if not user_mongodb_stores:
            raise HTTPException(status_code=500, detail="MongoDB not initialized")
        
        # Use the first available user store for admin purposes
        if user_mongodb_stores:
            first_store = next(iter(user_mongodb_stores.values()))
            stats = await first_store.get_statistics()
            
            return {
                "message": "MongoDB data accessible",
                "statistics": stats,
                "total_users": len(user_mongodb_stores),
                "connection_info": {
                    "database": first_store.db.name,
                    "collections": [
                        first_store.documents_collection.name,
                        first_store.chunks_collection.name,
                        first_store.chat_sessions_collection.name,
                        first_store.users_collection.name
                    ]
                },
                "access_instructions": {
                    "mongodb_compass": "Use connection string from MONGO_URL",
                    "mongodb_atlas": "Access via Atlas dashboard",
                    "cli": "Use mongo shell with connection string"
                }
            }
        else:
            raise HTTPException(status_code=500, detail="No MongoDB stores available")
        
    except Exception as e:
        logger.error(f"Error viewing MongoDB data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Favicon endpoint
@app.get("/favicon.ico")
async def favicon():
    """Return empty response for favicon to prevent 404 errors"""
    from fastapi import Response
    return Response(status_code=204)

# Web interface endpoint
@app.get("/", response_class=HTMLResponse)
async def web_interface(request: Request):
    """Serve the React web interface"""
    return templates.TemplateResponse("react.html", {
        "request": request,
        "timestamp": int(time.time())  # Cache busting timestamp
    })

# API root endpoint
@app.get("/api")
async def api_root():
    """API root endpoint"""
    return {
        "message": "PolyDoc AI - Multi-lingual Document Understanding System with MongoDB",
        "version": "2.0.0",
        "database": "MongoDB",
        "status": "operational",
        "endpoints": {
            "/health": "Health check",
            "/upload": "Upload document (POST) - requires user_id header",
            "/chat": "Chat with documents (POST)",
            "/ws/{client_id}": "WebSocket chat",
            "/documents": "List user documents - requires user_id header",
            "/search": "Search documents - requires user_id header",
            "/admin/mongodb-data": "View MongoDB data (admin)",
            "/stats": "System statistics"
        }
    }

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main_mongodb:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
