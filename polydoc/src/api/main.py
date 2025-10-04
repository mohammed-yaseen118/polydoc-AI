"""
PolyDoc - FastAPI Backend with Real-time Chat Interface
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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, Header, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from pydantic import BaseModel
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

# Import our custom modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.document_processor import DocumentProcessor, ProcessedDocument
from src.core.mongodb_store import MongoDBStore, get_mongodb_store
from src.core.vector_store import VectorStore
from src.models.ai_models import AIModelManager, DocumentAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)

# Pydantic models for API
class UserInfo(BaseModel):
    user_id: str
    email: Optional[str] = None
    display_name: Optional[str] = None

class ChatMessage(BaseModel):
    message: str
    document_id: Optional[str] = None
    language: Optional[str] = 'en'
    user_id: str  # Add user_id to all requests

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    summary: Optional[str] = None
    english_summary: Optional[str] = None
    original_language: Optional[str] = None
    translation_needed: Optional[bool] = False
    translation_confidence: Optional[float] = 0.0
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

# Simple user validation (since Firebase handles auth on frontend)
async def get_user_id(user_id: Optional[str] = Header(None, alias="X-User-ID")):
    """Extract user ID from header for authenticated requests"""
    if not user_id:
        # For backward compatibility, allow requests without user ID (use default)
        logger.info("No user ID provided, using default user")
        return "default_user"
    logger.info(f"User ID received: {user_id}")
    return user_id

# User-specific MongoDB store instances
user_mongodb_stores: Dict[str, MongoDBStore] = {}

async def get_user_mongodb_store(user_id: str = Depends(get_user_id)) -> MongoDBStore:
    """Get or create user-specific MongoDB store"""
    try:
        if user_id not in user_mongodb_stores:
            if not ai_models:
                raise HTTPException(status_code=503, detail="AI models not initialized")
            
            # Create new user-specific MongoDB store
            logger.info(f"Creating MongoDB store for user: {user_id}")
            store = MongoDBStore(ai_models, user_id=user_id)
            await store.connect()
            user_mongodb_stores[user_id] = store
            logger.info(f"MongoDB store created successfully for user: {user_id}")
        
        return user_mongodb_stores[user_id]
    except Exception as e:
        logger.error(f"Failed to create MongoDB store for user {user_id}: {e}")
        # Fallback to default storage without MongoDB
        raise HTTPException(status_code=503, detail="Database connection failed. Please ensure MongoDB is running.")

# Global instances
app = FastAPI(title="PolyDoc", version="1.0.0")
ai_models: Optional[AIModelManager] = None
document_processor: Optional[DocumentProcessor] = None
mongodb_store: Optional[MongoDBStore] = None  # Keep for backward compatibility
document_analyzer: Optional[DocumentAnalyzer] = None
vector_store: Optional[VectorStore] = None  # Keep but will be replaced by MongoDB

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

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global initialization status
initialization_status = {
    "status": "initializing",
    "progress": 0,
    "message": "Starting initialization...",
    "models_ready": False,
    "error": None
}

# Background model initialization
async def initialize_models_background():
    """Initialize AI models in the background"""
    global ai_models, document_processor, mongodb_store, document_analyzer, vector_store, initialization_status
    
    try:
        initialization_status["message"] = "Loading AI models..."
        initialization_status["progress"] = 10
        
        logger.info("Starting background AI model initialization...")
        
        # Initialize AI models (this may take some time)
        try:
            initialization_status["message"] = "Loading embedding model..."
            initialization_status["progress"] = 25
            logger.info("Initializing AI Model Manager...")
            ai_models = AIModelManager()
            logger.info("AI Model Manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AI models: {e}")
            raise
        
        try:
            initialization_status["message"] = "Initializing document processor..."
            initialization_status["progress"] = 50
            logger.info("Initializing Document Processor...")
            document_processor = DocumentProcessor()
            logger.info("Document Processor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize document processor: {e}")
            raise
        
        try:
            initialization_status["message"] = "Setting up vector store..."
            initialization_status["progress"] = 75
            logger.info("Initializing Vector Store...")
            vector_store = VectorStore(ai_models)
            logger.info("Vector Store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
        
        try:
            initialization_status["message"] = "Finalizing setup..."
            initialization_status["progress"] = 90
            logger.info("Initializing Document Analyzer...")
            document_analyzer = DocumentAnalyzer(ai_models)
            logger.info("Document Analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize document analyzer: {e}")
            raise
        
        initialization_status["status"] = "ready"
        initialization_status["progress"] = 100
        initialization_status["message"] = "System ready!"
        initialization_status["models_ready"] = True
        
        logger.info("PolyDoc initialization completed successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize PolyDoc: {e}")
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
    logger.info("Starting PolyDoc web server...")
    logger.info("Starting background AI model initialization...")
    
    # Start model initialization in background
    import asyncio
    task = asyncio.create_task(initialize_models_background())
    
    # Don't await the task - let it run in background
    logger.info("Web server is ready, AI models loading in background...")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "models_loaded": ai_models is not None,
        "vector_store_ready": vector_store is not None,
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
    user_store: MongoDBStore = Depends(get_user_mongodb_store),
    user_id: str = Depends(get_user_id)
):
    """Upload and process a document"""
    start_time = time.time()
    
    try:
        # Check if models are initialized
        if not all([ai_models, document_processor, vector_store, document_analyzer]):
            raise HTTPException(
                status_code=503, 
                detail="System is still initializing. Please wait a moment and try again."
            )
        # Validate file type (now supports many more formats)
        supported_extensions = {
            # Original formats
            '.pdf', '.docx', '.pptx', '.png', '.jpg', '.jpeg', '.tiff', '.bmp',
            # New text formats
            '.txt', '.rtf', '.md', '.markdown',
            # Spreadsheet formats
            '.csv', '.xlsx', '.xls',
            # Structured data formats
            '.json', '.xml', '.html', '.htm',
            # OpenDocument formats
            '.odt'
        }
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in supported_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_extension}. Supported formats: {', '.join(sorted(supported_extensions))}"
            )
        
        # Create unique document ID
        document_id = str(uuid.uuid4())
        
        # Save uploaded file
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / f"{document_id}_{file.filename}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Get estimated processing time
        time_estimate = document_processor.estimate_processing_time(str(file_path))
        logger.info(f"Estimated processing time: {time_estimate['estimated_seconds']}s for {file.filename}")
        
        # Process document
        logger.info(f"Processing document {document_id}: {file.filename}")
        processed_doc = await document_processor.process_document(str(file_path))
        
        # Add to user-specific MongoDB store instead of vector store
        chunks_added = await user_store.add_document(
            document_id=document_id,
            user_id=user_id,
            filename=file.filename,
            elements=processed_doc.elements
        )
        
        # Generate dual-language summary
        summary_data = await document_analyzer.generate_document_summary(
            processed_doc.elements, 
            summary_length='medium',
            dual_language=True
        )
        
        # Get document statistics
        stats = document_processor.get_document_stats(processed_doc)
        stats['chunks_added'] = chunks_added
        
        processing_time = time.time() - start_time
        
        logger.info(f"Document {document_id} processed successfully in {processing_time:.2f}s")
        
        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            status="processed",
            summary=summary_data.get('summary', 'No summary available'),
            english_summary=summary_data.get('english_summary', summary_data.get('summary', 'No summary available')),
            original_language=summary_data.get('original_language', 'unknown'),
            translation_needed=summary_data.get('translation_needed', False),
            translation_confidence=summary_data.get('translation_confidence', 0.0),
            statistics=stats,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Document list endpoint
@app.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    user_store: MongoDBStore = Depends(get_user_mongodb_store),
    user_id: str = Depends(get_user_id)
):
    """Get list of processed documents"""
    try:
        # Get user-specific documents from MongoDB
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
async def chat(
    message: ChatMessage,
    user_store: MongoDBStore = Depends(get_user_mongodb_store)
):
    """Process chat message and return response"""
    start_time = time.time()
    
    try:
        if not all([ai_models, vector_store]):
            raise HTTPException(status_code=500, detail="AI models not initialized")
        
        # Get relevant context from user's MongoDB store
        context, page_numbers = await user_store.get_context_for_question(
            question=message.message,
            user_id=message.user_id,
            document_id=message.document_id
        )
        
        if not context:
            return ChatResponse(
                response="I couldn't find relevant information in the documents to answer your question.",
                confidence=0.0,
                sources=[],
                processing_time=time.time() - start_time,
                timestamp=time.time()
            )
        
        # Generate response using AI model
        response = await ai_models.answer_question(
            question=message.message,
            context=context,
            language=message.language
        )
        
        # Format response with page references
        formatted_response = response.content
        if page_numbers:
            formatted_response += f"\n\n📄 Sources: Page(s) {', '.join(map(str, page_numbers))}"
        
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
                
                context, page_numbers = await vector_store.get_context_for_question(
                    chat_message.message,
                    document_id=chat_message.document_id
                )
                
                if not context:
                    response_msg = {
                        "type": "response",
                        "response": "I couldn't find relevant information to answer your question.",
                        "confidence": 0.0,
                        "sources": [],
                        "processing_time": time.time() - start_time,
                        "timestamp": time.time()
                    }
                else:
                    # Generate AI response
                    response = await ai_models.answer_question(
                        question=chat_message.message,
                        context=context,
                        language=chat_message.language or 'en'
                    )
                    
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

# Document analysis endpoint
@app.get("/analyze/{document_id}")
async def analyze_document(document_id: str):
    """Get detailed analysis of a document"""
    try:
        if document_id not in vector_store.document_metadata:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get document chunks
        doc_chunks = [chunk for chunk in vector_store.chunks.values() 
                     if chunk.document_id == document_id]
        
        if not doc_chunks:
            raise HTTPException(status_code=404, detail="No content found for document")
        
        # Perform analysis
        analysis = await document_analyzer.analyze_document_structure(doc_chunks)
        
        # Get vector store statistics
        vs_stats = vector_store.get_statistics()
        
        return {
            "document_id": document_id,
            "analysis": analysis,
            "vector_store_stats": vs_stats,
            "total_chunks": len(doc_chunks)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Delete document endpoint
@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document from the system"""
    try:
        success = await vector_store.remove_document(document_id)
        
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
    document_id: Optional[str] = None,
    top_k: int = 5
):
    """Search for relevant content across documents"""
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        results = await vector_store.search(
            query=query,
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

# Estimate processing time endpoint
@app.post("/estimate-time")
async def estimate_processing_time(file: UploadFile = File(...)):
    """Estimate processing time for uploaded file"""
    try:
        # Save file temporarily to get size
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Get estimate
        estimate = document_processor.estimate_processing_time(temp_file)
        
        # Clean up temp file
        os.remove(temp_file)
        
        return {
            "filename": file.filename,
            "estimated_seconds": estimate['estimated_seconds'],
            "estimated_minutes": estimate['estimated_minutes'],
            "file_size_mb": estimate['file_size_mb'],
            "complexity": estimate['complexity'],
            "file_type": Path(file.filename).suffix.lower()
        }
        
    except Exception as e:
        logger.error(f"Error estimating processing time: {e}")
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
        
        if vector_store:
            stats["vector_store"] = vector_store.get_statistics()
        
        if ai_models:
            stats["ai_models"] = ai_models.get_model_info()
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
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
    import time
    return templates.TemplateResponse("react.html", {
        "request": request,
        "timestamp": int(time.time())  # Cache busting timestamp
    })

# Test route for debugging static files
@app.get("/test", response_class=HTMLResponse)
async def test_static_files():
    """Test static file serving"""
    with open("test_static.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# API root endpoint
@app.get("/api")
async def api_root():
    """API root endpoint"""
    return {
        "message": "PolyDoc - Multi-lingual Document Understanding System",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "/health": "Health check",
            "/upload": "Upload document (POST)",
            "/chat": "Chat with documents (POST)",
            "/ws/{client_id}": "WebSocket chat",
            "/documents": "List documents",
            "/search": "Search documents",
            "/analyze/{document_id}": "Analyze document",
            "/stats": "System statistics"
        }
    }

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
