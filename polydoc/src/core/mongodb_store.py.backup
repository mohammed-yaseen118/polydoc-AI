"""
PolyDoc - MongoDB Integration
MongoDB-based document storage, metadata management, and vector search
"""

import asyncio
import logging
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import numpy as np
from pathlib import Path

# MongoDB imports
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING, TEXT
from bson import ObjectId
# import gridfs  # Removed for now - not compatible with async Motor

from src.models.ai_models import AIModelManager

@dataclass
class DocumentChunk:
    """Represents a chunk of document with metadata"""
    chunk_id: str
    document_id: str
    text: str
    page_number: int
    element_type: str
    bbox: Tuple[float, float, float, float]
    language: str
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass 
class SearchResult:
    """Search result with similarity score"""
    chunk: DocumentChunk
    score: float
    relevance: str  # 'high', 'medium', 'low'

class MongoDBStore:
    """MongoDB-based document storage and vector search for PolyDoc"""
    
    def __init__(self, ai_models: AIModelManager, mongo_url: str = None, user_id: str = None):
        self.ai_models = ai_models
        self.user_id = user_id
        self.logger = logging.getLogger(__name__)
        
        # Use only local MongoDB - Atlas configuration removed
        self.mongo_base_url = mongo_url or os.getenv('MONGO_URL', 'mongodb://localhost:27017')
        
        # Ensure we're only using local MongoDB
        if 'mongodb+srv' in self.mongo_base_url or '@' in self.mongo_base_url:
            self.logger.warning("Atlas configuration detected but not allowed. Using local MongoDB.")
            self.mongo_base_url = 'mongodb://localhost:27017'
        
        # Ensure no database name is specified in the URL
        if '://' in self.mongo_base_url and '/' in self.mongo_base_url.split('://')[1]:
            base_part = self.mongo_base_url.split('://')[0] + '://' + self.mongo_base_url.split('://')[1].split('/')[0]
            self.mongo_base_url = base_part
        
        # MongoDB client and database
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self.fs = None  # GridFS for file storage
        
        # Collection names
        self.documents_collection = "documents"
        self.chunks_collection = "document_chunks"
        self.chat_sessions_collection = "chat_sessions"
        self.users_collection = "users"  # For user management
        
        # Vector search configuration
        self.vector_index_name = "vector_search_index"
        self.embedding_dimension = 384  # sentence-transformers dimension
    
    async def connect(self):
        """Connect to MongoDB and create user-specific database"""
        try:
            self.client = AsyncIOMotorClient(self.mongo_base_url)
            
            # Test connection
            await self.client.admin.command('ping')
            
            # Create user-specific database name
            if self.user_id:
                # Clean user_id for database name (replace invalid chars)
                clean_user_id = self.user_id.replace('-', '_').replace('@', '_at_').replace('.', '_')
                db_name = f"polydoc_user_{clean_user_id}"
            else:
                db_name = 'polydoc_ai_default'
            
            self.db = self.client[db_name]
            
            # GridFS initialization removed - not compatible with async Motor
            # self.fs = gridfs.GridFS(self.client[db_name])
            
            # Create user record if it doesn't exist
            if self.user_id:
                await self._ensure_user_record()
            
            # Create indexes
            await self._create_indexes()
            
            self.logger.info(f"Connected to MongoDB: {db_name} for user: {self.user_id or 'default'}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from MongoDB"""
        if self.client:
            self.client.close()
            self.logger.info("Disconnected from MongoDB")
    
    async def _create_indexes(self):
        """Create necessary indexes for optimal performance"""
        try:
            # Documents collection indexes
            await self.db[self.documents_collection].create_index([
                ("user_id", ASCENDING),
                ("created_at", DESCENDING)
            ])
            
            # Document chunks indexes
            await self.db[self.chunks_collection].create_index([
                ("document_id", ASCENDING)
            ])
            await self.db[self.chunks_collection].create_index([
                ("user_id", ASCENDING)
            ])
            
            # Text search index for chunks (language-agnostic to support all languages)
            try:
                await self.db[self.chunks_collection].create_index([
                    ("text", TEXT)
                ], default_language='none')  # Use 'none' to disable language-specific stemming
            except Exception as text_index_error:
                self.logger.warning(f"Could not create language-specific text index: {text_index_error}")
                # Fallback: create basic text index without language support
                try:
                    await self.db[self.chunks_collection].create_index([
                        ("text", TEXT)
                    ])
                except Exception as fallback_error:
                    self.logger.warning(f"Could not create fallback text index: {fallback_error}")
            
            # Chat sessions indexes
            await self.db[self.chat_sessions_collection].create_index([
                ("user_id", ASCENDING),
                ("updated_at", DESCENDING)
            ])
            
            self.logger.info("MongoDB indexes created successfully")
            
        except Exception as e:
            self.logger.warning(f"Error creating indexes: {e}")
    
    async def _ensure_user_record(self):
        """Create user record if it doesn't exist"""
        try:
            if not self.user_id:
                return
            
            # Check if user record exists
            existing_user = await self.db[self.users_collection].find_one({"user_id": self.user_id})
            
            if not existing_user:
                user_record = {
                    "user_id": self.user_id,
                    "created_at": datetime.now(timezone.utc),
                    "last_login": datetime.now(timezone.utc),
                    "total_documents": 0,
                    "total_storage_used": 0,
                    "settings": {
                        "language": "en",
                        "theme": "system",
                        "notifications": True
                    },
                    "subscription": {
                        "type": "free",
                        "limits": {
                            "max_documents": 100,
                            "max_storage_mb": 1000,
                            "max_pages_per_doc": 50
                        }
                    }
                }
                
                await self.db[self.users_collection].insert_one(user_record)
                self.logger.info(f"Created user record for: {self.user_id}")
            else:
                # Update last login
                await self.db[self.users_collection].update_one(
                    {"user_id": self.user_id},
                    {"$set": {"last_login": datetime.now(timezone.utc)}}
                )
                
        except Exception as e:
            self.logger.error(f"Error ensuring user record: {e}")
    
    async def add_document(
        self, 
        document_id: str,
        user_id: str,
        filename: str,
        elements: List, 
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        user_email: Optional[str] = None
    ) -> int:
        """Add document to MongoDB with chunking and vector embeddings"""
        try:
            chunks_added = 0
            all_texts = []
            document_chunks = []
            
            # Create document record
            document_record = {
                "_id": ObjectId() if not ObjectId.is_valid(document_id) else ObjectId(document_id),
                "user_id": user_id,
                "user_email": user_email,  # Store user email for better identification
                "filename": filename,
                "file_size": 0,  # Calculate from elements
                "content_type": self._get_content_type(filename),
                "upload_date": datetime.now(timezone.utc),
                "processed_date": datetime.now(timezone.utc),
                "status": "processing",
                "language": "en",  # Detect from elements
                "page_count": 0,
                "metadata": {},
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }
            
            # Process elements into chunks
            for element in elements:
                if not element.text.strip():
                    continue
                
                # Update document metadata
                document_record["page_count"] = max(document_record["page_count"], element.page_number)
                if element.language and element.language != 'unknown':
                    document_record["language"] = element.language
                
                # Split large elements into chunks
                element_chunks = self._create_chunks(
                    element.text, 
                    chunk_size, 
                    chunk_overlap
                )
                
                for i, chunk_text in enumerate(element_chunks):
                    chunk_id = f"{document_id}_{element.page_number}_{len(document_chunks)}"
                    
                    chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        text=chunk_text,
                        page_number=element.page_number,
                        element_type=element.element_type,
                        bbox=element.bbox,
                        language=element.language or 'en',
                        metadata={
                            'confidence': element.confidence,
                            'font_info': element.font_info,
                            'chunk_index': i,
                            'total_chunks': len(element_chunks)
                        }
                    )
                    
                    document_chunks.append(chunk)
                    all_texts.append(chunk_text)
            
            if not all_texts:
                self.logger.warning(f"No text found in document {document_id}")
                return 0
            
            # Generate embeddings for all chunks
            self.logger.info(f"Generating embeddings for {len(all_texts)} chunks...")
            embeddings = await self.ai_models.generate_embeddings(all_texts)
            
            # Convert numpy arrays to lists for MongoDB storage
            embeddings_list = embeddings.tolist()
            
            # Prepare chunk documents for MongoDB
            chunk_documents = []
            for i, chunk in enumerate(document_chunks):
                # Map unsupported language codes to supported ones for MongoDB
                mongodb_language = self._map_language_for_mongodb(chunk.language)
                
                chunk_doc = {
                    "_id": ObjectId(),
                    "chunk_id": chunk.chunk_id,
                    "document_id": document_id,
                    "user_id": user_id,
                    "text": chunk.text,
                    "page_number": chunk.page_number,
                    "element_type": chunk.element_type,
                    "bbox": list(chunk.bbox),
                    "language": chunk.language,  # Keep original for our use
                    "mongodb_language": mongodb_language,  # Mapped version for MongoDB indexing
                    "embedding": embeddings_list[i],  # Store as list
                    "metadata": chunk.metadata,
                    "created_at": datetime.now(timezone.utc)
                }
                chunk_documents.append(chunk_doc)
                chunks_added += 1
            
            # Insert document record
            document_record["status"] = "completed"
            await self.db[self.documents_collection].insert_one(document_record)
            
            # Insert chunk documents in batch with error handling
            if chunk_documents:
                try:
                    await self.db[self.chunks_collection].insert_many(chunk_documents)
                except Exception as batch_error:
                    self.logger.warning(f"Batch insert failed: {batch_error}")
                    # Fallback: insert documents one by one
                    self.logger.info("Falling back to individual document insertion...")
                    successful_inserts = 0
                    for chunk_doc in chunk_documents:
                        try:
                            # Remove language field that might cause issues
                            safe_chunk_doc = chunk_doc.copy()
                            # Keep only essential fields to avoid language indexing issues
                            essential_fields = {
                                "_id": safe_chunk_doc["_id"],
                                "chunk_id": safe_chunk_doc["chunk_id"],
                                "document_id": safe_chunk_doc["document_id"],
                                "user_id": safe_chunk_doc["user_id"],
                                "text": safe_chunk_doc["text"],
                                "page_number": safe_chunk_doc["page_number"],
                                "element_type": safe_chunk_doc["element_type"],
                                "bbox": safe_chunk_doc["bbox"],
                                "language": safe_chunk_doc["language"],
                                "embedding": safe_chunk_doc["embedding"],
                                "metadata": safe_chunk_doc["metadata"],
                                "created_at": safe_chunk_doc["created_at"]
                            }
                            await self.db[self.chunks_collection].insert_one(essential_fields)
                            successful_inserts += 1
                        except Exception as individual_error:
                            self.logger.warning(f"Failed to insert chunk {chunk_doc['chunk_id']}: {individual_error}")
                            continue
                    
                    chunks_added = successful_inserts
                    self.logger.info(f"Successfully inserted {successful_inserts}/{len(chunk_documents)} chunks")
            
            self.logger.info(f"Added document {document_id} with {chunks_added} chunks to MongoDB")
            return chunks_added
            
        except Exception as e:
            self.logger.error(f"Error adding document {document_id} to MongoDB: {e}")
            # Update document status to failed
            await self.db[self.documents_collection].update_one(
                {"_id": ObjectId(document_id) if ObjectId.is_valid(document_id) else {"filename": filename}},
                {"$set": {"status": "failed", "updated_at": datetime.now(timezone.utc)}}
            )
            raise
    
    def _get_content_type(self, filename: str) -> str:
        """Get content type from filename"""
        ext = Path(filename).suffix.lower()
        content_types = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.ppt': 'application/vnd.ms-powerpoint',
            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            '.txt': 'text/plain',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.tiff': 'image/tiff',
            '.bmp': 'image/bmp'
        }
        return content_types.get(ext, 'application/octet-stream')
    
    def _map_language_for_mongodb(self, language: str) -> str:
        """Map language codes to MongoDB-supported languages for text indexing"""
        # MongoDB text search supported languages
        # Full list: https://docs.mongodb.com/manual/reference/text-search-languages/
        
        mongodb_supported = {
            'da': 'danish',
            'nl': 'dutch', 
            'en': 'english',
            'fi': 'finnish',
            'fr': 'french',
            'de': 'german',
            'hu': 'hungarian',
            'it': 'italian',
            'nb': 'norwegian',
            'pt': 'portuguese',
            'ro': 'romanian',
            'ru': 'russian',
            'es': 'spanish',
            'sv': 'swedish',
            'tr': 'turkish'
        }
        
        # Check if language is directly supported
        if language in mongodb_supported:
            return mongodb_supported[language]
        
        # Map Indian languages to 'none' to avoid language-specific processing
        indian_languages = {'hi', 'kn', 'mr', 'te', 'ta', 'bn', 'gu', 'pa', 'ml', 'or', 'as'}
        if language in indian_languages:
            return 'none'  # Use 'none' for unsupported languages
        
        # Default to 'none' for any other unsupported language
        return 'none'
    
    def _create_chunks(
        self, 
        text: str, 
        chunk_size: int, 
        chunk_overlap: int
    ) -> List[str]:
        """Create overlapping chunks from text"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                sentence_end = text.rfind('.', start, end)
                if sentence_end != -1 and sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - chunk_overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    async def search(
        self, 
        query: str, 
        user_id: str,
        top_k: int = 5, 
        document_id: Optional[str] = None,
        language: Optional[str] = None,
        page_number: Optional[int] = None
    ) -> List[SearchResult]:
        """Search for similar chunks using vector similarity and text search"""
        try:
            # Generate query embedding
            query_embedding = await self.ai_models.generate_embeddings([query])
            query_vector = query_embedding[0].tolist()
            
            # Build search pipeline
            pipeline = []
            
            # Match stage - filter by user, document, etc.
            match_stage = {"user_id": user_id}
            if document_id:
                match_stage["document_id"] = document_id
            if language:
                match_stage["language"] = language
            if page_number:
                match_stage["page_number"] = page_number
            
            pipeline.append({"$match": match_stage})
            
            # Add vector search scoring (manual cosine similarity)
            pipeline.append({
                "$addFields": {
                    "similarity_score": {
                        "$let": {
                            "vars": {
                                "dot_product": {
                                    "$reduce": {
                                        "input": {"$range": [0, len(query_vector)]},
                                        "initialValue": 0,
                                        "in": {
                                            "$add": [
                                                "$$value",
                                                {"$multiply": [
                                                    {"$arrayElemAt": ["$embedding", "$$this"]},
                                                    query_vector["$$this"]
                                                ]}
                                            ]
                                        }
                                    }
                                }
                            },
                            "in": "$$dot_product"  # Simplified: assuming normalized vectors
                        }
                    }
                }
            })
            
            # Sort by similarity score
            pipeline.append({"$sort": {"similarity_score": -1}})
            
            # Limit results
            pipeline.append({"$limit": top_k})
            
            # Execute aggregation pipeline
            cursor = self.db[self.chunks_collection].aggregate(pipeline)
            results = []
            
            async for doc in cursor:
                # Create DocumentChunk from MongoDB document
                chunk = DocumentChunk(
                    chunk_id=doc["chunk_id"],
                    document_id=doc["document_id"],
                    text=doc["text"],
                    page_number=doc["page_number"],
                    element_type=doc["element_type"],
                    bbox=tuple(doc["bbox"]),
                    language=doc["language"],
                    embedding=doc["embedding"],
                    metadata=doc.get("metadata")
                )
                
                score = doc.get("similarity_score", 0.0)
                
                # Determine relevance based on score
                if score > 0.8:
                    relevance = 'high'
                elif score > 0.6:
                    relevance = 'medium'
                else:
                    relevance = 'low'
                
                results.append(SearchResult(
                    chunk=chunk,
                    score=float(score),
                    relevance=relevance
                ))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in MongoDB search: {e}")
            return []
    
    async def get_context_for_question(
        self, 
        question: str,
        user_id: str,
        document_id: Optional[str] = None,
        max_context_length: int = 2000
    ) -> Tuple[str, List[int]]:
        """Get relevant context for answering a question"""
        try:
            # Search for relevant chunks
            results = await self.search(
                query=question,
                user_id=user_id,
                top_k=10,
                document_id=document_id
            )
            
            if not results:
                return "", []
            
            # Build context from high-relevance chunks first
            context_parts = []
            page_numbers = set()
            current_length = 0
            
            for result in results:
                if result.relevance == 'low' and len(context_parts) > 2:
                    continue  # Skip low relevance if we have enough high/medium relevance
                
                chunk_text = result.chunk.text
                
                if current_length + len(chunk_text) > max_context_length:
                    # Truncate if needed
                    remaining_space = max_context_length - current_length
                    if remaining_space > 100:  # Only add if we have reasonable space
                        chunk_text = chunk_text[:remaining_space] + "..."
                        context_parts.append(f"[Page {result.chunk.page_number}] {chunk_text}")
                        page_numbers.add(result.chunk.page_number)
                    break
                
                context_parts.append(f"[Page {result.chunk.page_number}] {chunk_text}")
                page_numbers.add(result.chunk.page_number)
                current_length += len(chunk_text) + 20  # +20 for page annotation
            
            context = "\n\n".join(context_parts)
            pages = sorted(list(page_numbers))
            
            return context, pages
            
        except Exception as e:
            self.logger.error(f"Error getting context from MongoDB: {e}")
            return "", []
    
    async def list_user_documents(self, user_id: str) -> List[Dict[str, Any]]:
        """List all documents for a user"""
        try:
            cursor = self.db[self.documents_collection].find(
                {"user_id": user_id},
                sort=[("created_at", DESCENDING)]
            )
            
            documents = []
            async for doc in cursor:
                # Count chunks for this document
                chunk_count = await self.db[self.chunks_collection].count_documents({
                    "document_id": str(doc["_id"])
                })
                
                documents.append({
                    "document_id": str(doc["_id"]),
                    "filename": doc["filename"],
                    "status": doc["status"],
                    "upload_date": doc["upload_date"],
                    "page_count": doc["page_count"],
                    "language": doc["language"],
                    "chunk_count": chunk_count,
                    "user_email": doc.get("user_email"),  # Include user email in response
                    "content_type": doc.get("content_type")
                })
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Error listing documents for user {user_id}: {e}")
            return []
    
    async def delete_document(self, document_id: str, user_id: str) -> bool:
        """Delete a document and all its chunks"""
        try:
            # Verify ownership
            doc = await self.db[self.documents_collection].find_one({
                "_id": ObjectId(document_id),
                "user_id": user_id
            })
            
            if not doc:
                return False
            
            # Delete chunks
            await self.db[self.chunks_collection].delete_many({
                "document_id": document_id
            })
            
            # Delete document
            await self.db[self.documents_collection].delete_one({
                "_id": ObjectId(document_id)
            })
            
            self.logger.info(f"Deleted document {document_id} for user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting document {document_id}: {e}")
            return False
    
    async def save_chat_session(
        self, 
        user_id: str, 
        document_id: str, 
        title: str, 
        messages: List[Dict]
    ) -> str:
        """Save chat session to MongoDB"""
        try:
            chat_session = {
                "_id": ObjectId(),
                "user_id": user_id,
                "document_id": document_id,
                "title": title,
                "messages": messages,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }
            
            result = await self.db[self.chat_sessions_collection].insert_one(chat_session)
            
            return str(result.inserted_id)
            
        except Exception as e:
            self.logger.error(f"Error saving chat session: {e}")
            return None
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            stats = {
                "total_documents": await self.db[self.documents_collection].count_documents({}),
                "total_chunks": await self.db[self.chunks_collection].count_documents({}),
                "total_chat_sessions": await self.db[self.chat_sessions_collection].count_documents({}),
                "database_name": self.db.name,
                "connection_string": self.mongo_base_url.split('@')[-1] if '@' in self.mongo_base_url else self.mongo_base_url
            }
            
            # Language distribution
            pipeline = [
                {"$group": {"_id": "$language", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            
            language_stats = []
            async for doc in self.db[self.documents_collection].aggregate(pipeline):
                language_stats.append({"language": doc["_id"], "count": doc["count"]})
            
            stats["languages"] = language_stats
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}

# Global instance
mongodb_store: Optional[MongoDBStore] = None

async def get_mongodb_store(ai_models: AIModelManager) -> MongoDBStore:
    """Get or create MongoDB store instance"""
    global mongodb_store
    
    if mongodb_store is None:
        mongodb_store = MongoDBStore(ai_models)
        await mongodb_store.connect()
    
    return mongodb_store
