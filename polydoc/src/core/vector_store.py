"""
PolyDoc - Vector Storage and Retrieval System
Uses FAISS for efficient similarity search and document context retrieval
"""

import asyncio
import logging
import json
import pickle
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import faiss
from pathlib import Path

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
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass 
class SearchResult:
    """Search result with similarity score"""
    chunk: DocumentChunk
    score: float
    relevance: str  # 'high', 'medium', 'low'

class VectorStore:
    """Vector database for document chunks using FAISS"""
    
    def __init__(self, ai_models: AIModelManager, store_path: str = "vector_store"):
        self.ai_models = ai_models
        self.store_path = Path(store_path)
        self.store_path.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # FAISS index
        self.index = None
        self.dimension = 384  # Dimension for paraphrase-multilingual-MiniLM-L12-v2
        
        # Document storage
        self.chunks: Dict[str, DocumentChunk] = {}
        self.document_metadata: Dict[str, Dict] = {}
        
        # Initialize empty index
        self._initialize_index()
        
        # Load existing data if available
        self._load_store()
    
    def _initialize_index(self):
        """Initialize FAISS index"""
        try:
            # Use IndexFlatIP for inner product similarity (cosine similarity with normalized vectors)
            self.index = faiss.IndexFlatIP(self.dimension)
            self.logger.info(f"Initialized FAISS index with dimension {self.dimension}")
        except Exception as e:
            self.logger.error(f"Error initializing FAISS index: {e}")
            raise
    
    async def add_document(
        self, 
        document_id: str, 
        elements: List, 
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> int:
        """Add document elements to vector store with chunking"""
        try:
            chunks_added = 0
            all_texts = []
            document_chunks = []
            
            # Create chunks from document elements
            for element in elements:
                if not element.text.strip():
                    continue
                
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
            
            # Generate embeddings for all chunks at once
            self.logger.info(f"Generating embeddings for {len(all_texts)} chunks...")
            embeddings = await self.ai_models.generate_embeddings(all_texts)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to FAISS index
            start_idx = self.index.ntotal
            self.index.add(embeddings.astype('float32'))
            
            # Store chunks with embeddings
            for i, chunk in enumerate(document_chunks):
                chunk.embedding = embeddings[i]
                self.chunks[chunk.chunk_id] = chunk
                chunks_added += 1
            
            # Store document metadata
            self.document_metadata[document_id] = {
                'total_chunks': len(document_chunks),
                'added_timestamp': asyncio.get_event_loop().time(),
                'languages': list(set(chunk.language for chunk in document_chunks)),
                'pages': list(set(chunk.page_number for chunk in document_chunks)),
                'element_types': list(set(chunk.element_type for chunk in document_chunks))
            }
            
            self.logger.info(f"Added {chunks_added} chunks for document {document_id}")
            
            # Save to disk
            await self._save_store()
            
            return chunks_added
            
        except Exception as e:
            self.logger.error(f"Error adding document {document_id}: {e}")
            raise
    
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
        top_k: int = 5, 
        document_id: Optional[str] = None,
        language: Optional[str] = None,
        page_number: Optional[int] = None
    ) -> List[SearchResult]:
        """Search for similar chunks using vector similarity"""
        try:
            if self.index.ntotal == 0:
                return []
            
            # Generate query embedding
            query_embedding = await self.ai_models.generate_embeddings([query])
            query_embedding = query_embedding.astype('float32')
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search in FAISS index
            scores, indices = self.index.search(query_embedding, min(top_k * 2, self.index.ntotal))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                
                # Find chunk by index
                chunk = self._get_chunk_by_index(idx)
                if chunk is None:
                    continue
                
                # Apply filters
                if document_id and chunk.document_id != document_id:
                    continue
                if language and chunk.language != language:
                    continue
                if page_number and chunk.page_number != page_number:
                    continue
                
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
            
            # Sort by score and limit to top_k
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error in search: {e}")
            return []
    
    def _get_chunk_by_index(self, faiss_idx: int) -> Optional[DocumentChunk]:
        """Get chunk by FAISS index"""
        # Since FAISS indices are sequential, we need to map them to chunk IDs
        # This is a simplified approach - in production you might want a more efficient mapping
        chunks_list = list(self.chunks.values())
        if 0 <= faiss_idx < len(chunks_list):
            return chunks_list[faiss_idx]
        return None
    
    async def get_context_for_question(
        self, 
        question: str,
        document_id: Optional[str] = None,
        max_context_length: int = 8000  # Increased from 4000 to 8000
    ) -> Tuple[str, List[int]]:
        """Get relevant context for answering a question"""
        try:
            # Search for relevant chunks
            results = await self.search(
                query=question,
                top_k=15,
                document_id=document_id
            )
            
            if not results:
                return "", []
            
            # Build context from high-relevance chunks first
            context_parts = []
            page_numbers = set()
            current_length = 0
            
            for result in results:
                # Include more low relevance results for comprehensive context
                if result.relevance == 'low' and len(context_parts) > 5:  # Changed from 2 to 5
                    continue  # Skip low relevance if we have enough high/medium relevance
                
                chunk_text = result.chunk.text
                
                if current_length + len(chunk_text) > max_context_length:
                    # Be more generous with remaining space
                    remaining_space = max_context_length - current_length
                    if remaining_space > 200:  # Changed from 100 to 200
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
            self.logger.error(f"Error getting context: {e}")
            return "", []
    
    async def get_document_summary_context(self, document_id: str) -> str:
        """Get representative chunks for document summary"""
        try:
            if document_id not in self.document_metadata:
                return ""
            
            # Get chunks for this document
            doc_chunks = [chunk for chunk in self.chunks.values() 
                         if chunk.document_id == document_id]
            
            if not doc_chunks:
                return ""
            
            # Group by page and element type
            page_groups = {}
            for chunk in doc_chunks:
                page_key = f"page_{chunk.page_number}"
                if page_key not in page_groups:
                    page_groups[page_key] = []
                page_groups[page_key].append(chunk)
            
            # Select representative chunks from each page
            summary_chunks = []
            for page_key, chunks in page_groups.items():
                # Prefer headings and paragraphs over other types
                sorted_chunks = sorted(chunks, key=lambda x: (
                    0 if x.element_type == 'heading' else
                    1 if x.element_type == 'paragraph' else
                    2 if x.element_type == 'table' else 3
                ))
                
                # Take first 2 chunks per page
                summary_chunks.extend(sorted_chunks[:2])
            
            # Combine text
            summary_text = "\n\n".join([
                f"[Page {chunk.page_number}] {chunk.text}" 
                for chunk in summary_chunks[:10]  # Limit to 10 chunks
            ])
            
            return summary_text
            
        except Exception as e:
            self.logger.error(f"Error getting summary context: {e}")
            return ""
    
    async def _save_store(self):
        """Save vector store to disk"""
        try:
            # Save FAISS index
            index_path = self.store_path / "faiss_index.idx"
            faiss.write_index(self.index, str(index_path))
            
            # Save chunks (without embeddings to save space)
            chunks_data = {}
            for chunk_id, chunk in self.chunks.items():
                chunk_dict = asdict(chunk)
                chunk_dict['embedding'] = None  # Don't save embedding separately
                chunks_data[chunk_id] = chunk_dict
            
            chunks_path = self.store_path / "chunks.json"
            with open(chunks_path, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, ensure_ascii=False, indent=2)
            
            # Save document metadata
            metadata_path = self.store_path / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.document_metadata, f, ensure_ascii=False, indent=2)
            
            self.logger.info("Vector store saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving vector store: {e}")
    
    def _load_store(self):
        """Load vector store from disk"""
        try:
            index_path = self.store_path / "faiss_index.idx"
            chunks_path = self.store_path / "chunks.json"
            metadata_path = self.store_path / "metadata.json"
            
            if not all(p.exists() for p in [index_path, chunks_path, metadata_path]):
                self.logger.info("No existing vector store found, starting fresh")
                return
            
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Load chunks
            with open(chunks_path, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            for chunk_id, chunk_dict in chunks_data.items():
                # Convert back to DocumentChunk
                chunk = DocumentChunk(
                    chunk_id=chunk_dict['chunk_id'],
                    document_id=chunk_dict['document_id'],
                    text=chunk_dict['text'],
                    page_number=chunk_dict['page_number'],
                    element_type=chunk_dict['element_type'],
                    bbox=tuple(chunk_dict['bbox']),
                    language=chunk_dict.get('language', 'en'),  # Default to 'en' if not found
                    metadata=chunk_dict.get('metadata')
                )
                self.chunks[chunk_id] = chunk
            
            # Load metadata
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.document_metadata = json.load(f)
            
            self.logger.info(f"Loaded vector store with {len(self.chunks)} chunks")
            
        except Exception as e:
            self.logger.error(f"Error loading vector store: {e}")
            # Initialize empty store on error
            self._initialize_index()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            'total_chunks': len(self.chunks),
            'total_documents': len(self.document_metadata),
            'faiss_index_size': self.index.ntotal if self.index else 0,
            'languages': list(set(chunk.language for chunk in self.chunks.values())),
            'element_types': list(set(chunk.element_type for chunk in self.chunks.values())),
            'store_path': str(self.store_path)
        }
    
    async def remove_document(self, document_id: str) -> bool:
        """Remove a document from the vector store"""
        try:
            if document_id not in self.document_metadata:
                return False
            
            # Remove chunks
            chunks_to_remove = [chunk_id for chunk_id, chunk in self.chunks.items() 
                              if chunk.document_id == document_id]
            
            for chunk_id in chunks_to_remove:
                del self.chunks[chunk_id]
            
            # Remove metadata
            del self.document_metadata[document_id]
            
            # Rebuild FAISS index (since we can't easily remove from index)
            await self._rebuild_index()
            
            self.logger.info(f"Removed document {document_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing document {document_id}: {e}")
            return False
    
    async def _rebuild_index(self):
        """Rebuild FAISS index from current chunks"""
        try:
            if not self.chunks:
                self._initialize_index()
                return
            
            # Generate embeddings for all chunks
            texts = [chunk.text for chunk in self.chunks.values()]
            embeddings = await self.ai_models.generate_embeddings(texts)
            
            # Normalize embeddings
            faiss.normalize_L2(embeddings)
            
            # Rebuild index
            self._initialize_index()
            self.index.add(embeddings.astype('float32'))
            
            self.logger.info("Rebuilt FAISS index")
            
        except Exception as e:
            self.logger.error(f"Error rebuilding index: {e}")
            raise
