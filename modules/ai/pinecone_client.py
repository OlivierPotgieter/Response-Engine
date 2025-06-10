"""
Pinecone Vector Search Module
Handles vector embeddings and similarity search using Pinecone and OpenAI.
"""

import os
import logging
import pandas as pd
from typing import Dict, List, Optional
from openai import OpenAI
from pinecone import Pinecone

logger = logging.getLogger(__name__)

class PineconeClient:
    def __init__(self):
        """Initialize Pinecone and OpenAI clients"""
        self.openai_client = None
        self.pinecone_client = None
        self.pinecone_index = None
        self.labeled_replies_lookup = {}
        self._initialize_clients()
        self._load_labeled_replies_lookup()

    def _initialize_clients(self):
        """Initialize OpenAI and Pinecone clients"""
        try:
            # Initialize OpenAI client
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            logger.info("OpenAI client initialized")

            # Initialize Pinecone client
            self.pinecone_client = Pinecone(
                api_key=os.getenv("PINECONE_API_KEY"),
                environment=os.getenv("PINECONE_ENV")
            )

            # Get the index
            index_name = os.getenv("PINECONE_INDEX")
            self.pinecone_index = self.pinecone_client.Index(index_name)
            logger.info(f"Pinecone client initialized with index: {index_name}")

        except Exception as e:
            logger.error(f"Error initializing clients: {e}")
            raise

    def _load_labeled_replies_lookup(self):
        """Load labeled replies for lookup by ID (matching your original script)"""
        try:
            # Try to load the CSV file you generate in your scripts
            df = pd.read_csv("labeled_full_replies.csv", encoding="utf-8-sig")
            self.labeled_replies_lookup = df.set_index("id")[["customer_comment", "full_reply_text"]].to_dict("index")
            logger.info(f"Loaded {len(self.labeled_replies_lookup)} labeled replies for lookup")

        except FileNotFoundError:
            logger.warning("labeled_full_replies.csv not found - Pinecone responses will have limited context")
            self.labeled_replies_lookup = {}
        except Exception as e:
            logger.error(f"Error loading labeled replies: {e}")
            self.labeled_replies_lookup = {}

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using OpenAI

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        try:
            logger.info(f"Generating embedding for text: {text[:100]}...")

            response = self.openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )

            vector = response.data[0].embedding
            logger.info(f"Embedding generated successfully, dimension: {len(vector)}")

            return vector

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def search_similar_responses(self, comment: str, top_k: int = 3) -> List[Dict]:
        """
        Find similar past responses using Pinecone vector search

        Args:
            comment: Customer comment to search for
            top_k: Number of similar responses to return

        Returns:
            List of dictionaries with similarity results
        """
        try:
            logger.info(f"Searching for similar responses to: {comment[:100]}...")

            # Generate embedding for the comment
            vector = self.generate_embedding(comment)

            # Query Pinecone
            result = self.pinecone_index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True
            )

            logger.info(f"Pinecone query completed. Found {len(result.matches)} matches")

            # Format results
            matches = []
            for match in result.matches:
                match_data = {
                    'id': match.id,
                    'score': float(match.score),  # Ensure JSON serializable
                    'metadata': match.metadata or {}
                }
                matches.append(match_data)
                logger.info(f"Match ID: {match.id}, Score: {match.score:.4f}")

            return matches

        except Exception as e:
            logger.error(f"Pinecone search error: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return []

    def get_response_examples(self, comment: str, top_k: int = 3) -> List[Dict]:
        """
        Get response examples with full context from labeled replies

        Args:
            comment: Customer comment to search for
            top_k: Number of examples to return

        Returns:
            List of response examples with original comment and reply text
        """
        try:
            # Get similar responses from Pinecone
            similar_responses = self.search_similar_responses(comment, top_k)

            # Build response examples with full context
            response_examples = []
            for match in similar_responses:
                rec_id = int(match['id'])
                if rec_id in self.labeled_replies_lookup:
                    # Safely get the data and handle NaN values
                    original_comment = self.labeled_replies_lookup[rec_id].get('customer_comment', '')
                    original_reply = self.labeled_replies_lookup[rec_id].get('full_reply_text', '')

                    # Handle NaN values (which show up as floats)
                    if not isinstance(original_comment, str):
                        original_comment = str(original_comment) if original_comment else ''
                    if not isinstance(original_reply, str):
                        original_reply = str(original_reply) if original_reply else ''

                    # Skip if both comment and reply are empty/invalid
                    if not original_comment.strip() and not original_reply.strip():
                        logger.warning(f"Skipping record {rec_id} - both comment and reply are empty")
                        continue

                    example = {
                        'similarity_score': match['score'],
                        'original_comment': original_comment,
                        'original_reply': original_reply,
                        'metadata': match.get('metadata', {}),
                        'record_id': rec_id
                    }
                    response_examples.append(example)
                    logger.info(f"Added response example from record {rec_id} with similarity {match['score']:.3f}")

            logger.info(f"Generated {len(response_examples)} response examples")
            return response_examples

        except Exception as e:
            logger.error(f"Error getting response examples: {e}")
            return []

    def search_with_context(self, comment: str, top_k: int = 3) -> Dict:
        """
        Comprehensive search that returns both raw matches and formatted examples

        Args:
            comment: Customer comment to search for
            top_k: Number of results to return

        Returns:
            Dict with raw matches and formatted examples
        """
        try:
            # Get raw similarity matches
            similar_responses = self.search_similar_responses(comment, top_k)

            # Get formatted response examples
            response_examples = self.get_response_examples(comment, top_k)

            return {
                "query_comment": comment,
                "similar_responses_found": len(similar_responses),
                "top_matches": similar_responses,
                "response_examples": response_examples,
                "has_labeled_context": len(response_examples) > 0,
                "labeled_replies_available": len(self.labeled_replies_lookup) > 0
            }

        except Exception as e:
            logger.error(f"Error in comprehensive search: {e}")
            return {
                "query_comment": comment,
                "similar_responses_found": 0,
                "top_matches": [],
                "response_examples": [],
                "has_labeled_context": False,
                "labeled_replies_available": len(self.labeled_replies_lookup) > 0,
                "error": str(e)
            }

    def test_connection(self) -> Dict:
        """
        Test both OpenAI and Pinecone connections

        Returns:
            Dict with connection status
        """
        status = {
            "openai_connected": False,
            "pinecone_connected": False,
            "index_accessible": False,
            "labeled_replies_loaded": len(self.labeled_replies_lookup) > 0
        }

        # Test OpenAI
        try:
            test_embedding = self.generate_embedding("test")
            status["openai_connected"] = len(test_embedding) > 0
            logger.info("✅ OpenAI connection test passed")
        except Exception as e:
            logger.error(f"❌ OpenAI connection test failed: {e}")

        # Test Pinecone
        try:
            test_result = self.pinecone_index.query(
                vector=[0.0] * 3072,  # Dummy vector for text-embedding-3-large
                top_k=1,
                include_metadata=True
            )
            status["pinecone_connected"] = True
            status["index_accessible"] = True
            logger.info("✅ Pinecone connection test passed")
        except Exception as e:
            logger.error(f"❌ Pinecone connection test failed: {e}")

        return status


# Global client instance (lazy-loaded)
_pinecone_client_instance = None

def get_pinecone_client() -> PineconeClient:
    """
    Get a global Pinecone client instance (singleton pattern)

    Returns:
        PineconeClient instance
    """
    global _pinecone_client_instance
    if _pinecone_client_instance is None:
        _pinecone_client_instance = PineconeClient()
    return _pinecone_client_instance


# Convenience functions
def search_similar_comments(comment: str, top_k: int = 3) -> List[Dict]:
    """
    Convenience function to search for similar comments

    Args:
        comment: Customer comment to search for
        top_k: Number of similar responses to return

    Returns:
        List of similar response matches
    """
    client = get_pinecone_client()
    return client.search_similar_responses(comment, top_k)


def get_response_examples_for_comment(comment: str, top_k: int = 3) -> List[Dict]:
    """
    Convenience function to get response examples with full context

    Args:
        comment: Customer comment to search for
        top_k: Number of examples to return

    Returns:
        List of response examples
    """
    client = get_pinecone_client()
    return client.get_response_examples(comment, top_k)


def search_with_full_context(comment: str, top_k: int = 3) -> Dict:
    """
    Convenience function for comprehensive search with context

    Args:
        comment: Customer comment to search for
        top_k: Number of results to return

    Returns:
        Dict with comprehensive search results
    """
    client = get_pinecone_client()
    return client.search_with_context(comment, top_k)


def test_pinecone_connection() -> Dict:
    """
    Convenience function to test Pinecone connection

    Returns:
        Dict with connection status
    """
    client = get_pinecone_client()
    return client.test_connection()