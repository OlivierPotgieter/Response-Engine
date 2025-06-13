"""
Embedding Service for Product Identifier - FIXED: Add progress tracking for unlimited uploads
Generates embeddings for products and customer comments using OpenAI
"""

import logging
import time
import pickle
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass
from openai import OpenAI
import re

from .config import ProductIdentifierConfig

logger = logging.getLogger(__name__)


@dataclass
class ProductEmbedding:
    """Data class for product embeddings"""

    product_id: int
    embedding: List[float]
    metadata: Dict[str, Any]
    searchable_text: str
    created_at: str


class EmbeddingService:
    """Service for generating and managing product embeddings with progress tracking"""

    def __init__(self):
        """Initialize the embedding service"""
        self.client = OpenAI(api_key=ProductIdentifierConfig.OPENAI_API_KEY)
        self.config = ProductIdentifierConfig.get_embedding_config()

        if not ProductIdentifierConfig.OPENAI_API_KEY:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
            )

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for embedding generation

        Args:
            text: Raw text to preprocess

        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower().strip()

        # Remove extra whitespace and normalize
        text = re.sub(r"\s+", " ", text)

        # Remove special characters but keep important ones
        text = re.sub(r"[^\w\s\-\.\(\)\/\+]", " ", text)

        # Remove common prefixes/suffixes that don't add meaning
        prefixes_to_remove = ["wootware", "desktop", "notebook"]
        for prefix in prefixes_to_remove:
            text = re.sub(f"^{prefix}\\s+", "", text)
            text = re.sub(f"\\s+{prefix}\\s+", " ", text)

        # Normalize common terms
        normalizations = {
            "graphics card": "gpu",
            "video card": "gpu",
            "processor": "cpu",
            "central processing unit": "cpu",
            "solid state drive": "ssd",
            "hard disk drive": "hdd",
            "random access memory": "ram",
        }

        for old_term, new_term in normalizations.items():
            text = text.replace(old_term, new_term)

        return text.strip()

    def create_product_searchable_text(self, product_data: Dict) -> str:
        """
        Create searchable text using the SearchText field + core product info
        Simple, clean, and uses data that's already curated

        Args:
            product_data: Product information dictionary

        Returns:
            Searchable text from SearchText field plus core identifiers
        """
        components = []

        # Core product identifiers (always include)
        core_fields = ["name", "manufacturer", "category", "sku"]
        for field in core_fields:
            if product_data.get(field):
                components.append(str(product_data[field]))

        # Use the SearchText field - this is what it's designed for!
        if product_data.get("search_text"):
            components.append(product_data["search_text"])

        # Join, clean, and return
        searchable_text = " ".join(components)
        return self.preprocess_text(searchable_text)

    def generate_embedding(self, text: str, max_retries: int = 3) -> Optional[List[float]]:
        """
        Generate embedding for a single text with retry logic

        Args:
            text: Text to embed
            max_retries: Maximum number of retry attempts

        Returns:
            Embedding vector or None if failed
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None

        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.config['model'],
                    input=text,
                    encoding_format="float"
                )

                embedding = response.data[0].embedding
                return embedding

            except Exception as e:
                logger.warning(f"Embedding generation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(self.config['retry_delay'] * (attempt + 1))
                else:
                    logger.error(f"Failed to generate embedding after {max_retries} attempts: {e}")
                    return None

        return None

    def generate_batch_embeddings(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts in batches with progress tracking

        Args:
            texts: List of texts to embed
            batch_size: Size of batches (uses config default if None)
            progress_callback: Optional callback function for progress updates (current, total)

        Returns:
            List of embeddings (None for failed embeddings)
        """
        if not texts:
            return []

        if batch_size is None:
            batch_size = self.config['batch_size']

        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        logger.info(f"Generating embeddings for {len(texts):,} texts in {total_batches:,} batches")

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1

            logger.info(f"Processing batch {batch_num:,}/{total_batches:,} ({len(batch_texts)} texts)")

            try:
                # Filter out empty texts
                valid_texts = [(idx, text) for idx, text in enumerate(batch_texts) if text and text.strip()]

                if not valid_texts:
                    all_embeddings.extend([None] * len(batch_texts))
                    continue

                # Generate embeddings for valid texts
                response = self.client.embeddings.create(
                    model=self.config['model'],
                    input=[text for _, text in valid_texts],
                    encoding_format="float"
                )

                # Map embeddings back to original positions
                batch_embeddings = [None] * len(batch_texts)
                for result_idx, (original_idx, text) in enumerate(valid_texts):
                    embedding = response.data[result_idx].embedding
                    batch_embeddings[original_idx] = embedding

                all_embeddings.extend(batch_embeddings)

                # Call progress callback if provided
                if progress_callback:
                    progress_callback(len(all_embeddings), len(texts))

                # Rate limiting
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Batch embedding generation failed for batch {batch_num}: {e}")
                all_embeddings.extend([None] * len(batch_texts))

        success_count = sum(1 for emb in all_embeddings if emb is not None)
        logger.info(f"Successfully generated {success_count:,}/{len(texts):,} embeddings")

        return all_embeddings

    def create_product_embeddings(
        self,
        products_data: List[Dict],
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[ProductEmbedding]:
        """
        Create embeddings for ALL products with progress tracking (no limits)

        Args:
            products_data: List of all product data (unlimited)
            progress_callback: Optional callback (current, total, product_name)
        """
        total_products = len(products_data)
        logger.info(
            f"🚀 Creating embeddings for ALL {total_products:,} products (UNLIMITED MODE)"
        )

        # Create searchable texts using SearchText field
        searchable_texts = []
        for i, product in enumerate(products_data):
            searchable_text = self.create_product_searchable_text(product)
            searchable_texts.append(searchable_text)

            # Progress callback for text preparation
            if progress_callback and (i % 500 == 0 or i == total_products - 1):
                product_name = product.get("name", "Unknown")
                progress_callback(i + 1, total_products, f"Preparing: {product_name}")

        # Generate embeddings in batches with progress tracking
        def embedding_progress(current, total):
            if progress_callback:
                current_product = products_data[min(current - 1, len(products_data) - 1)]
                product_name = current_product.get("name", "Unknown")
                progress_callback(current, total, f"Embedding: {product_name}")

        embeddings = self.generate_batch_embeddings(
            searchable_texts,
            progress_callback=embedding_progress
        )

        # Create ProductEmbedding objects
        product_embeddings = []
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")

        for i, (product, embedding, searchable_text) in enumerate(
            zip(products_data, embeddings, searchable_texts)
        ):
            if embedding is not None:
                # Create metadata for Pinecone
                metadata = {
                    "name": product.get("name", ""),
                    "manufacturer": product.get("manufacturer", ""),
                    "manufacturer_id": str(product.get("manufacturer_id", "")),
                    "category": product.get("category", ""),
                    "sku": product.get("sku", ""),
                    "is_enabled": product.get("is_enabled", True),
                    "is_eol": product.get("is_eol", False),
                    "searchable_text": searchable_text[:1000],  # Limit metadata size
                }

                product_embedding = ProductEmbedding(
                    product_id=product.get("product_id"),
                    embedding=embedding,
                    metadata=metadata,
                    searchable_text=searchable_text,
                    created_at=current_time,
                )

                product_embeddings.append(product_embedding)
            else:
                logger.warning(
                    f"Failed to create embedding for product {product.get('product_id')}: {product.get('name')}"
                )

        success_rate = (len(product_embeddings) / total_products) * 100
        logger.info(
            f"✅ Successfully created {len(product_embeddings):,}/{total_products:,} product embeddings ({success_rate:.1f}% success rate)"
        )
        return product_embeddings

    def preprocess_comment(self, comment: str) -> str:
        """
        Preprocess customer comment for product extraction

        Args:
            comment: Raw customer comment

        Returns:
            Preprocessed comment text
        """
        if not comment:
            return ""

        # Basic cleaning
        processed = self.preprocess_text(comment)

        # Expand common abbreviations
        abbreviations = {
            "gpu": "graphics card",
            "cpu": "processor",
            "mobo": "motherboard",
            "psu": "power supply",
            "ssd": "solid state drive",
            "hdd": "hard drive",
        }

        words = processed.split()
        expanded_words = []

        for word in words:
            if word in abbreviations:
                expanded_words.append(abbreviations[word])
            else:
                expanded_words.append(word)

        return " ".join(expanded_words)

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Return cache stats (no actual cache - placeholder for compatibility)
        """
        return {
            'cache_size': 0,  # No cache anymore
            'model_used': self.config['model'],
            'dimension': self.config['dimension'],
            'note': 'Caching disabled - embeddings stored in Pinecone only'
        }

    def estimate_embedding_cost(self, num_products: int) -> Dict[str, Any]:
        """
        Estimate the cost of generating embeddings for a given number of products

        Args:
            num_products: Number of products to estimate for

        Returns:
            Cost estimation details
        """
        # OpenAI pricing for text-embedding-3-large (as of 2024)
        # $0.00013 per 1K tokens
        avg_tokens_per_product = 50  # Estimate based on SearchText field
        total_tokens = num_products * avg_tokens_per_product

        cost_per_1k_tokens = 0.00013
        estimated_cost = (total_tokens / 1000) * cost_per_1k_tokens

        return {
            "num_products": num_products,
            "estimated_tokens": total_tokens,
            "estimated_cost_usd": round(estimated_cost, 4),
            "model": self.config['model'],
            "note": "This is an estimate - actual costs may vary"
        }


if __name__ == "__main__":
    # Test the unlimited embedding service
    from .config import ProductIdentifierConfig

    ProductIdentifierConfig.setup_logging()

    try:
        service = EmbeddingService()

        # Test single embedding
        test_text = "NVIDIA GeForce RTX 4090 Gaming Graphics Card"
        embedding = service.generate_embedding(test_text)

        if embedding:
            print(f"✅ Successfully generated embedding for: {test_text}")
            print(f"   Embedding dimension: {len(embedding)}")
        else:
            print("❌ Failed to generate embedding")

        # Test cost estimation
        cost_estimate = service.estimate_embedding_cost(10000)
        print(f"\n💰 Cost estimate for 10,000 products:")
        print(f"   Estimated cost: ${cost_estimate['estimated_cost_usd']}")
        print(f"   Estimated tokens: {cost_estimate['estimated_tokens']:,}")

        # Test preprocessing
        test_comment = "Looking for a good GPU for gaming, maybe RTX 4080 or similar"
        processed = service.preprocess_comment(test_comment)
        print("\n📝 Comment preprocessing:")
        print(f"   Original: {test_comment}")
        print(f"   Processed: {processed}")

        print("\n🚀 UNLIMITED MODE: Ready to process ALL products without limits!")

    except Exception as e:
        print(f"❌ Error testing unlimited embedding service: {e}")