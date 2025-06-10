"""
Embedding Service for Product Identifier
Generates embeddings for products and customer comments using OpenAI
"""

import logging
import time
import pickle
import json
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import openai
from openai import OpenAI
import re

from config import ProductIdentifierConfig

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
    """Service for generating and managing product embeddings"""

    def __init__(self):
        """Initialize the embedding service"""
        self.client = OpenAI(api_key=ProductIdentifierConfig.OPENAI_API_KEY)
        self.config = ProductIdentifierConfig.get_embedding_config()
        self.embeddings_cache = {}

        if not ProductIdentifierConfig.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")

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
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\-\.\(\)\/\+]', ' ', text)

        # Remove common prefixes/suffixes that don't add meaning
        prefixes_to_remove = ['wootware', 'desktop', 'notebook']
        for prefix in prefixes_to_remove:
            text = re.sub(f'^{prefix}\\s+', '', text)
            text = re.sub(f'\\s+{prefix}\\s+', ' ', text)

        # Normalize common terms
        normalizations = {
            'graphics card': 'gpu',
            'video card': 'gpu',
            'processor': 'cpu',
            'central processing unit': 'cpu',
            'solid state drive': 'ssd',
            'hard disk drive': 'hdd',
            'random access memory': 'ram'
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
        core_fields = ['name', 'manufacturer', 'category', 'sku']
        for field in core_fields:
            if product_data.get(field):
                components.append(str(product_data[field]))

        # Use the SearchText field - this is what it's designed for!
        if product_data.get('search_text'):
            components.append(product_data['search_text'])

        # Join, clean, and return
        searchable_text = ' '.join(components)
        return self.preprocess_text(searchable_text)

    def _extract_feature_words(self, text: str) -> List[str]:
        """
        Extract important feature words from text
        Focus on specific features, not generic terms

        Args:
            text: Text to extract features from

        Returns:
            List of feature words
        """
        # Important feature keywords that add value
        valuable_features = [
            # Performance features
            'turbo', 'boost', 'overclocked', 'overclocking', 'factory', 'custom',

            # Build quality features
            'military', 'grade', 'reinforced', 'premium', 'professional', 'enterprise',

            # Cooling features
            'liquid', 'cooled', 'silent', 'quiet', 'fanless', 'thermal',

            # Aesthetic features
            'rgb', 'led', 'backlit', 'illuminated', 'tempered', 'glass',

            # Connectivity features
            'wireless', 'bluetooth', 'wifi', 'gigabit', 'dual', 'quad',

            # Gaming specific
            'gaming', 'esports', 'competitive', 'streaming', 'content',

            # Form factors
            'compact', 'mini', 'micro', 'slim', 'low-profile', 'full-size',

            # Efficiency
            'energy', 'efficient', 'eco', 'green', 'sustainable',

            # Compatibility
            'compatible', 'universal', 'standard', 'certified'
        ]

        found_features = []
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)

        for word in words:
            if word.lower() in valuable_features:
                found_features.append(word.lower())

        return found_features

    def _clean_and_deduplicate_terms(self, terms: List[str]) -> List[str]:
        """
        Clean and deduplicate terms while preserving important ones

        Args:
            terms: List of terms to clean

        Returns:
            Cleaned and deduplicated list of terms
        """
        if not terms:
            return []

        # Convert to lowercase and strip
        cleaned_terms = []
        for term in terms:
            if term and isinstance(term, str):
                cleaned = term.lower().strip()
                if len(cleaned) >= 2:  # Minimum length
                    cleaned_terms.append(cleaned)

        # Remove duplicates while preserving order
        seen = set()
        deduplicated = []

        for term in cleaned_terms:
            if term not in seen:
                seen.add(term)
                deduplicated.append(term)

        return deduplicated

    def _get_category_terms(self, category: str) -> List[str]:
        """Get relevant terms for a product category"""
        category_mappings = {
            'graphics cards': ['gpu', 'graphics', 'video card', 'gaming'],
            'processors': ['cpu', 'processor', 'computing'],
            'motherboards': ['motherboard', 'mobo', 'mainboard'],
            'memory': ['ram', 'memory', 'ddr4', 'ddr5'],
            'storage': ['storage', 'drive', 'ssd', 'hdd'],
            'monitors': ['monitor', 'display', 'screen'],
            'peripherals': ['gaming', 'rgb', 'mechanical']
        }

        category_lower = category.lower()
        for cat_key, terms in category_mappings.items():
            if cat_key in category_lower:
                return terms

        return []

    def generate_embedding(self, text: str, max_retries: int = 3) -> Optional[List[float]]:
        """
        Generate embedding for a single text

        Args:
            text: Text to embed
            max_retries: Maximum number of retry attempts

        Returns:
            Embedding vector or None if failed
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None

        # Check cache first
        text_hash = hash(text)
        if text_hash in self.embeddings_cache:
            return self.embeddings_cache[text_hash]

        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.config['model'],
                    input=text,
                    encoding_format="float"
                )

                embedding = response.data[0].embedding

                # Cache the result
                self.embeddings_cache[text_hash] = embedding

                return embedding

            except Exception as e:
                logger.warning(f"Embedding generation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(self.config['retry_delay'] * (attempt + 1))
                else:
                    logger.error(f"Failed to generate embedding after {max_retries} attempts: {e}")
                    return None

        return None

    def generate_batch_embeddings(self, texts: List[str], batch_size: Optional[int] = None) -> List[
        Optional[List[float]]]:
        """
        Generate embeddings for multiple texts in batches

        Args:
            texts: List of texts to embed
            batch_size: Size of batches (uses config default if None)

        Returns:
            List of embeddings (None for failed embeddings)
        """
        if not texts:
            return []

        if batch_size is None:
            batch_size = self.config['batch_size']

        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        logger.info(f"Generating embeddings for {len(texts)} texts in {total_batches} batches")

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1

            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)")

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

                    # Cache the result
                    text_hash = hash(text)
                    self.embeddings_cache[text_hash] = embedding

                all_embeddings.extend(batch_embeddings)

                # Rate limiting
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Batch embedding generation failed for batch {batch_num}: {e}")
                all_embeddings.extend([None] * len(batch_texts))

        success_count = sum(1 for emb in all_embeddings if emb is not None)
        logger.info(f"Successfully generated {success_count}/{len(texts)} embeddings")

        return all_embeddings

    def create_product_embeddings(self, products_data: List[Dict]) -> List[ProductEmbedding]:
        """
        Create embeddings using SearchText field approach
        Much simpler and cleaner
        """
        logger.info(f"Creating embeddings for {len(products_data)} products using SearchText field")

        # Create searchable texts using SearchText field
        searchable_texts = []
        for product in products_data:
            searchable_text = self.create_product_searchable_text(product)
            searchable_texts.append(searchable_text)

        # Generate embeddings in batches
        embeddings = self.generate_batch_embeddings(searchable_texts)

        # Create ProductEmbedding objects
        product_embeddings = []
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')

        for i, (product, embedding, searchable_text) in enumerate(zip(products_data, embeddings, searchable_texts)):
            if embedding is not None:
                # Create metadata for Pinecone
                metadata = {
                    'name': product.get('name', ''),
                    'manufacturer': product.get('manufacturer', ''),
                    'manufacturer_id': str(product.get('manufacturer_id', '')),
                    'category': product.get('category', ''),
                    'sku': product.get('sku', ''),
                    'is_enabled': product.get('is_enabled', True),
                    'is_eol': product.get('is_eol', False),
                    'searchable_text': searchable_text[:1000]  # Limit metadata size
                }

                product_embedding = ProductEmbedding(
                    product_id=product.get('product_id'),
                    embedding=embedding,
                    metadata=metadata,
                    searchable_text=searchable_text,
                    created_at=current_time
                )

                product_embeddings.append(product_embedding)
            else:
                logger.warning(
                    f"Failed to create embedding for product {product.get('product_id')}: {product.get('name')}")

        logger.info(f"Successfully created {len(product_embeddings)} product embeddings using SearchText")
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
            'gpu': 'graphics card',
            'cpu': 'processor',
            'mobo': 'motherboard',
            'psu': 'power supply',
            'ssd': 'solid state drive',
            'hdd': 'hard drive'
        }

        words = processed.split()
        expanded_words = []

        for word in words:
            if word in abbreviations:
                expanded_words.append(abbreviations[word])
            else:
                expanded_words.append(word)

        return ' '.join(expanded_words)

    def save_embeddings_cache(self, filepath: str = None):
        """Save embeddings cache to file"""
        if filepath is None:
            filepath = ProductIdentifierConfig.EMBEDDINGS_CACHE_PATH

        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
            logger.info(f"Embeddings cache saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save embeddings cache: {e}")

    def load_embeddings_cache(self, filepath: str = None) -> bool:
        """Load embeddings cache from file"""
        if filepath is None:
            filepath = ProductIdentifierConfig.EMBEDDINGS_CACHE_PATH

        try:
            with open(filepath, 'rb') as f:
                self.embeddings_cache = pickle.load(f)
            logger.info(f"Embeddings cache loaded from {filepath} ({len(self.embeddings_cache)} entries)")
            return True
        except Exception as e:
            logger.warning(f"Failed to load embeddings cache: {e}")
            self.embeddings_cache = {}
            return False

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about embeddings cache"""
        return {
            'cache_size': len(self.embeddings_cache),
            'model_used': self.config['model'],
            'dimension': self.config['dimension']
        }


if __name__ == "__main__":
    # Test the embedding service
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

        # Test preprocessing
        test_comment = "Looking for a good GPU for gaming, maybe RTX 4080 or similar"
        processed = service.preprocess_comment(test_comment)
        print(f"\n📝 Comment preprocessing:")
        print(f"   Original: {test_comment}")
        print(f"   Processed: {processed}")

    except Exception as e:
        print(f"❌ Error testing embedding service: {e}")