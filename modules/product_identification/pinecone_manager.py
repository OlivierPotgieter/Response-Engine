"""
Pinecone Manager for Product Identifier - FIXED: Remove 5000 item limit
Handles all Pinecone vector database operations with unlimited uploads
"""

import logging
import time
from typing import List, Dict, Optional, Any
import json

from pinecone import Pinecone, ServerlessSpec

from .config import ProductIdentifierConfig
from .embedding_service import ProductEmbedding

logger = logging.getLogger(__name__)


class PineconeManager:
    """Manager for Pinecone vector database operations - UNLIMITED UPLOADS"""

    def __init__(self):
        """Initialize Pinecone manager"""
        self.pc = None
        self.index = None
        self.index_name = ProductIdentifierConfig.PINECONE_INDEX_NAME
        self.config = ProductIdentifierConfig.get_pinecone_config()

        if not ProductIdentifierConfig.PINECONE_API_KEY:
            raise ValueError(
                "Pinecone API key not found. Please set PINECONE_API_KEY environment variable."
            )

        self._initialize_pinecone()

    def _initialize_pinecone(self):
        """Initialize Pinecone client"""
        try:
            self.pc = Pinecone(api_key=ProductIdentifierConfig.PINECONE_API_KEY)
            logger.info("Pinecone client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client: {e}")
            raise

    def create_index(self, force_recreate: bool = False) -> bool:
        """
        Create Pinecone index for product embeddings

        Args:
            force_recreate: Whether to delete and recreate existing index

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if index already exists
            existing_indexes = self.pc.list_indexes()
            index_exists = any(
                idx.name == self.index_name for idx in existing_indexes.indexes
            )

            if index_exists:
                if force_recreate:
                    logger.info(f"Deleting existing index: {self.index_name}")
                    self.pc.delete_index(self.index_name)

                    # Wait for deletion to complete
                    while self.index_name in [
                        idx.name for idx in self.pc.list_indexes().indexes
                    ]:
                        logger.info("Waiting for index deletion to complete...")
                        time.sleep(5)
                else:
                    logger.info(f"Index {self.index_name} already exists")
                    self.index = self.pc.Index(self.index_name)
                    return True

            # Create new index
            logger.info(f"Creating new index: {self.index_name}")

            self.pc.create_index(
                name=self.index_name,
                dimension=self.config["dimension"],
                metric=self.config["metric"],
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

            # Wait for index to be ready
            while not self.pc.describe_index(self.index_name).status["ready"]:
                logger.info("Waiting for index to be ready...")
                time.sleep(5)

            # Connect to the index
            self.index = self.pc.Index(self.index_name)

            logger.info(f"Index {self.index_name} created and ready")
            return True

        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False

    def connect_to_index(self) -> bool:
        """
        Connect to existing Pinecone index

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.index:
                self.index = self.pc.Index(self.index_name)

            # Test connection by getting index stats
            stats = self.index.describe_index_stats()
            logger.info(
                f"Connected to index {self.index_name}. Total vectors: {stats.total_vector_count}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to connect to index {self.index_name}: {e}")
            return False

    def upload_product_embeddings(
        self, product_embeddings: List[ProductEmbedding], batch_size: int = 200
    ) -> bool:
        """
        FIXED: Upload ALL product embeddings to Pinecone - NO LIMITS
        Uses optimized batch size of 200 for better performance

        Args:
            product_embeddings: List of ProductEmbedding objects (UNLIMITED)
            batch_size: Number of vectors to upload per batch (optimized to 200)

        Returns:
            True if successful, False otherwise
        """
        if not self.index:
            logger.error("No index connection available")
            return False

        if not product_embeddings:
            logger.warning("No product embeddings to upload")
            return True

        total_embeddings = len(product_embeddings)
        logger.info(
            f"🚀 UNLIMITED UPLOAD: Processing {total_embeddings:,} product embeddings in batches of {batch_size}"
        )

        try:
            # Prepare vectors for upload
            vectors_to_upload = []

            for product_emb in product_embeddings:
                vector_data = {
                    "id": str(product_emb.product_id),
                    "values": product_emb.embedding,
                    "metadata": product_emb.metadata,
                }
                vectors_to_upload.append(vector_data)

            # Upload in batches - NO LIMITS, PROCESS ALL
            total_batches = (len(vectors_to_upload) + batch_size - 1) // batch_size
            successful_uploads = 0
            failed_uploads = 0

            logger.info(f"📊 Starting upload of {total_embeddings:,} embeddings in {total_batches:,} batches")

            for i in range(0, len(vectors_to_upload), batch_size):
                batch = vectors_to_upload[i : i + batch_size]
                batch_num = (i // batch_size) + 1

                try:
                    logger.info(
                        f"⬆️ Uploading batch {batch_num:,}/{total_batches:,} ({len(batch)} vectors) - Progress: {(batch_num/total_batches)*100:.1f}%"
                    )

                    # Upload batch with retry logic
                    retry_count = 0
                    max_retries = 3

                    while retry_count < max_retries:
                        try:
                            self.index.upsert(vectors=batch)
                            successful_uploads += len(batch)
                            break
                        except Exception as retry_error:
                            retry_count += 1
                            if retry_count < max_retries:
                                wait_time = retry_count * 2  # Exponential backoff
                                logger.warning(f"Batch {batch_num} failed (attempt {retry_count}), retrying in {wait_time}s: {retry_error}")
                                time.sleep(wait_time)
                            else:
                                logger.error(f"Batch {batch_num} failed after {max_retries} attempts: {retry_error}")
                                failed_uploads += len(batch)

                    # Rate limiting to avoid overwhelming Pinecone
                    if batch_num % 10 == 0:  # Every 10 batches, longer pause
                        time.sleep(1.0)
                    else:
                        time.sleep(0.1)

                    # Progress logging every 50 batches
                    if batch_num % 50 == 0:
                        logger.info(f"✅ Progress update: {successful_uploads:,}/{total_embeddings:,} embeddings uploaded successfully")

                except Exception as e:
                    logger.error(f"Failed to upload batch {batch_num}: {e}")
                    failed_uploads += len(batch)
                    continue

            # Final statistics
            success_rate = (successful_uploads / total_embeddings) * 100 if total_embeddings > 0 else 0

            logger.info(f"🎉 UPLOAD COMPLETE!")
            logger.info(f"📈 Successfully uploaded: {successful_uploads:,}/{total_embeddings:,} embeddings ({success_rate:.1f}%)")

            if failed_uploads > 0:
                logger.warning(f"⚠️ Failed uploads: {failed_uploads:,} embeddings")

            # Wait for index to update
            logger.info("⏳ Waiting for index to update...")
            time.sleep(5)

            # Verify upload
            try:
                stats = self.index.describe_index_stats()
                final_count = stats.total_vector_count
                logger.info(f"🔍 Index verification: {final_count:,} total vectors in index")

                if final_count >= successful_uploads * 0.95:  # Allow for some indexing delay
                    logger.info("✅ Upload verification successful!")
                else:
                    logger.warning(f"⚠️ Upload verification: Expected ~{successful_uploads:,}, found {final_count:,}")

            except Exception as e:
                logger.error(f"Could not verify upload: {e}")

            return successful_uploads > 0

        except Exception as e:
            logger.error(f"Failed to upload product embeddings: {e}")
            return False

    def upload_embeddings_with_progress_callback(
        self,
        product_embeddings: List[ProductEmbedding],
        batch_size: int = 200,
        progress_callback = None
    ) -> bool:
        """
        BONUS: Upload embeddings with progress callback for better monitoring

        Args:
            product_embeddings: List of ProductEmbedding objects
            batch_size: Batch size for uploads
            progress_callback: Function to call with progress updates (batch_num, total_batches, success_count)

        Returns:
            True if successful, False otherwise
        """
        if not self.index:
            logger.error("No index connection available")
            return False

        if not product_embeddings:
            logger.warning("No product embeddings to upload")
            return True

        total_embeddings = len(product_embeddings)
        total_batches = (total_embeddings + batch_size - 1) // batch_size
        successful_uploads = 0

        logger.info(f"🚀 Starting upload with progress tracking: {total_embeddings:,} embeddings")

        try:
            # Prepare vectors
            vectors_to_upload = []
            for product_emb in product_embeddings:
                vector_data = {
                    "id": str(product_emb.product_id),
                    "values": product_emb.embedding,
                    "metadata": product_emb.metadata,
                }
                vectors_to_upload.append(vector_data)

            # Upload with progress tracking
            for i in range(0, len(vectors_to_upload), batch_size):
                batch = vectors_to_upload[i : i + batch_size]
                batch_num = (i // batch_size) + 1

                try:
                    self.index.upsert(vectors=batch)
                    successful_uploads += len(batch)

                    # Call progress callback if provided
                    if progress_callback:
                        progress_callback(batch_num, total_batches, successful_uploads)

                    time.sleep(0.1)  # Rate limiting

                except Exception as e:
                    logger.error(f"Batch {batch_num} failed: {e}")
                    continue

            return successful_uploads > 0

        except Exception as e:
            logger.error(f"Upload with progress tracking failed: {e}")
            return False

    def search_products(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Search for products using query embedding

        Args:
            query_embedding: Query vector embedding
            top_k: Number of results to return
            filter_dict: Optional metadata filter

        Returns:
            List of search results with scores and metadata
        """
        if not self.index:
            logger.error("No index connection available")
            return []

        try:
            # Perform vector search
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict,
            )

            # Format results
            formatted_results = []
            for match in search_results.matches:
                result = {
                    "product_id": int(match.id),
                    "confidence": float(match.score),
                    "metadata": match.metadata,
                }
                formatted_results.append(result)

            logger.info(f"Found {len(formatted_results)} product matches")
            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def search_products_with_filters(
        self,
        query_embedding: List[float],
        category: Optional[str] = None,
        brand: Optional[str] = None,
        price_range: Optional[str] = None,
        enabled_only: bool = True,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Search products with metadata filters

        Args:
            query_embedding: Query vector embedding
            category: Filter by category
            brand: Filter by brand
            price_range: Filter by price range (budget/mid/premium)
            enabled_only: Only return enabled products
            top_k: Number of results to return

        Returns:
            List of filtered search results
        """
        # Build filter dictionary
        filter_dict = {}

        if enabled_only:
            filter_dict["is_enabled"] = True

        if category:
            filter_dict["category"] = {"$eq": category}

        if brand:
            filter_dict["brand"] = {"$eq": brand}

        if price_range:
            filter_dict["price_range"] = {"$eq": price_range}

        return self.search_products(query_embedding, top_k, filter_dict)

    def get_product_by_id(self, product_id: int) -> Optional[Dict]:
        """
        Get product by ID from Pinecone

        Args:
            product_id: Product ID to fetch

        Returns:
            Product data or None if not found
        """
        if not self.index:
            logger.error("No index connection available")
            return None

        try:
            result = self.index.fetch(ids=[str(product_id)])

            if str(product_id) in result.vectors:
                vector_data = result.vectors[str(product_id)]
                return {
                    "product_id": product_id,
                    "metadata": vector_data.metadata,
                    "embedding": vector_data.values,
                }
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to fetch product {product_id}: {e}")
            return None

    def update_product_metadata(self, product_id: int, metadata_updates: Dict) -> bool:
        """
        Update metadata for a specific product

        Args:
            product_id: Product ID to update
            metadata_updates: Dictionary of metadata fields to update

        Returns:
            True if successful, False otherwise
        """
        if not self.index:
            logger.error("No index connection available")
            return False

        try:
            # Get current vector
            current_data = self.get_product_by_id(product_id)
            if not current_data:
                logger.error(f"Product {product_id} not found")
                return False

            # Update metadata
            updated_metadata = current_data["metadata"].copy()
            updated_metadata.update(metadata_updates)

            # Upsert with updated metadata
            self.index.upsert(
                vectors=[
                    {
                        "id": str(product_id),
                        "values": current_data["embedding"],
                        "metadata": updated_metadata,
                    }
                ]
            )

            logger.info(f"Updated metadata for product {product_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update product {product_id}: {e}")
            return False

    def delete_product(self, product_id: int) -> bool:
        """
        Delete product from Pinecone index

        Args:
            product_id: Product ID to delete

        Returns:
            True if successful, False otherwise
        """
        if not self.index:
            logger.error("No index connection available")
            return False

        try:
            self.index.delete(ids=[str(product_id)])
            logger.info(f"Deleted product {product_id} from index")
            return True

        except Exception as e:
            logger.error(f"Failed to delete product {product_id}: {e}")
            return False

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Pinecone index

        Returns:
            Dictionary with index statistics
        """
        if not self.index:
            return {"error": "No index connection available"}

        try:
            stats = self.index.describe_index_stats()

            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": dict(stats.namespaces) if stats.namespaces else {},
            }

        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {"error": str(e)}

    def list_products_by_category(self, category: str, limit: int = 100) -> List[Dict]:
        """
        List products by category

        Args:
            category: Category to filter by
            limit: Maximum number of products to return

        Returns:
            List of products in the category
        """
        if not self.index:
            logger.error("No index connection available")
            return []

        try:
            # Use a dummy query vector to get products by category
            # This is a workaround since Pinecone doesn't have a direct metadata-only query
            dummy_vector = [0.0] * self.config["dimension"]

            results = self.search_products_with_filters(
                query_embedding=dummy_vector, category=category, top_k=limit
            )

            return results

        except Exception as e:
            logger.error(f"Failed to list products by category {category}: {e}")
            return []

    def clear_index(self) -> bool:
        """
        Clear all vectors from the index

        Returns:
            True if successful, False otherwise
        """
        if not self.index:
            logger.error("No index connection available")
            return False

        try:
            # Delete all vectors by namespace (default namespace is '')
            self.index.delete(delete_all=True)
            logger.info("Cleared all vectors from index")
            return True

        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Pinecone connection and index

        Returns:
            Health check results
        """
        health_status = {
            "pinecone_client": False,
            "index_connection": False,
            "index_stats": None,
            "errors": [],
        }

        try:
            # Check Pinecone client
            if self.pc:
                self.pc.list_indexes()
                health_status["pinecone_client"] = True
            else:
                health_status["errors"].append("Pinecone client not initialized")

        except Exception as e:
            health_status["errors"].append(f"Pinecone client error: {e}")

        try:
            # Check index connection
            if self.index:
                stats = self.get_index_stats()
                if "error" not in stats:
                    health_status["index_connection"] = True
                    health_status["index_stats"] = stats
                else:
                    health_status["errors"].append(
                        f"Index stats error: {stats['error']}"
                    )
            else:
                health_status["errors"].append("Index not connected")

        except Exception as e:
            health_status["errors"].append(f"Index connection error: {e}")

        return health_status

    def backup_index_metadata(self, output_file: str = "pinecone_backup.json") -> bool:
        """
        Backup index metadata to JSON file

        Args:
            output_file: Path to output backup file

        Returns:
            True if successful, False otherwise
        """
        if not self.index:
            logger.error("No index connection available")
            return False

        try:
            # This is a simplified backup - in production you'd want to backup all vectors
            stats = self.get_index_stats()
            backup_data = {
                "index_name": self.index_name,
                "config": self.config,
                "stats": stats,
                "backup_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            with open(output_file, "w") as f:
                json.dump(backup_data, f, indent=2)

            logger.info(f"Index metadata backed up to {output_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to backup index metadata: {e}")
            return False


if __name__ == "__main__":
    # Test the unlimited Pinecone manager
    from .config import ProductIdentifierConfig

    ProductIdentifierConfig.setup_logging()

    try:
        manager = PineconeManager()

        # Perform health check
        health = manager.health_check()
        print("🏥 Pinecone Health Check:")
        print(f"   Client: {'✅' if health['pinecone_client'] else '❌'}")
        print(f"   Index: {'✅' if health['index_connection'] else '❌'}")

        if health["errors"]:
            print("   Errors:")
            for error in health["errors"]:
                print(f"     - {error}")

        if health["index_stats"]:
            stats = health["index_stats"]
            print("\n📊 Index Statistics:")
            print(f"   Total vectors: {stats.get('total_vector_count', 0):,}")
            print(f"   Dimension: {stats.get('dimension', 0)}")
            print(f"   Index fullness: {stats.get('index_fullness', 0):.2%}")

        print("\n🚀 UNLIMITED UPLOAD MODE ACTIVE - No 5000 item limit!")

    except Exception as e:
        print(f"❌ Error testing unlimited Pinecone manager: {e}")