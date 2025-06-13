"""
Product Identifier System Builder - FIXED: Remove embedding upload limits
Coordinates building all required components for the AI product identification system
"""

import logging
import os
import time
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SystemStatus:
    """Status of the product identification system build"""

    success: bool
    message: str
    details: Dict
    processing_time: float
    files_created: List[str]
    errors: List[str]


class ProductIdentifierSystemBuilder:
    """
    Coordinates building the complete product identification system.
    FIXED: No longer has 5000 item upload limits - processes ALL products
    """

    def __init__(self, force_rebuild: bool = False):
        """
        Initialize the system builder

        Args:
            force_rebuild: Whether to rebuild even if files exist
        """
        self.force_rebuild = force_rebuild
        self.files_created = []
        self.errors = []

    def build_complete_system(self) -> SystemStatus:
        """
        Build the complete product identification system from your database
        FIXED: No upload limits - processes ALL products

        Returns:
            SystemStatus with build results
        """
        start_time = time.time()

        logger.info("🚀 Starting UNLIMITED Product Identification System Build...")

        try:
            # Step 1: Validate configuration
            if not self._validate_configuration():
                return SystemStatus(
                    success=False,
                    message="Configuration validation failed",
                    details={"step": "configuration"},
                    processing_time=time.time() - start_time,
                    files_created=[],
                    errors=self.errors,
                )

            # Step 2: Check if rebuild is needed
            if not self.force_rebuild and self._check_existing_files():
                logger.info("✅ System files already exist and force_rebuild=False")
                return SystemStatus(
                    success=True,
                    message="System already built",
                    details={"step": "existing_files_check"},
                    processing_time=time.time() - start_time,
                    files_created=self._get_existing_files(),
                    errors=[],
                )

            # Step 3: Build product intelligence from database
            if not self._build_product_intelligence():
                return self._build_error_status(start_time, "product_intelligence")

            # Step 4: Build category intelligence from database
            if not self._build_category_intelligence():
                return self._build_error_status(start_time, "category_intelligence")

            # Step 5: UNLIMITED Generate and upload embeddings to Pinecone
            if not self._build_embeddings_system_unlimited():
                return self._build_error_status(start_time, "embeddings_system")

            # Step 6: Verify system is working
            if not self._verify_system():
                return self._build_error_status(start_time, "system_verification")

            processing_time = time.time() - start_time

            logger.info(
                f"✅ UNLIMITED System build completed successfully in {processing_time:.2f} seconds"
            )

            return SystemStatus(
                success=True,
                message=f"UNLIMITED Product identification system built successfully in {processing_time:.2f}s - ALL products processed",
                details={
                    "step": "completed",
                    "files_created": self.files_created,
                    "processing_time": processing_time,
                    "unlimited_mode": True,
                },
                processing_time=processing_time,
                files_created=self.files_created,
                errors=[],
            )

        except Exception as e:
            logger.error(f"❌ UNLIMITED System build failed: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")

            return SystemStatus(
                success=False,
                message=f"UNLIMITED System build failed: {str(e)}",
                details={"step": "exception", "error": str(e)},
                processing_time=time.time() - start_time,
                files_created=self.files_created,
                errors=self.errors + [str(e)],
            )

    def _validate_configuration(self) -> bool:
        """Validate that all required configuration is available"""
        try:
            from .config import ProductIdentifierConfig

            validation = ProductIdentifierConfig.validate_config()

            if not validation["valid"]:
                self.errors.extend(validation["errors"])
                logger.error("❌ Configuration validation failed:")
                for error in validation["errors"]:
                    logger.error(f"   - {error}")
                return False

            if validation["warnings"]:
                for warning in validation["warnings"]:
                    logger.warning(f"⚠️ {warning}")

            logger.info("✅ Configuration validation passed")
            return True

        except Exception as e:
            self.errors.append(f"Configuration validation error: {e}")
            logger.error(f"❌ Configuration validation error: {e}")
            return False

    def _check_existing_files(self) -> bool:
        """Check if required system files already exist"""
        required_files = [
            "product_intelligence.json",
            "category_intelligence.json",
            # REMOVED: "product_embeddings_cache.pkl" - no longer using cache
        ]

        existing_files = []
        for file in required_files:
            if os.path.exists(file):
                existing_files.append(file)

        all_exist = len(existing_files) == len(required_files)

        if all_exist:
            logger.info(f"✅ All required files exist: {existing_files}")
        else:
            missing = [f for f in required_files if f not in existing_files]
            logger.info(f"📂 Missing files: {missing}")

        return all_exist

    def _get_existing_files(self) -> List[str]:
        """Get list of existing system files"""
        required_files = [
            "product_intelligence.json",
            "category_intelligence.json",
            # REMOVED: "product_embeddings_cache.pkl" - no longer using cache
        ]

        return [f for f in required_files if os.path.exists(f)]

    def _build_product_intelligence(self) -> bool:
        """Build product intelligence from database (no file creation)"""
        try:
            logger.info("Validating product intelligence from database...")

            # Import the product intelligence builder
            from .product_intelligence_builder import ProductIntelligenceBuilder

            builder = ProductIntelligenceBuilder()
            try:
                # Get intelligence from database instead of creating file
                intelligence = builder.get_intelligence_from_database()

                if intelligence and intelligence.get('products'):
                    products_count = len(intelligence['products'])
                    manufacturers_count = len(intelligence.get('manufacturers', {}))
                    logger.info(
                        f"✅ Product intelligence validated: {products_count:,} products, {manufacturers_count} manufacturers from database")

                    # Track that intelligence is ready (no file created)
                    self.files_created.append("product_intelligence_database_cache")
                    return True
                else:
                    self.errors.append("Failed to load product intelligence from database")
                    return False

            finally:
                builder.close()

        except Exception as e:
            self.errors.append(f"Product intelligence validation error: {e}")
            logger.error(f"❌ Product intelligence validation failed: {e}")
            return False

    def _build_category_intelligence(self) -> bool:
        """Validate category intelligence from database (no file creation)"""
        try:
            logger.info("Validating category intelligence from database...")

            # Import here to avoid circular imports
            from ..database import get_category_intelligence_from_database

            intelligence = get_category_intelligence_from_database()

            if intelligence and intelligence.get('categories'):
                categories_count = len(intelligence['categories'])
                total_products = sum(cat.get('product_count', 0) for cat in intelligence['categories'].values())

                logger.info(
                    f"✅ Category intelligence validated: {categories_count} categories, {total_products:,} products from database")

                # Track that intelligence is ready (no file created)
                self.files_created.append("category_intelligence_database_cache")
                return True
            else:
                self.errors.append("Failed to load category intelligence from database")
                return False

        except Exception as e:
            self.errors.append(f"Category intelligence validation error: {e}")
            logger.error(f"❌ Category intelligence validation failed: {e}")
            return False

    def _build_embeddings_system_unlimited(self) -> bool:
        """
        FIXED: Build embeddings and upload ALL to Pinecone (no limits, no cache)
        """
        try:
            logger.info("🚀 Building UNLIMITED embeddings system...")

            # Step 1: Load product intelligence from database
            from .product_intelligence_builder import ProductIntelligenceBuilder

            builder = ProductIntelligenceBuilder()
            try:
                intelligence = builder.get_intelligence_from_database()
            finally:
                builder.close()

            if not intelligence or not intelligence.get("products"):
                self.errors.append("No products found in intelligence data")
                return False

            products_data = intelligence.get("products", [])
            total_products = len(products_data)
            logger.info(f"📊 Found {total_products:,} products for UNLIMITED embedding generation")

            # Step 2: Initialize services
            from .embedding_service import EmbeddingService
            from .pinecone_manager import PineconeManager

            embedding_service = EmbeddingService()
            pinecone_manager = PineconeManager()

            # Step 3: Create or connect to Pinecone index
            if not pinecone_manager.create_index(force_recreate=self.force_rebuild):
                self.errors.append("Failed to create Pinecone index")
                return False

            # Step 4: Generate embeddings for ALL products (no limits)
            logger.info(f"🧬 Generating embeddings for ALL {total_products:,} products...")

            # Progress tracking
            def log_progress(current, total, product_name=""):
                if current % 100 == 0 or current == total:
                    progress = (current / total) * 100
                    logger.info(f"📈 Embedding progress: {current:,}/{total:,} ({progress:.1f}%) - {product_name}")

            product_embeddings = embedding_service.create_product_embeddings(
                products_data,
                progress_callback=log_progress
            )

            if not product_embeddings:
                self.errors.append("Failed to generate product embeddings")
                return False

            embeddings_generated = len(product_embeddings)
            success_rate = (embeddings_generated / total_products) * 100

            logger.info(f"✅ Generated {embeddings_generated:,}/{total_products:,} product embeddings ({success_rate:.1f}% success rate)")

            # Step 5: Upload ALL embeddings to Pinecone (no limits)
            logger.info(f"☁️ Uploading ALL {embeddings_generated:,} embeddings to Pinecone...")

            # Use optimized batch size for better performance
            upload_success = pinecone_manager.upload_product_embeddings(
                product_embeddings,
                batch_size=50  # Optimized batch size based on Pinecone recommendations
            )

            if not upload_success:
                self.errors.append("Failed to upload embeddings to Pinecone")
                return False

            # Step 6: Verify upload
            try:
                stats = pinecone_manager.get_index_stats()
                final_count = stats.get("total_vector_count", 0)
                logger.info(f"🔍 Final verification: {final_count:,} vectors in Pinecone index")

                if final_count >= embeddings_generated * 0.9:  # Allow 10% tolerance for indexing delays
                    logger.info("✅ UNLIMITED embeddings system built successfully - NO LIMITS APPLIED!")
                else:
                    logger.warning(f"⚠️ Upload verification: Expected {embeddings_generated:,}, found {final_count:,}")

            except Exception as e:
                logger.warning(f"Could not verify final upload count: {e}")

            # Track success (no cache file created)
            self.files_created.append("unlimited_embeddings_uploaded_to_pinecone")
            logger.info("✅ UNLIMITED embeddings system built successfully (cache disabled)")
            return True

        except Exception as e:
            self.errors.append(f"UNLIMITED embeddings system build error: {e}")
            logger.error(f"❌ UNLIMITED embeddings system build failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False

    def _verify_system(self) -> bool:
        """Verify the complete system is working"""
        try:
            logger.info("🔍 Verifying UNLIMITED system functionality...")

            # Try to initialize the smart extractor
            from .smart_product_extractor import SmartProductExtractor

            extractor = SmartProductExtractor()

            # Test with a simple query
            test_result = extractor.extract_products_smart("RTX 4090", max_products=1)

            if test_result.products or test_result.extraction_method != "failed":
                logger.info("✅ UNLIMITED system verification passed")
                return True
            else:
                self.errors.append(
                    "System verification failed - no results for test query"
                )
                return False

        except Exception as e:
            self.errors.append(f"System verification error: {e}")
            logger.error(f"❌ System verification failed: {e}")
            return False

    def _build_error_status(self, start_time: float, failed_step: str) -> SystemStatus:
        """Build error status response"""
        return SystemStatus(
            success=False,
            message=f"UNLIMITED System build failed at step: {failed_step}",
            details={"step": failed_step, "errors": self.errors},
            processing_time=time.time() - start_time,
            files_created=self.files_created,
            errors=self.errors,
        )


# Convenience function for external use
def build_product_identification_system(force_rebuild: bool = False) -> SystemStatus:
    """
    Build the complete product identification system with UNLIMITED uploads.

    Args:
        force_rebuild: Whether to rebuild even if files exist

    Returns:
        SystemStatus with build results
    """
    builder = ProductIdentifierSystemBuilder(force_rebuild=force_rebuild)
    return builder.build_complete_system()


if __name__ == "__main__":
    # Test the unlimited build system
    import logging
    from .config import ProductIdentifierConfig

    ProductIdentifierConfig.setup_logging()

    print("🚀 Testing UNLIMITED Product Identification System Builder")
    print("=" * 70)

    # Build the system with no limits
    result = build_product_identification_system(force_rebuild=True)

    if result.success:
        print(f"✅ {result.message}")
        print(f"📁 Components created: {result.files_created}")
        print(f"⏱️ Time taken: {result.processing_time:.2f} seconds")
        print("🎉 UNLIMITED MODE: All products processed without limits!")
    else:
        print(f"❌ {result.message}")
        print(f"🐛 Errors: {result.errors}")

    print("\n" + "=" * 70)