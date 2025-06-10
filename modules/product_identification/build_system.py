"""
Product Identifier System Builder
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
    This replaces the need for manual script running.
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

        Returns:
            SystemStatus with build results
        """
        start_time = time.time()

        logger.info("🚀 Starting Product Identification System Build...")

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

            # Step 3: Build product intelligence
            if not self._build_product_intelligence():
                return self._build_error_status(start_time, "product_intelligence")

            # Step 4: Build category intelligence
            if not self._build_category_intelligence():
                return self._build_error_status(start_time, "category_intelligence")

            # Step 5: Generate and upload embeddings
            if not self._build_embeddings_system():
                return self._build_error_status(start_time, "embeddings_system")

            # Step 6: Verify system is working
            if not self._verify_system():
                return self._build_error_status(start_time, "system_verification")

            processing_time = time.time() - start_time

            logger.info(
                f"✅ System build completed successfully in {processing_time:.2f} seconds"
            )

            return SystemStatus(
                success=True,
                message=f"Product identification system built successfully in {processing_time:.2f}s",
                details={
                    "step": "completed",
                    "files_created": self.files_created,
                    "processing_time": processing_time,
                },
                processing_time=processing_time,
                files_created=self.files_created,
                errors=[],
            )

        except Exception as e:
            logger.error(f"❌ System build failed: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")

            return SystemStatus(
                success=False,
                message=f"System build failed: {str(e)}",
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
            "product_embeddings_cache.pkl",
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
            "product_embeddings_cache.pkl",
        ]

        return [f for f in required_files if os.path.exists(f)]

    def _build_product_intelligence(self) -> bool:
        """Build product intelligence from database"""
        try:
            logger.info("🧠 Building product intelligence from database...")

            from .product_intelligence_builder import build_product_intelligence

            intelligence = build_product_intelligence(save_to_file=True)

            if intelligence and "error" not in intelligence:
                self.files_created.append("product_intelligence.json")
                logger.info("✅ Product intelligence built successfully")
                return True
            else:
                self.errors.append("Failed to build product intelligence")
                return False

        except Exception as e:
            self.errors.append(f"Product intelligence build error: {e}")
            logger.error(f"❌ Product intelligence build failed: {e}")
            return False

    def _build_category_intelligence(self) -> bool:
        """Build category intelligence (this would need to be implemented)"""
        try:
            logger.info("📂 Building category intelligence...")

            # For now, create a simple category intelligence file
            # This could be enhanced later with actual category analysis
            category_intelligence = {
                "categories": {
                    "Graphics Cards": {
                        "keywords": [
                            "gpu",
                            "graphics",
                            "video card",
                            "rtx",
                            "gtx",
                            "radeon",
                        ],
                        "patterns": ["rtx \\d{4}", "gtx \\d{4}", "radeon rx \\d{4}"],
                        "brands": ["nvidia", "amd", "asus", "msi", "gigabyte"],
                    },
                    "Processors": {
                        "keywords": ["cpu", "processor", "ryzen", "intel", "core"],
                        "patterns": [
                            "ryzen \\d \\d{4}",
                            "core i[3579]-\\d{4,5}",
                            "intel core",
                        ],
                        "brands": ["amd", "intel"],
                    },
                    "Memory": {
                        "keywords": ["ram", "memory", "ddr4", "ddr5"],
                        "patterns": ["\\d+gb ddr[45]", "ddr[45]-\\d{4}"],
                        "brands": ["corsair", "kingston", "crucial", "gskill"],
                    },
                    "Storage": {
                        "keywords": ["ssd", "hdd", "nvme", "storage", "drive"],
                        "patterns": ["\\d+gb", "\\d+tb", "nvme", "sata"],
                        "brands": ["samsung", "western digital", "seagate", "crucial"],
                    },
                },
                "created": datetime.now().isoformat(),
                "note": "Simple category intelligence - can be enhanced with ML analysis",
            }

            import json

            with open("category_intelligence.json", "w") as f:
                json.dump(category_intelligence, f, indent=2)

            self.files_created.append("category_intelligence.json")
            logger.info("✅ Category intelligence built successfully")
            return True

        except Exception as e:
            self.errors.append(f"Category intelligence build error: {e}")
            logger.error(f"❌ Category intelligence build failed: {e}")
            return False

    def _build_embeddings_system(self) -> bool:
        """Build embeddings and upload to Pinecone"""
        try:
            logger.info("🔗 Building embeddings system...")

            # Step 1: Load product intelligence
            import json

            with open("product_intelligence.json", "r") as f:
                intelligence = json.load(f)

            products_data = intelligence.get("products", [])
            if not products_data:
                self.errors.append("No products found in intelligence data")
                return False

            logger.info(
                f"📊 Found {len(products_data)} products for embedding generation"
            )

            # Step 2: Initialize services
            from .embedding_service import EmbeddingService
            from .pinecone_manager import PineconeManager

            embedding_service = EmbeddingService()
            pinecone_manager = PineconeManager()

            # Step 3: Create or connect to Pinecone index
            if not pinecone_manager.create_index(force_recreate=self.force_rebuild):
                self.errors.append("Failed to create Pinecone index")
                return False

            # Step 4: Generate embeddings
            logger.info("🧬 Generating embeddings for products...")
            product_embeddings = embedding_service.create_product_embeddings(
                products_data
            )

            if not product_embeddings:
                self.errors.append("Failed to generate product embeddings")
                return False

            logger.info(f"✅ Generated {len(product_embeddings)} product embeddings")

            # Step 5: Upload to Pinecone
            logger.info("☁️ Uploading embeddings to Pinecone...")
            if not pinecone_manager.upload_product_embeddings(product_embeddings):
                self.errors.append("Failed to upload embeddings to Pinecone")
                return False

            # Step 6: Save embeddings cache
            embedding_service.save_embeddings_cache()
            self.files_created.append("product_embeddings_cache.pkl")

            logger.info("✅ Embeddings system built successfully")
            return True

        except Exception as e:
            self.errors.append(f"Embeddings system build error: {e}")
            logger.error(f"❌ Embeddings system build failed: {e}")
            return False

    def _verify_system(self) -> bool:
        """Verify the complete system is working"""
        try:
            logger.info("🔍 Verifying system functionality...")

            # Try to initialize the smart extractor
            from .smart_product_extractor import SmartProductExtractor

            extractor = SmartProductExtractor()

            # Test with a simple query
            test_result = extractor.extract_products_smart("RTX 4090", max_products=1)

            if test_result.products or test_result.extraction_method != "failed":
                logger.info("✅ System verification passed")
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
            message=f"System build failed at step: {failed_step}",
            details={"step": failed_step, "errors": self.errors},
            processing_time=time.time() - start_time,
            files_created=self.files_created,
            errors=self.errors,
        )


# Convenience function for external use
def build_product_identification_system(force_rebuild: bool = False) -> SystemStatus:
    """
    Build the complete product identification system.

    Args:
        force_rebuild: Whether to rebuild even if files exist

    Returns:
        SystemStatus with build results
    """
    builder = ProductIdentifierSystemBuilder(force_rebuild=force_rebuild)
    return builder.build_complete_system()


if __name__ == "__main__":
    # Test the build system
    import logging
    from .config import ProductIdentifierConfig

    ProductIdentifierConfig.setup_logging()

    print("🚀 Testing Product Identification System Builder")
    print("=" * 60)

    # Build the system
    result = build_product_identification_system(force_rebuild=True)

    if result.success:
        print(f"✅ {result.message}")
        print(f"📁 Files created: {result.files_created}")
        print(f"⏱️ Time taken: {result.processing_time:.2f} seconds")
    else:
        print(f"❌ {result.message}")
        print(f"🐛 Errors: {result.errors}")

    print("\n" + "=" * 60)
