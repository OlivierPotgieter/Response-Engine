"""
Configuration Management for Product Identifier
Handles environment variables and system settings
"""

import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ProductIdentifierConfig:
    """Configuration class for Product Identifier system"""

    # Database Configuration
    DB_CONFIG = {
        "host": os.getenv("BACKEND_DB_HOST", "localhost"),
        "database": os.getenv("BACKEND_DB_NAME", "product_db"),
        "user": os.getenv("BACKEND_DB_USER", "root"),
        "password": os.getenv("BACKEND_DB_PASSWORD", ""),
    }

    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
    EMBEDDING_DIMENSION = 3072  # text-embedding-3-large dimension

    # Pinecone Configuration
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp-free")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "product-identifier")

    # Product Intelligence Settings
    INTELLIGENCE_FILE_PATH = "product_intelligence.json"
    EMBEDDINGS_CACHE_PATH = "product_embeddings_cache.pkl"

    # Search Configuration
    SEARCH_CONFIG = {
        "top_k": 10,  # Number of results to retrieve
        "confidence_threshold": {"high": 0.8, "medium": 0.6, "low": 0.4},
        "max_products_per_comment": 5,
    }

    # Text Processing Configuration
    TEXT_PROCESSING = {
        "min_token_length": 2,
        "max_token_length": 50,
        "stop_words": [
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "is",
            "was",
            "are",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "cant",
            "wont",
            "dont",
            "isnt",
            "arent",
        ],
        "product_keywords": [
            "gpu",
            "graphics",
            "card",
            "cpu",
            "processor",
            "motherboard",
            "ram",
            "memory",
            "storage",
            "ssd",
            "hdd",
            "monitor",
            "display",
            "keyboard",
            "mouse",
            "headset",
            "speaker",
            "case",
            "power",
            "supply",
            "cooler",
            "fan",
            "rtx",
            "gtx",
            "radeon",
            "ryzen",
            "intel",
            "amd",
            "nvidia",
        ],
    }

    # Logging Configuration
    LOGGING_CONFIG = {
        "level": logging.INFO,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file_path": "product_identifier.log",
    }

    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """
        Validate configuration settings

        Returns:
            Dict with validation results
        """
        validation_results = {"valid": True, "errors": [], "warnings": []}

        # Check required API keys
        if not cls.OPENAI_API_KEY:
            validation_results["errors"].append(
                "OPENAI_API_KEY not found in environment"
            )
            validation_results["valid"] = False

        if not cls.PINECONE_API_KEY:
            validation_results["errors"].append(
                "PINECONE_API_KEY not found in environment"
            )
            validation_results["valid"] = False

        # Check database configuration
        required_db_fields = ["host", "database", "user"]
        for field in required_db_fields:
            if not cls.DB_CONFIG.get(field):
                validation_results["errors"].append(f"Database {field} not configured")
                validation_results["valid"] = False

        # Warnings for optional settings
        if not cls.DB_CONFIG.get("password"):
            validation_results["warnings"].append(
                "Database password not set - using empty password"
            )

        return validation_results

    @classmethod
    def setup_logging(cls):
        """Setup logging configuration"""
        logging.basicConfig(
            level=cls.LOGGING_CONFIG["level"],
            format=cls.LOGGING_CONFIG["format"],
            handlers=[
                logging.FileHandler(cls.LOGGING_CONFIG["file_path"]),
                logging.StreamHandler(),
            ],
        )

        # Set specific loggers
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("pinecone").setLevel(logging.WARNING)
        logging.getLogger("mysql.connector").setLevel(logging.WARNING)

    @classmethod
    def get_pinecone_config(cls) -> Dict[str, Any]:
        """Get Pinecone index configuration"""
        return {
            "dimension": cls.EMBEDDING_DIMENSION,
            "metric": "cosine",
            "metadata_config": {
                "indexed": [
                    "category",
                    "brand",
                    "brand_id",
                    "is_enabled",
                    "is_eol",
                    "price_range",
                    "product_type",
                ]
            },
        }

    @classmethod
    def get_embedding_config(cls) -> Dict[str, Any]:
        """Get embedding generation configuration"""
        return {
            "model": cls.OPENAI_EMBEDDING_MODEL,
            "dimension": cls.EMBEDDING_DIMENSION,
            "batch_size": 100,  # Process embeddings in batches
            "max_retries": 3,
            "retry_delay": 1.0,
        }


# Environment validation on import
if __name__ == "__main__":
    # Test configuration
    ProductIdentifierConfig.setup_logging()

    validation = ProductIdentifierConfig.validate_config()

    if validation["valid"]:
        print("✅ Configuration is valid!")

        if validation["warnings"]:
            print("\n⚠️ Warnings:")
            for warning in validation["warnings"]:
                print(f"  - {warning}")
    else:
        print("❌ Configuration errors found:")
        for error in validation["errors"]:
            print(f"  - {error}")

        print(
            "\n📝 Please check your .env file and ensure all required variables are set:"
        )
        print("OPENAI_API_KEY=your_openai_api_key")
        print("PINECONE_API_KEY=your_pinecone_api_key")
        print("PINECONE_ENVIRONMENT=your_pinecone_environment")
        print("BACKEND_DB_HOST=your_db_host")
        print("BACKEND_DB_NAME=your_db_name")
        print("BACKEND_DB_USER=your_db_user")
        print("BACKEND_DB_PASSWORD=your_db_password")
