#!/usr/bin/env python3
"""
Rebuild Pinecone Index Script
Rebuilds the Pinecone index with ALL products from the database
"""

import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def rebuild_pinecone_index():
    """Rebuild PRODUCT Pinecone index with all products from database"""

    print("🚀 Starting PRODUCT Pinecone Index Rebuild...")
    print("=" * 60)
    print("🎯 Target Index: product-identifier (HARDCODED)")
    print("⚠️  This will NOT touch your replies index!")
    print("=" * 60)

    try:
        # HARDCODED: Always use the AI product identification system
        # This ensures we rebuild the PRODUCT index, not the replies index
        from modules.product_identification import initialize_product_identification_system

        print("🔄 Initializing AI Product Identification System...")
        print("📊 Target: product-identifier index (for products)")
        print("⚠️  This will rebuild the entire PRODUCT Pinecone index with ALL 20,298 products")
        print("⏱️  This process may take 10-15 minutes")
        print("💰 This will use OpenAI API credits for embedding generation")
        print()

        # Ask for confirmation
        print("🛡️  SAFETY CHECK:")
        print("   ✅ Will rebuild: product-identifier (products)")
        print("   ❌ Will NOT touch: your replies index")
        print()
        response = input("Do you want to proceed with PRODUCT index rebuild? (yes/no): ").lower().strip()
        if response not in ['yes', 'y']:
            print("❌ Rebuild cancelled by user")
            return False

        print("\n🏗️ Starting PRODUCT index rebuild with force_rebuild=True...")
        start_time = datetime.now()

        # Force rebuild the PRODUCT identification system only
        result = initialize_product_identification_system(force_rebuild=True)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        if result.get('success'):
            print(f"\n✅ PRODUCT Pinecone index rebuild completed successfully!")
            print(f"⏱️  Total time: {duration:.1f} seconds ({duration / 60:.1f} minutes)")
            print(f"📁 Files created: {result.get('files_created', [])}")
            print(f"📊 Details: {result.get('details', {})}")

            # Test the rebuilt PRODUCT system
            print("\n🧪 Testing rebuilt PRODUCT system...")
            from modules.product_identification import search_comment_for_products

            test_result = search_comment_for_products("RTX 4090", max_results=3)
            if test_result.get('search_successful'):
                products_found = len(test_result.get('products_found', []))
                confidence = test_result.get('confidence', 0)
                print(f"✅ Test successful: Found {products_found} products with {confidence:.2f} confidence")

                if products_found > 0:
                    best_match = test_result.get('best_match', {})
                    print(f"🎯 Best match: {best_match.get('name', 'Unknown')}")
            else:
                print("❌ Test failed - system may need debugging")

            return True

        else:
            print(f"\n❌ Rebuild failed: {result.get('message', 'Unknown error')}")
            print(f"🐛 Errors: {result.get('errors', [])}")
            return False

    except Exception as e:
        print(f"\n💥 Rebuild script failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_current_status():
    """Check current PRODUCT Pinecone index status"""

    print("🔍 Checking Current PRODUCT Pinecone Index Status...")
    print("=" * 50)
    print("🎯 Checking: product-identifier (HARDCODED)")

    try:
        # HARDCODED: Check the product-identifier index directly
        from pinecone import Pinecone
        import os

        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index("product-identifier")

        # Get stats from the hardcoded product index
        stats = index.describe_index_stats()
        current_vectors = stats.total_vector_count

        print(f"PRODUCT Index Vectors: {current_vectors:,}")

        # Check database products
        from modules.product_identification.product_intelligence_builder import ProductIntelligenceBuilder
        builder = ProductIntelligenceBuilder()
        try:
            intelligence = builder.get_intelligence_from_database()
            database_products = len(intelligence.get('products', []))
            print(f"Database Products: {database_products:,}")

            # Calculate coverage
            if database_products > 0:
                coverage = (current_vectors / database_products) * 100
                print(f"PRODUCT Index Coverage: {coverage:.1f}%")

                if coverage < 90:
                    print("⚠️  WARNING: PRODUCT Pinecone index is significantly out of date!")
                    print("🔧 Recommendation: Rebuild the PRODUCT index")
                else:
                    print("✅ PRODUCT Index coverage looks good")

        finally:
            builder.close()

        return current_vectors, database_products

    except Exception as e:
        print(f"❌ Error checking status: {e}")
        return 0, 0


if __name__ == "__main__":
    print("🎯 PRODUCT Pinecone Index Rebuild Utility")
    print("=" * 60)
    print("🛡️  SAFETY: This script only touches product-identifier index")
    print("🛡️  SAFETY: Your replies index will NOT be affected")
    print("=" * 60)

    # Check current status first
    current_vectors, database_products = check_current_status()

    if current_vectors > 0 and database_products > 0:
        coverage = (current_vectors / database_products) * 100

        if coverage >= 90:
            print(f"\n✅ Your PRODUCT index looks up to date ({coverage:.1f}% coverage)")
            response = input("Do you still want to rebuild the PRODUCT index? (yes/no): ").lower().strip()
            if response not in ['yes', 'y']:
                print("👍 Keeping existing PRODUCT index")
                sys.exit(0)

    print(f"\n🔧 Your PRODUCT index needs rebuilding:")
    print(f"   Current: {current_vectors:,} vectors")
    print(f"   Should be: {database_products:,} vectors")
    print(f"   Missing: {database_products - current_vectors:,} vectors")

    # Perform the rebuild
    success = rebuild_pinecone_index()

    if success:
        print("\n🎉 PRODUCT Index rebuild completed successfully!")
        print("🚀 Your AI product identification system is now up to date!")
        print("🛡️  Your replies index was not touched")
        sys.exit(0)
    else:
        print("\n💥 PRODUCT Index rebuild failed!")
        print("🔧 Check the logs above for error details")
        sys.exit(1)