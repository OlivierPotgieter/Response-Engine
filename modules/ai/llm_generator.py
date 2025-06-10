"""
LLM Response Generation Module - Enhanced with Real-Time Data Prioritization and Product Viability Logic
Handles response generation using OpenAI's language models with smart prompt engineering for pricing/stock queries and EOL detection.
"""

import os
import logging
import re
from typing import Dict, List
from openai import OpenAI
from datetime import datetime
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

try:
    from product_identification.confidence import (
        ConfidenceThresholds,
        get_confidence_level,
    )
except ImportError:
    # Fallback implementations
    class ConfidenceThresholds:
        HIGH_CONFIDENCE = 0.85
        MEDIUM_CONFIDENCE = 0.70
        LOW_CONFIDENCE = 0.50
        MINIMUM_THRESHOLD = 0.50

    def get_confidence_level(confidence_score):
        if confidence_score >= ConfidenceThresholds.HIGH_CONFIDENCE:
            return "high"
        elif confidence_score >= ConfidenceThresholds.MEDIUM_CONFIDENCE:
            return "medium"
        elif confidence_score >= ConfidenceThresholds.LOW_CONFIDENCE:
            return "low"
        else:
            return "insufficient"


class LLMGenerator:
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.3):
        """
        Initialize the LLM generator with enhanced real-time data capabilities and product viability logic

        Args:
            model: OpenAI model to use for generation
            temperature: Temperature setting for generation
        """
        self.model = model
        self.temperature = temperature
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            logger.info(
                f"Enhanced LLM generator with product viability logic initialized with model: {self.model}"
            )
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            raise

    def _append_rep_signature(self, response: str, woot_rep: str) -> str:
        """
        Append the Wootware rep's signature to the response

        Args:
            response: Generated response text
            woot_rep: The rep's name from the database

        Returns:
            Response with rep signature appended
        """
        if not woot_rep or not woot_rep.strip():
            # Fallback to generic signature if no rep name
            return f"{response}\n\nKind Regards,\nWootware Sales Team"

        # Clean the response - remove any existing signatures
        cleaned_response = response.strip()

        # Remove common ending patterns that might conflict
        ending_patterns = [
            r"\n\nKind [Rr]egards,?.*?$",
            r"\n\nBest [Rr]egards,?.*?$",
            r"\n\nSincerely,?.*?$",
            r"\n\nCheers,?.*?$",
            r"\n\nThank you,?.*?$",
        ]

        for pattern in ending_patterns:
            cleaned_response = re.sub(pattern, "", cleaned_response, flags=re.DOTALL)

        # Append the rep's signature
        return f"{cleaned_response.strip()}\n\nKind Regards,\n{woot_rep}"

    def _analyze_product_viability_for_prompt(self, context: Dict) -> Dict:
        """
        Analyze product viability to determine prompt strategy

        Args:
            context: Enhanced context with product selection

        Returns:
            Dict with viability analysis for prompt building
        """
        analysis = {
            "has_viable_product": False,
            "primary_product_viable": False,
            "secondary_product_viable": False,
            "should_provide_pricing": False,
            "should_provide_stock": False,
            "alternative_recommended": False,
            "eol_warning_needed": False,
            "primary_product": None,
            "secondary_product": None,
        }

        # Get product selection info
        product_selection = context.get("product_selection", {})
        real_time_data = context.get("real_time_data", {})

        primary_product_data = real_time_data.get("primary_product", {})
        secondary_product_data = real_time_data.get("secondary_product", {})

        # Analyze primary product
        if primary_product_data:
            analysis["primary_product"] = primary_product_data
            analysis["primary_product_viable"] = primary_product_data.get(
                "is_viable", False
            )
            analysis["has_viable_product"] = analysis["primary_product_viable"]

            if analysis["primary_product_viable"]:
                analysis["should_provide_pricing"] = bool(
                    primary_product_data.get("pricing")
                )
                analysis["should_provide_stock"] = bool(
                    primary_product_data.get("stock")
                )

        # Analyze secondary product
        if secondary_product_data:
            analysis["secondary_product"] = secondary_product_data
            analysis["secondary_product_viable"] = secondary_product_data.get(
                "is_viable", False
            )

        # Check if alternative was recommended
        analysis["alternative_recommended"] = product_selection.get(
            "has_alternative_recommendation", False
        )

        # Check if we need EOL warning
        selection_reason = product_selection.get("selection_reason", "")
        if (
            "not viable" in selection_reason.lower()
            or "eol" in selection_reason.lower()
        ):
            analysis["eol_warning_needed"] = True

        logger.info(
            f"Product viability analysis: viable={analysis['has_viable_product']}, "
            f"pricing={analysis['should_provide_pricing']}, stock={analysis['should_provide_stock']}, "
            f"alt_rec={analysis['alternative_recommended']}, eol_warn={analysis['eol_warning_needed']}"
        )

        return analysis

    def _build_pricing_prompt_with_viability(
        self, customer_comment: str, context: Dict
    ) -> str:
        """
        Build specialized prompt for pricing-related queries with product viability logic

        Args:
            customer_comment: Customer's comment
            context: Enhanced context with real-time data and viability

        Returns:
            Specialized pricing prompt with viability considerations
        """
        viability = self._analyze_product_viability_for_prompt(context)

        # ENHANCED: Add confidence-based language guidance
        confidence = context.get("product_search_result", {}).get("confidence", 0)
        confidence_level = get_confidence_level(confidence)

        confidence_language_guide = {
            "high": "Use authoritative language: 'The [product] is currently priced at...'",
            "medium": "Use qualified language: 'Based on your description, the [product] appears to be...'",
            "low": "Use exploratory language: 'I found several products that might match. Could you confirm...'",
            "insufficient": "Provide general guidance and ask for clarification",
        }

        prompt_parts = [
            "You are Wootware's expert sales assistant responding to a PRICING INQUIRY.",
            "",
            f"🎯 CONFIDENCE GUIDANCE ({confidence_level.upper()} confidence: {confidence:.2f}):",
            confidence_language_guide.get(
                confidence_level, confidence_language_guide["insufficient"]
            ),
            "",
            "⚠️ CRITICAL: Use ONLY the current pricing data provided below. DO NOT use any pricing from examples as it may be outdated.",
            "",
            f"CUSTOMER INQUIRY: {customer_comment}",
            "",
        ]

        if viability["has_viable_product"] and viability["should_provide_pricing"]:
            # We have viable product with pricing data
            primary_product = viability["primary_product"]
            pricing = primary_product.get("pricing", {})

            prompt_parts.extend(
                [
                    "📊 CURRENT PRICING DATA (Use this authoritative data):",
                    f"• Product: {primary_product.get('product_name', 'N/A')}",
                    f"• SKU: {primary_product.get('sku', 'N/A')}",
                    f"• Current Price: R{pricing.get('current_price', 'TBD')}",
                ]
            )

            if pricing.get("is_on_promotion"):
                special_price = pricing.get("special_price")
                if special_price:
                    prompt_parts.append(
                        f"• 🔥 Special Price: R{special_price} (PROMOTION ACTIVE)"
                    )

            if pricing.get("price_updated"):
                prompt_parts.append(f"• Last Updated: {pricing.get('price_updated')}")

            prompt_parts.append("")

            # Add secondary product if available
            if viability["secondary_product_viable"]:
                secondary_product = viability["secondary_product"]
                secondary_pricing = secondary_product.get("pricing", {})
                if secondary_pricing:
                    prompt_parts.extend(
                        [
                            "📋 ALTERNATIVE OPTION:",
                            f"• Product: {secondary_product.get('product_name', 'N/A')}",
                            f"• Current Price: R{secondary_pricing.get('current_price', 'TBD')}",
                            "",
                        ]
                    )

        else:
            # No viable product or pricing data
            prompt_parts.extend(
                [
                    "⚠️ PRODUCT AVAILABILITY NOTICE:",
                    "The requested product is no longer available or has been discontinued.",
                    "",
                ]
            )

            if (
                viability["alternative_recommended"]
                and viability["secondary_product_viable"]
            ):
                secondary_product = viability["secondary_product"]
                secondary_pricing = secondary_product.get("pricing", {})
                if secondary_pricing:
                    prompt_parts.extend(
                        [
                            "📋 RECOMMENDED ALTERNATIVE:",
                            f"• Product: {secondary_product.get('product_name', 'N/A')}",
                            f"• Current Price: R{secondary_pricing.get('current_price', 'TBD')}",
                            "",
                        ]
                    )

        # Add minimal example for tone only
        examples = context.get("examples", [])
        if examples:
            example = examples[0]
            prompt_parts.extend(
                [
                    "📝 TONE REFERENCE (DO NOT copy pricing from this):",
                    f"Example response style: {example.get('original_reply', '')[:150]}...",
                    "",
                    "⚠️ WARNING: Ignore any pricing mentioned in the above example - use only current data provided above.",
                    "",
                ]
            )

        if viability["has_viable_product"] and viability["should_provide_pricing"]:
            prompt_parts.extend(
                [
                    "RESPONSE GUIDELINES:",
                    "1. ✅ Use ONLY the current pricing data provided above",
                    "2. ❌ NEVER use pricing from examples - they may be outdated",
                    "3. Include product name and SKU for clarity",
                    "4. Mention if item is on promotion with special pricing",
                    "5. Be helpful and professional in Wootware's tone",
                    "6. Do NOT include signature - it will be added automatically",
                    "",
                ]
            )
        else:
            prompt_parts.extend(
                [
                    "RESPONSE GUIDELINES:",
                    "1. ❌ DO NOT provide pricing for discontinued/unavailable products",
                    "2. Explain that the original product is no longer available",
                    "3. Suggest the recommended alternative if available",
                    "4. Offer to help find similar products if needed",
                    "5. Be helpful and professional in Wootware's tone",
                    "6. Do NOT include signature - it will be added automatically",
                    "",
                ]
            )

        real_time_data = context.get("real_time_data", {})
        prompt_parts.extend(
            [
                f"Data freshness: {real_time_data.get('data_freshness', 'Unknown')}",
                "",
                "Generate your pricing response now:",
            ]
        )

        return "\n".join(prompt_parts)

    def _build_stock_prompt_with_viability(
        self, customer_comment: str, context: Dict
    ) -> str:
        """
        Build specialized prompt for stock availability queries with product viability logic

        Args:
            customer_comment: Customer's comment
            context: Enhanced context with real-time data and viability

        Returns:
            Specialized stock prompt with viability considerations
        """
        viability = self._analyze_product_viability_for_prompt(context)

        prompt_parts = [
            "You are Wootware's expert sales assistant responding to a STOCK AVAILABILITY INQUIRY.",
            "",
            "⚠️ CRITICAL: Use ONLY the current stock data provided below. DO NOT use any availability info from examples as it may be outdated.",
            "",
            f"CUSTOMER INQUIRY: {customer_comment}",
            "",
        ]

        if viability["has_viable_product"] and viability["should_provide_stock"]:
            # We have viable product with stock data
            primary_product = viability["primary_product"]
            stock = primary_product.get("stock", {})

            prompt_parts.extend(
                [
                    "📦 CURRENT STOCK STATUS (Use this authoritative data):",
                    f"• Product: {primary_product.get('product_name', 'N/A')}",
                    f"• SKU: {primary_product.get('sku', 'N/A')}",
                    f"• Stock Status: {'✅ IN STOCK' if stock.get('is_in_stock') else '❌ OUT OF STOCK'}",
                ]
            )

            if stock.get("is_in_stock"):
                quantity = stock.get("stock_quantity", 0)
                if quantity > 0:
                    prompt_parts.append(f"• Available Quantity: {quantity}")

                dispatch = stock.get("expected_dispatch")
                if dispatch:
                    prompt_parts.append(f"• Expected Dispatch: {dispatch}")
            else:
                # Out of stock - provide lead time and ETA
                lead_time = stock.get("lead_time")
                if lead_time:
                    prompt_parts.append(f"• Lead Time: {lead_time}")

                eta = stock.get("eta")
                if eta:
                    prompt_parts.append(f"• Estimated Arrival: {eta}")

            if stock.get("stock_updated"):
                prompt_parts.append(f"• Last Updated: {stock.get('stock_updated')}")

            prompt_parts.append("")

            # Add secondary product if available
            if viability["secondary_product_viable"]:
                secondary_product = viability["secondary_product"]
                secondary_stock = secondary_product.get("stock", {})
                if secondary_stock:
                    prompt_parts.extend(
                        [
                            "📋 ALTERNATIVE OPTION:",
                            f"• Product: {secondary_product.get('product_name', 'N/A')}",
                            f"• Status: {'✅ IN STOCK' if secondary_stock.get('is_in_stock') else '❌ OUT OF STOCK'}",
                            "",
                        ]
                    )

        else:
            # No viable product
            prompt_parts.extend(
                [
                    "⚠️ PRODUCT AVAILABILITY NOTICE:",
                    "The requested product is no longer available or has been discontinued.",
                    "",
                ]
            )

            if (
                viability["alternative_recommended"]
                and viability["secondary_product_viable"]
            ):
                secondary_product = viability["secondary_product"]
                secondary_stock = secondary_product.get("stock", {})
                if secondary_stock:
                    prompt_parts.extend(
                        [
                            "📋 RECOMMENDED ALTERNATIVE:",
                            f"• Product: {secondary_product.get('product_name', 'N/A')}",
                            f"• Status: {'✅ IN STOCK' if secondary_stock.get('is_in_stock') else '❌ OUT OF STOCK'}",
                            "",
                        ]
                    )

        # Add minimal example for tone only
        examples = context.get("examples", [])
        if examples:
            example = examples[0]
            prompt_parts.extend(
                [
                    "📝 TONE REFERENCE (DO NOT copy stock info from this):",
                    f"Example response style: {example.get('original_reply', '')[:150]}...",
                    "",
                    "⚠️ WARNING: Ignore any stock/availability mentioned in the above example - use only current data provided above.",
                    "",
                ]
            )

        if viability["has_viable_product"] and viability["should_provide_stock"]:
            prompt_parts.extend(
                [
                    "RESPONSE GUIDELINES:",
                    "1. ✅ Use ONLY the current stock data provided above",
                    "2. ❌ NEVER use availability info from examples - they may be outdated",
                    "3. Clearly state if item is in stock or out of stock",
                    "4. Provide lead times and ETAs for out-of-stock items",
                    "5. Mention dispatch timeframes for in-stock items",
                    "6. Suggest alternatives if main product is unavailable",
                    "7. Do NOT include signature - it will be added automatically",
                    "",
                ]
            )
        else:
            prompt_parts.extend(
                [
                    "RESPONSE GUIDELINES:",
                    "1. ❌ DO NOT provide stock info for discontinued/unavailable products",
                    "2. Explain that the original product is no longer available",
                    "3. Suggest the recommended alternative if available",
                    "4. Offer to help find similar products if needed",
                    "5. Be helpful and professional in Wootware's tone",
                    "6. Do NOT include signature - it will be added automatically",
                    "",
                ]
            )

        real_time_data = context.get("real_time_data", {})
        prompt_parts.extend(
            [
                f"Data freshness: {real_time_data.get('data_freshness', 'Unknown')}",
                "",
                "Generate your stock availability response now:",
            ]
        )

        return "\n".join(prompt_parts)

    def _build_combined_prompt_with_viability(
        self, customer_comment: str, context: Dict
    ) -> str:
        """
        Build prompt for queries needing both pricing and stock data with viability logic

        Args:
            customer_comment: Customer's comment
            context: Enhanced context with real-time data and viability

        Returns:
            Combined pricing and stock prompt with viability considerations
        """
        viability = self._analyze_product_viability_for_prompt(context)

        prompt_parts = [
            "You are Wootware's expert sales assistant responding to an inquiry about BOTH PRICING AND AVAILABILITY.",
            "",
            "⚠️ CRITICAL: Use ONLY the current data provided below. DO NOT use any pricing or stock info from examples as it may be outdated.",
            "",
            f"CUSTOMER INQUIRY: {customer_comment}",
            "",
        ]

        if viability["has_viable_product"]:
            # We have viable product
            primary_product = viability["primary_product"]

            prompt_parts.extend(
                [
                    "📊 CURRENT PRODUCT INFORMATION (Use this authoritative data):",
                    f"• Product: {primary_product.get('product_name', 'N/A')}",
                    f"• SKU: {primary_product.get('sku', 'N/A')}",
                    "",
                ]
            )

            # Pricing section
            if viability["should_provide_pricing"]:
                pricing = primary_product.get("pricing", {})
                prompt_parts.append("💰 PRICING:")
                prompt_parts.append(
                    f"  • Current Price: R{pricing.get('current_price', 'TBD')}"
                )

                if pricing.get("is_on_promotion"):
                    special_price = pricing.get("special_price")
                    if special_price:
                        prompt_parts.append(
                            f"  • 🔥 Special Price: R{special_price} (PROMOTION ACTIVE)"
                        )

                prompt_parts.append("")

            # Stock section
            if viability["should_provide_stock"]:
                stock = primary_product.get("stock", {})
                prompt_parts.append("📦 AVAILABILITY:")
                prompt_parts.append(
                    f"  • Stock Status: {'✅ IN STOCK' if stock.get('is_in_stock') else '❌ OUT OF STOCK'}"
                )

                if stock.get("is_in_stock"):
                    dispatch = stock.get("expected_dispatch")
                    if dispatch:
                        prompt_parts.append(f"  • Expected Dispatch: {dispatch}")
                else:
                    lead_time = stock.get("lead_time")
                    if lead_time:
                        prompt_parts.append(f"  • Lead Time: {lead_time}")

                prompt_parts.append("")

            # Add secondary product if available
            if viability["secondary_product_viable"]:
                secondary_product = viability["secondary_product"]
                prompt_parts.extend(
                    [
                        "📋 ALTERNATIVE OPTION:",
                        f"• Product: {secondary_product.get('product_name', 'N/A')}",
                    ]
                )

                if viability["should_provide_pricing"]:
                    secondary_pricing = secondary_product.get("pricing", {})
                    if secondary_pricing:
                        prompt_parts.append(
                            f"• Price: R{secondary_pricing.get('current_price', 'TBD')}"
                        )

                if viability["should_provide_stock"]:
                    secondary_stock = secondary_product.get("stock", {})
                    if secondary_stock:
                        prompt_parts.append(
                            f"• Availability: {'✅ IN STOCK' if secondary_stock.get('is_in_stock') else '❌ OUT OF STOCK'}"
                        )

                prompt_parts.append("")

        else:
            # No viable product
            prompt_parts.extend(
                [
                    "⚠️ PRODUCT AVAILABILITY NOTICE:",
                    "The requested product is no longer available or has been discontinued.",
                    "",
                ]
            )

            if (
                viability["alternative_recommended"]
                and viability["secondary_product_viable"]
            ):
                secondary_product = viability["secondary_product"]
                prompt_parts.extend(
                    [
                        "📋 RECOMMENDED ALTERNATIVE:",
                        f"• Product: {secondary_product.get('product_name', 'N/A')}",
                    ]
                )

                secondary_pricing = secondary_product.get("pricing", {})
                if secondary_pricing:
                    prompt_parts.append(
                        f"• Price: R{secondary_pricing.get('current_price', 'TBD')}"
                    )

                secondary_stock = secondary_product.get("stock", {})
                if secondary_stock:
                    prompt_parts.append(
                        f"• Availability: {'✅ IN STOCK' if secondary_stock.get('is_in_stock') else '❌ OUT OF STOCK'}"
                    )

                prompt_parts.append("")

        # Add minimal example for tone only
        examples = context.get("examples", [])
        if examples:
            example = examples[0]
            prompt_parts.extend(
                [
                    "📝 TONE REFERENCE (DO NOT copy any data from this):",
                    f"Example response style: {example.get('original_reply', '')[:150]}...",
                    "",
                    "⚠️ WARNING: Ignore any pricing or stock info in the above example - use only current data provided above.",
                    "",
                ]
            )

        if viability["has_viable_product"]:
            prompt_parts.extend(
                [
                    "RESPONSE GUIDELINES:",
                    "1. ✅ Use ONLY the current data provided above",
                    "2. ❌ NEVER use pricing or availability info from examples",
                    "3. Address both pricing and availability in your response",
                    "4. Mention promotions if active",
                    "5. Provide lead times for out-of-stock items",
                    "6. Suggest alternatives if main product is unavailable",
                    "7. Do NOT include signature - it will be added automatically",
                    "",
                ]
            )
        else:
            prompt_parts.extend(
                [
                    "RESPONSE GUIDELINES:",
                    "1. ❌ DO NOT provide pricing or stock info for discontinued products",
                    "2. Explain that the original product is no longer available",
                    "3. Focus on the recommended alternative if available",
                    "4. Offer to help find similar products if needed",
                    "5. Be helpful and professional in Wootware's tone",
                    "6. Do NOT include signature - it will be added automatically",
                    "",
                ]
            )

        real_time_data = context.get("real_time_data", {})
        prompt_parts.extend(
            [
                f"Data freshness: {real_time_data.get('data_freshness', 'Unknown')}",
                "",
                "Generate your comprehensive response now:",
            ]
        )

        return "\n".join(prompt_parts)

    def generate_response_from_examples(
        self, customer_comment: str, examples: List[Dict], woot_rep: str = None
    ) -> str:
        """
        Generate response using examples (matching your 6-2 script approach)

        Args:
            customer_comment: The customer's comment
            examples: List of response examples from Pinecone search
            woot_rep: The rep's name to append to the response

        Returns:
            Generated response text with rep signature
        """
        try:
            logger.info(f"Generating response for comment: {customer_comment[:100]}...")
            logger.info(f"Using {len(examples)} examples, Rep: {woot_rep}")

            # Build messages exactly like your 6-2 script
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful Wootware sales assistant.",
                }
            ]

            # Add few-shot examples from Pinecone matches
            for i, example in enumerate(examples[:3], start=1):
                original_comment = example.get("original_comment", "")
                original_reply = example.get("original_reply", "")

                # Only add examples that have both comment and reply
                if (
                    isinstance(original_comment, str)
                    and original_comment.strip()
                    and isinstance(original_reply, str)
                    and original_reply.strip()
                ):

                    messages.append(
                        {
                            "role": "user",
                            "content": f"Example {i}:\nCustomer: {original_comment}\nAgent: {original_reply}",
                        }
                    )
                    logger.info(
                        f"Added example {i} with similarity {example.get('similarity_score', 0):.3f}"
                    )

            # Add the actual customer comment
            messages.append(
                {
                    "role": "user",
                    "content": f"Now reply to this:\nCustomer: {customer_comment}\nAgent:",
                }
            )

            logger.info(f"Sending {len(messages)} messages to OpenAI...")

            # 6-2 Call
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature
            )

            generated_reply = response.choices[0].message.content.strip()

            # Append rep signature
            final_reply = self._append_rep_signature(generated_reply, woot_rep)

            logger.info(
                f"Generated response length: {len(final_reply)} (with {woot_rep} signature)"
            )

            return final_reply

        except Exception as e:
            logger.error(f"Response generation error: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            fallback_response = "I apologize, but I'm having trouble generating a response right now. Please contact our sales team directly."
            return self._append_rep_signature(fallback_response, woot_rep)

    def generate_enhanced_response(
        self, customer_comment: str, context: Dict, woot_rep: str = None
    ) -> str:
        """
        Generate response with INTENT-BASED prompt selection and graceful fallback

        Args:
            customer_comment: The customer's comment
            context: Dict with enhanced context including real-time data, prioritization, and viability
            woot_rep: The rep's name to append to the response

        Returns:
            Generated response text with rep signature
        """
        try:
            logger.info(
                f"Generating enhanced response for: {customer_comment[:100]}..."
            )
            logger.info(f"Rep: {woot_rep}")

            # FIXED: Use intent-based prompt selection (not keyword-based)
            predicted_intent = context.get("predicted_intent", "General Inquiry")
            prompt_strategy = context.get("data_needs", {}).get(
                "prompt_strategy", "general_helpful"
            )

            logger.info(f"Intent: {predicted_intent}, Strategy: {prompt_strategy}")

            # INTENT-BASED prompt selection
            if predicted_intent == "Warranty Inquiry":
                enhanced_prompt = self._build_warranty_prompt(customer_comment, context)
                prompt_type = "warranty_focused"

            elif predicted_intent == "Pricing Inquiry":
                enhanced_prompt = self._build_pricing_prompt_with_viability(
                    customer_comment, context
                )
                prompt_type = "pricing_focused"

            elif predicted_intent == "Stock Availability":
                enhanced_prompt = self._build_stock_prompt_with_viability(
                    customer_comment, context
                )
                prompt_type = "stock_focused"

            elif predicted_intent == "Quotation Request":
                enhanced_prompt = self._build_combined_prompt_with_viability(
                    customer_comment, context
                )
                prompt_type = "quotation_focused"

            else:
                # For General Inquiry, Order Assistance, etc.
                enhanced_prompt = self._build_standard_enhanced_prompt(
                    customer_comment, context
                )
                prompt_type = "general_helpful"

            logger.info(f"Using {prompt_type} prompt for intent: {predicted_intent}")

            # Generate response using selected prompt
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are Wootware's expert sales assistant. Provide helpful, accurate responses based on the given context and instructions. Always follow the data usage guidelines provided.",
                    },
                    {"role": "user", "content": enhanced_prompt},
                ],
                temperature=self.temperature,
                max_tokens=1000,
            )

            generated_reply = response.choices[0].message.content.strip()

            # CRITICAL: Check for minimal responses and fall back if needed
            if self._is_response_too_minimal(generated_reply):
                logger.warning(
                    f"Enhanced response too minimal ({len(generated_reply)} chars), falling back to examples"
                )
                examples = context.get("examples", [])
                if examples:
                    return self.generate_response_from_examples(
                        customer_comment, examples, woot_rep
                    )
                else:
                    logger.warning("No examples available, using simple generation")
                    return self.generate_simple_response(
                        customer_comment, predicted_intent, woot_rep
                    )

            # Append rep signature
            final_reply = self._append_rep_signature(generated_reply, woot_rep)

            logger.info(
                f"Generated enhanced response length: {len(final_reply)} using {prompt_type} (with {woot_rep} signature)"
            )

            return final_reply

        except Exception as e:
            logger.error(f"Enhanced response generation error: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")

            # GRACEFUL FALLBACK: Try examples first, then simple
            try:
                logger.info("Falling back to examples-based generation")
                examples = context.get("examples", [])
                if examples:
                    return self.generate_response_from_examples(
                        customer_comment, examples, woot_rep
                    )
                else:
                    logger.info(
                        "No examples available, falling back to simple generation"
                    )
                    predicted_intent = context.get(
                        "predicted_intent", "General Inquiry"
                    )
                    return self.generate_simple_response(
                        customer_comment, predicted_intent, woot_rep
                    )
            except Exception as fallback_error:
                logger.error(f"Fallback generation also failed: {fallback_error}")
                fallback_response = "I apologize, but I'm having trouble generating a response right now. Please contact our sales team directly."
                return self._append_rep_signature(fallback_response, woot_rep)

    def _is_response_too_minimal(self, response: str) -> bool:
        """
        Check if response is too minimal (like the "Hey there, Kind Regards" issue)

        Args:
            response: Generated response text

        Returns:
            True if response seems too minimal
        """
        # Remove signature and common greetings to check actual content
        cleaned_response = response.strip()

        # Remove common patterns
        patterns_to_remove = [
            r"hey there,?\s*",
            r"hi there,?\s*",
            r"hello,?\s*",
            r"good morning,?\s*",
            r"good afternoon,?\s*",
            r"thanks?\s*for\s*reaching\s*out,?\s*",
            r"kind\s*regards,?\s*\w*",
            r"best\s*regards,?\s*\w*",
            r"sincerely,?\s*\w*",
        ]

        for pattern in patterns_to_remove:
            cleaned_response = re.sub(
                pattern, "", cleaned_response, flags=re.IGNORECASE
            )

        # Check if there's substantial content left
        substantial_content = cleaned_response.strip()

        # Consider minimal if:
        # 1. Less than 50 characters of actual content
        # 2. Contains mostly whitespace or punctuation
        # 3. Has very few actual words

        if len(substantial_content) < 50:
            return True

        word_count = len(
            [word for word in substantial_content.split() if len(word) > 2]
        )
        if word_count < 10:
            return True

        return False

    def _build_warranty_prompt(self, customer_comment: str, context: Dict) -> str:
        """
        ENHANCED: Build specialized prompt for WARRANTY INQUIRIES with better fallback handling

        Args:
            customer_comment: Customer's comment
            context: Enhanced context

        Returns:
            Warranty-focused prompt with appropriate fallback strategy
        """
        prompt_parts = [
            "You are Wootware's expert sales assistant responding to a WARRANTY INQUIRY.",
            "",
            "🚨 CRITICAL: NEVER direct customers to contact manufacturers for warranty claims.",
            "🛡️ FOCUS: Wootware handles ALL warranty claims internally - we are the customer's warranty support.",
            "",
            f"CUSTOMER INQUIRY: {customer_comment}",
            "",
        ]

        # Check product availability and external comment status
        product_selection = context.get("product_selection", {})
        is_external_comment = context.get("is_external_comment", False)
        has_product_data = product_selection.get("has_product_data", False)

        if has_product_data:
            # We have product data - use it
            primary_product = product_selection.get("primary_product")
            if primary_product and primary_product.get("is_viable", False):
                # Extract warranty info from product description
                description = primary_product.get("description", "")
                warranty_info = self._extract_warranty_info(description)

                prompt_parts.extend(
                    [
                        "📋 CURRENT PRODUCT INFORMATION:",
                        f"• Product: {primary_product.get('product_name', 'N/A')}",
                        f"• SKU: {primary_product.get('sku', 'N/A')}",
                        "",
                    ]
                )

                if warranty_info:
                    prompt_parts.extend(
                        ["🛡️ WARRANTY INFORMATION:", f"• {warranty_info}", ""]
                    )
                else:
                    prompt_parts.extend(
                        [
                            "🛡️ WARRANTY INFORMATION:",
                            "• Warranty details available - check product specifications",
                            "• Wootware handles ALL warranty claims internally - we'll assist with the process",
                            "• NEVER direct customers to contact manufacturers for warranty",
                            "",
                        ]
                    )
            else:
                # Product data exists but not viable (truly EOL)
                prompt_parts.extend(
                    [
                        "⚠️ PRODUCT STATUS:",
                        "The requested product appears to be discontinued or end-of-life.",
                        "Warranty may still be valid depending on purchase date.",
                        "",
                    ]
                )

        elif is_external_comment:
            # External comment - explain why we need to look up product
            prompt_parts.extend(
                [
                    "🔍 PRODUCT LOOKUP NEEDED:",
                    "This inquiry came through an external channel without product details linked.",
                    "We'll need to look up the specific product to provide accurate warranty information.",
                    "",
                ]
            )

            # Try to extract model number from comment for helpful response
            model_number = self._extract_model_number_from_comment(customer_comment)
            if model_number:
                prompt_parts.extend(
                    [
                        f"📝 IDENTIFIED PRODUCT: {model_number}",
                        "We can help look this up in our system for detailed warranty information.",
                        "",
                    ]
                )

        else:
            # No product data and not external - general case
            prompt_parts.extend(
                [
                    "🔍 PRODUCT INFORMATION NEEDED:",
                    "To provide accurate warranty details, we'll need to look up the specific product.",
                    "",
                ]
            )

        # Add example for tone (if available)
        examples = context.get("examples", [])
        if examples:
            example = examples[0]
            # Only show non-pricing examples for warranty queries
            if "warranty" in example.get("original_reply", "").lower():
                prompt_parts.extend(
                    [
                        "📝 TONE REFERENCE:",
                        f"Example warranty response style: {example.get('original_reply', '')[:150]}...",
                        "",
                    ]
                )

        # Guidelines based on data availability
        if has_product_data:
            prompt_parts.extend(
                [
                    "RESPONSE GUIDELINES:",
                    "1. ✅ Use the warranty information provided above",
                    "2. ❌ DO NOT include pricing unless specifically requested",
                    "3. ❌ NEVER direct customers to contact manufacturers for warranty",
                    "4. ✅ ALWAYS emphasize that Wootware handles warranty claims internally",
                    "5. ✅ Provide clear warranty terms and duration when available",
                    "6. ✅ Guide customers through Wootware's warranty claims process",
                    "7. ✅ Be helpful and professional",
                    "8. ❌ Do NOT include signature - it will be added automatically",
                ]
            )
        else:
            prompt_parts.extend(
                [
                    "RESPONSE GUIDELINES:",
                    "1. ✅ Acknowledge the warranty inquiry professionally",
                    "2. ✅ Offer to look up the specific product for detailed warranty info",
                    "3. ✅ Emphasize that Wootware handles all warranty claims internally",
                    "4. ❌ NEVER direct customers to contact manufacturers for warranty",
                    "5. ❌ DO NOT make up warranty terms without product data",
                    "6. ❌ DO NOT claim products are discontinued without confirmation",
                    "7. ✅ Ask for product details if needed for lookup",
                    "8. ✅ Be helpful and offer next steps through Wootware support",
                    "9. ❌ Do NOT include signature - it will be added automatically",
                ]
            )

        prompt_parts.extend(["", "Generate your warranty-focused response now:"])

        return "\n".join(prompt_parts)

    def _extract_model_number_from_comment(self, comment: str) -> str:
        """
        Extract product model number from customer comment

        Args:
            comment: Customer comment text

        Returns:
            Extracted model number or empty string
        """
        # Common patterns for product model numbers
        model_patterns = [
            r"\b[A-Z]{2,}\d{4,}[A-Z\d]*\b",  # ST20000NM002H, GTX1080, etc.
            r"\b\d{1,2}TB\b",  # Storage capacity
            r"\bGTX\s*\d{4}\b",  # Graphics cards
            r"\bRTX\s*\d{4}\b",  # RTX cards
            r"\bRyzen\s*\d+\s*\d{4}[A-Z]*\b",  # AMD processors
            r"\bi[3579]-\d{4,5}[A-Z]*\b",  # Intel processors
        ]

        found_models = []
        for pattern in model_patterns:
            matches = re.findall(pattern, comment, re.IGNORECASE)
            found_models.extend(matches)

        # Return the longest/most specific match
        if found_models:
            return max(found_models, key=len)

        return ""

    def _build_general_inquiry_with_lookup_prompt(
        self, customer_comment: str, context: Dict
    ) -> str:
        """
        NEW: Build prompt for general inquiries that need product lookup

        Args:
            customer_comment: Customer's comment
            context: Enhanced context

        Returns:
            General inquiry prompt with product lookup guidance
        """
        prompt_parts = [
            "You are Wootware's expert sales assistant responding to a GENERAL INQUIRY.",
            "",
            f"CUSTOMER INQUIRY: {customer_comment}",
            "",
        ]

        # Check if this is an external comment needing lookup
        is_external_comment = context.get("is_external_comment", False)
        has_product_data = context.get("product_selection", {}).get(
            "has_product_data", False
        )

        if is_external_comment and not has_product_data:
            prompt_parts.extend(
                [
                    "🔍 SITUATION:",
                    "This inquiry came through an external channel without linked product information.",
                    "We should offer to help look up the specific product they're asking about.",
                    "",
                ]
            )

            # Try to extract what they're looking for
            model_number = self._extract_model_number_from_comment(customer_comment)
            if model_number:
                prompt_parts.extend([f"📝 IDENTIFIED PRODUCT: {model_number}", ""])

        elif has_product_data:
            # We have product data - use it
            primary_product = context.get("product_selection", {}).get(
                "primary_product"
            )
            if primary_product:
                prompt_parts.extend(
                    [
                        "📋 PRODUCT INFORMATION AVAILABLE:",
                        f"• Product: {primary_product.get('product_name', 'N/A')}",
                        f"• SKU: {primary_product.get('sku', 'N/A')}",
                        "",
                    ]
                )

        # Add examples for tone and style
        examples = context.get("examples", [])
        if examples:
            prompt_parts.extend(
                ["📝 SIMILAR PAST RESPONSES (for style reference):", ""]
            )
            for i, example in enumerate(examples[:2], 1):
                if isinstance(example.get("original_comment"), str) and isinstance(
                    example.get("original_reply"), str
                ):
                    prompt_parts.append(
                        f"Example {i} (similarity: {example.get('similarity_score', 0):.3f}):"
                    )
                    prompt_parts.append(
                        f"Customer: {example['original_comment'][:100]}..."
                    )
                    prompt_parts.append(f"Agent: {example['original_reply'][:150]}...")
                    prompt_parts.append("")

        prompt_parts.extend(
            [
                "RESPONSE GUIDELINES:",
                "1. ✅ Be helpful and professional",
                "2. ✅ Offer to look up specific products if needed",
                "3. ✅ Use any available product information appropriately",
                "4. ✅ Provide accurate information based on available data",
                "5. ❌ Don't make claims about product availability without data",
                "6. ❌ Don't assume products are discontinued without confirmation",
                "7. ✅ Guide customer on next steps if more info needed",
                "8. Match Wootware's friendly but professional tone",
                "9. Do NOT include signature - it will be added automatically",
                "",
                "Generate your helpful response now:",
            ]
        )

        return "\n".join(prompt_parts)

    def _extract_warranty_info(self, description: str) -> str:
        """
        Extract warranty information from product description

        Args:
            description: Product description HTML/text

        Returns:
            Extracted warranty information or empty string
        """
        if not description:
            return ""

        # Look for warranty patterns in description
        warranty_patterns = [
            r"(\d+\s*year[s]?\s*(?:limited\s*)?warranty)",
            r"(warranty[:\s]*\d+\s*year[s]?)",
            r"(\d+\s*year[s]?\s*limited\s*warranty)",
            r"(warranty[:\s]*.*?year[s]?)",
        ]

        description_lower = description.lower()

        for pattern in warranty_patterns:
            match = re.search(pattern, description_lower, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Look for warranty in table format
        if "warranty" in description_lower:
            # Try to extract from HTML table structure
            soup = (
                BeautifulSoup(description, "html.parser")
                if "<" in description
                else None
            )
            if soup:
                # Look for warranty in table cells
                for cell in soup.find_all(["td", "th"]):
                    if cell.text and "warranty" in cell.text.lower():
                        next_cell = cell.find_next_sibling(["td", "th"])
                        if next_cell and next_cell.text.strip():
                            return next_cell.text.strip()

        return ""

    def _build_standard_enhanced_prompt(
        self, customer_comment: str, context: Dict
    ) -> str:
        """
        Build standard enhanced prompt for non-pricing/stock queries

        Args:
            customer_comment: Customer's comment
            context: Enhanced context

        Returns:
            Standard enhanced prompt
        """
        # ENHANCED: Add confidence-based language guidance
        confidence = context.get("product_search_result", {}).get("confidence", 0)
        confidence_level = get_confidence_level(confidence)

        prompt_parts = [
            f"You are Wootware's expert sales assistant. Generate a helpful, accurate response with {confidence_level} confidence language:",
            "",
            f"CUSTOMER COMMENT: {customer_comment}",
            "",
        ]

        # Add customer context if available
        customer_data = context.get("customer_data", {})
        if customer_data:
            prompt_parts.extend(
                [
                    "CUSTOMER INFORMATION:",
                    f"- Name: {customer_data.get('customer_firstname', 'N/A')}",
                    f"- Email: {customer_data.get('customer_email', 'N/A')}",
                    "",
                ]
            )

        # Add product context if available - but check viability
        product_selection = context.get("product_selection", {})
        if product_selection.get("primary_product"):
            primary_product = product_selection["primary_product"]
            prompt_parts.extend(
                [
                    "MAIN PRODUCT INFORMATION:",
                    f"- Product: {primary_product.get('name', 'N/A')}",
                    f"- SKU: {primary_product.get('sku', 'N/A')}",
                    "",
                ]
            )

            # Only include pricing/stock if product is viable
            if primary_product.get("is_viable", False):
                prompt_parts.extend(
                    [
                        f"- Current Price: R{primary_product.get('current_price', 'TBD')}",
                        f"- Stock Status: {'In Stock' if primary_product.get('is_in_stock') else 'Out of Stock'}",
                        f"- Lead Time: {primary_product.get('lead_time', 'TBD')}",
                        "",
                    ]
                )
            else:
                prompt_parts.extend(["- Note: This product is no longer available", ""])

        # Add alternative product if available
        if product_selection.get("secondary_product"):
            secondary_product = product_selection["secondary_product"]
            if secondary_product.get("is_viable", False):
                prompt_parts.extend(
                    [
                        "ALTERNATIVE PRODUCT:",
                        f"- Product: {secondary_product.get('name', 'N/A')}",
                        f"- Current Price: R{secondary_product.get('current_price', 'TBD')}",
                        f"- Stock Status: {'In Stock' if secondary_product.get('is_in_stock') else 'Out of Stock'}",
                        "",
                    ]
                )

        # Add similar response examples if available
        examples = context.get("examples", [])
        if examples:
            prompt_parts.extend(["SIMILAR PAST RESPONSES (for reference):", ""])
            for i, example in enumerate(examples[:2], 1):
                if isinstance(example.get("original_comment"), str) and isinstance(
                    example.get("original_reply"), str
                ):
                    prompt_parts.append(
                        f"Example {i} (similarity: {example.get('similarity_score', 0):.3f}):"
                    )
                    prompt_parts.append(
                        f"Customer: {example['original_comment'][:100]}..."
                    )
                    prompt_parts.append(f"Agent: {example['original_reply'][:100]}...")
                    prompt_parts.append("")

        prompt_parts.extend(
            [
                "RESPONSE GUIDELINES:",
                "1. Be helpful, professional, and specific",
                "2. Include accurate product details when available",
                "3. Do not provide pricing or stock info for discontinued products",
                "4. Offer alternatives if the requested item isn't available",
                "5. Include next steps or contact information if needed",
                "6. Match Wootware's friendly but professional tone",
                "7. Do NOT include a signature - this will be added automatically",
                "",
                "Generate your response:",
            ]
        )

        return "\n".join(prompt_parts)

    def generate_simple_response(
        self, customer_comment: str, intent: str = None, woot_rep: str = None
    ) -> str:
        """
        Generate a simple response based on customer comment and intent

        Args:
            customer_comment: The customer's comment
            intent: Predicted intent (optional)
            woot_rep: The rep's name to append to the response

        Returns:
            Generated response text with rep signature
        """
        try:
            logger.info(f"Generating simple response for: {customer_comment[:100]}...")
            logger.info(f"Rep: {woot_rep}")

            system_prompt = "You are Wootware's helpful sales assistant. Provide a professional and helpful response to the customer's inquiry. Do not include signatures as they will be added automatically. Never provide pricing or stock information for discontinued or End-of-Life products without confirming availability."

            user_prompt = f"Customer Comment: {customer_comment}"
            if intent:
                user_prompt += f"\nPredicted Intent: {intent}"
            user_prompt += "\n\nPlease provide a helpful response:"

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=500,
            )

            generated_reply = response.choices[0].message.content.strip()

            # Append rep signature
            final_reply = self._append_rep_signature(generated_reply, woot_rep)

            logger.info(
                f"Generated simple response length: {len(final_reply)} (with {woot_rep} signature)"
            )

            return final_reply

        except Exception as e:
            logger.error(f"Simple response generation error: {e}")
            fallback_response = "I apologize, but I'm having trouble generating a response right now. Please contact our sales team directly."
            return self._append_rep_signature(fallback_response, woot_rep)

    def test_generation(
        self, test_comment: str = "Hello, I need help with a product inquiry"
    ) -> Dict:
        """
        Test the LLM generation capabilities with product viability logic

        Args:
            test_comment: Comment to test with

        Returns:
            Dict with test results
        """
        try:
            logger.info(
                "Testing enhanced LLM generation with product viability logic..."
            )

            # Test simple generation
            simple_response = self.generate_simple_response(test_comment)

            # Test with fake examples
            fake_examples = [
                {
                    "similarity_score": 0.8,
                    "original_comment": "I need help with shipping",
                    "original_reply": "Thank you for contacting us. We offer various shipping options...",
                }
            ]
            example_response = self.generate_response_from_examples(
                test_comment, fake_examples
            )

            # Test enhanced response with fake viable product data
            fake_context = {
                "data_needs": {
                    "needs_pricing": True,
                    "needs_stock": False,
                    "needs_real_time_data": True,
                },
                "product_selection": {
                    "primary_product": {
                        "name": "Test Product",
                        "sku": "TEST-001",
                        "product_id": "test123",
                        "is_viable": True,
                        "viability_reason": "Product is in stock",
                    },
                    "has_viable_product": True,
                    "selection_reason": "Using main product (viable)",
                },
                "real_time_data": {
                    "primary_product": {
                        "product_name": "Test Product",
                        "sku": "TEST-001",
                        "is_viable": True,
                        "pricing": {"current_price": 1299.99, "is_on_promotion": False},
                    },
                    "data_freshness": datetime.now().isoformat(),
                },
                "prioritization_strategy": "real_time_priority",
                "examples": fake_examples[:1],
            }
            enhanced_response = self.generate_enhanced_response(
                "What's the price of this product?", fake_context
            )

            # Test enhanced response with EOL product
            fake_eol_context = {
                "data_needs": {
                    "needs_pricing": True,
                    "needs_stock": False,
                    "needs_real_time_data": True,
                },
                "product_selection": {
                    "primary_product": None,
                    "has_viable_product": False,
                    "selection_reason": "No viable products available (EOL)",
                },
                "real_time_data": {
                    "primary_product": {
                        "product_name": "EOL Product",
                        "sku": "EOL-001",
                        "is_viable": False,
                        "viability_reason": "Product is End of Life (EOL)",
                    },
                    "data_freshness": datetime.now().isoformat(),
                },
                "prioritization_strategy": "real_time_priority",
                "examples": fake_examples[:1],
            }
            eol_response = self.generate_enhanced_response(
                "What's the price of this EOL product?", fake_eol_context
            )

            return {
                "test_successful": True,
                "simple_response": simple_response,
                "example_response": example_response,
                "enhanced_response": enhanced_response,
                "eol_response": eol_response,
                "simple_response_length": len(simple_response),
                "example_response_length": len(example_response),
                "enhanced_response_length": len(enhanced_response),
                "eol_response_length": len(eol_response),
                "model_used": self.model,
                "temperature": self.temperature,
                "enhanced_features": "Real-time data prioritization with product viability logic active",
            }

        except Exception as e:
            logger.error(f"Enhanced LLM generation test failed: {e}")
            return {
                "test_successful": False,
                "error": str(e),
                "model_used": self.model,
                "temperature": self.temperature,
            }


# Global generator instance (lazy-loaded)
_llm_generator_instance = None


def get_llm_generator() -> LLMGenerator:
    """
    Get a global LLM generator instance (singleton pattern)

    Returns:
        LLMGenerator instance
    """
    global _llm_generator_instance
    if _llm_generator_instance is None:
        _llm_generator_instance = LLMGenerator()
    return _llm_generator_instance


# Convenience functions
def generate_response_with_examples(
    customer_comment: str, examples: List[Dict], woot_rep: str = None
) -> str:
    """
    Convenience function to generate response using examples

    Args:
        customer_comment: Customer's comment
        examples: Response examples from Pinecone search
        woot_rep: Rep name to append to response

    Returns:
        Generated response text with rep signature
    """
    generator = get_llm_generator()
    return generator.generate_response_from_examples(
        customer_comment, examples, woot_rep
    )


def generate_enhanced_response_with_context(
    customer_comment: str, context: Dict, woot_rep: str = None
) -> str:
    """
    Convenience function to generate response with enhanced context, real-time data prioritization, and product viability logic

    Args:
        customer_comment: Customer's comment
        context: Enhanced context with real-time data, prioritization strategy, and viability checks
        woot_rep: Rep name to append to response

    Returns:
        Generated response text with rep signature
    """
    generator = get_llm_generator()
    return generator.generate_enhanced_response(customer_comment, context, woot_rep)


def generate_simple_response_for_comment(
    customer_comment: str, intent: str = None, woot_rep: str = None
) -> str:
    """
    Convenience function to generate simple response

    Args:
        customer_comment: Customer's comment
        intent: Predicted intent (optional)
        woot_rep: Rep name to append to response

    Returns:
        Generated response text with rep signature
    """
    generator = get_llm_generator()
    return generator.generate_simple_response(customer_comment, intent, woot_rep)


def test_llm_generation() -> Dict:
    """
    Convenience function to test enhanced LLM generation with product viability logic

    Returns:
        Dict with test results
    """
    generator = get_llm_generator()
    return generator.test_generation()
