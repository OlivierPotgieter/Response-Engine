# AGENTS.md - Response Engine AI Development Guide

## Project Overview

The Response Engine is an AI-powered system that generates customer service responses by learning from past staff interactions. It uses intent classification, vector similarity search, and real-time product data integration to create accurate, contextual responses.

### Core Architecture
- **Intent Classification**: ML model predicting 10 customer intent types
- **Vector Search**: Pinecone-based similarity matching with past responses
- **Product Intelligence**: AI-powered product identification using embeddings
- **Response Generation**: OpenAI GPT-4o with context-aware prompt engineering
- **Real-time Data**: Dynamic integration of current pricing/stock information

## Primary Development Objectives

### 1. Enhanced Response Confidence & Accuracy

#### Current Issues
- Responses sometimes sound uncertain or generic
- Inconsistent handling of product-specific queries without IDs
- Need better confidence thresholds for product extraction

#### Proposed Improvements

**A. Confidence Scoring System**
```python
# Add to smart_product_extractor.py
class ConfidenceThresholds:
    HIGH_CONFIDENCE = 0.85    # Direct product match, use without hesitation
    MEDIUM_CONFIDENCE = 0.70  # Good match, use with slight hedging
    LOW_CONFIDENCE = 0.50     # Uncertain, offer multiple options
    MINIMUM_THRESHOLD = 0.50  # Below this, don't attempt product-specific responses
```

**B. Response Confidence Indicators**
- Modify `llm_generator.py` to adjust language based on confidence:
  - High: "The RTX 4090 is currently priced at..."
  - Medium: "Based on your description, the RTX 4090 appears to be..."
  - Low: "I found several products that might match. Could you confirm..."

**C. Fallback Strategies**
- When no product ID exists and extraction confidence < 0.5:
  - Provide general category information
  - Ask clarifying questions
  - Suggest top products in detected category

### 2. Intent-Specific Response Strategies

#### Supported Intents (In Scope)
1. **General Inquiry** - Provide helpful information, offer to lookup products
2. **Order Assistance** - Help with order-related questions
3. **Pricing Inquiry** - Only answer if product identified (ID or high-confidence extraction)
4. **Quotation Request** - Combine pricing + availability for identified products
5. **Stock Availability** - Provide real-time stock for identified products
6. **Warranty Inquiry** - General warranty info, product-specific if identified
7. **Returns and Issues** - Policy information and support guidance
8. **Shipping and Collection** - Delivery options and timeframes

#### Out of Scope
- **Product Recommendation** - Redirect to human staff
- **Compatibility or Upgrade** - Too complex for automated responses

### 3. Product Identification Enhancement

#### Current State
- Uses AI embeddings via Pinecone for product matching
- Extracts products from comments when no ID provided
- Needs confidence threshold implementation

#### Proposed Enhancements

**A. Minimum Confidence Implementation**
```python
# In data_processor.py - attempt_product_lookup_from_comment()
def should_use_extracted_product(extraction_result):
    """Determine if extracted product is reliable enough to use"""
    confidence = extraction_result.get("confidence", 0)
    best_match = extraction_result.get("best_match", {})
    
    # Only use for pricing/stock if confidence >= 0.7
    if confidence >= 0.7:
        return True, "high_confidence"
    elif confidence >= 0.5:
        return True, "medium_confidence"  
    else:
        return False, "insufficient_confidence"
```

**B. Multi-Signal Validation**
- Cross-reference extracted products with:
  - Category detection confidence
  - Brand/model pattern matching
  - Historical query patterns
  - Price range expectations

**C. Extraction Improvement Pipeline**
1. Enhance `category_intelligence.py` with more patterns
2. Add brand-specific model number formats
3. Implement fuzzy matching for common misspellings
4. Cache successful extractions for learning

### 4. Response Quality Improvements

#### A. Context-Aware Prompting
```python
# Enhanced prompt strategies based on confidence and data availability
PROMPT_STRATEGIES = {
    "high_confidence_with_data": "authoritative_response",
    "medium_confidence_with_data": "qualified_response", 
    "low_confidence_no_data": "exploratory_response",
    "no_product_generic": "category_guidance_response"
}
```

#### B. Dynamic Example Selection
- Prioritize examples with similar confidence levels
- Match examples by product category when possible
- Weight recent examples higher for trending products

#### C. Response Validation Layer
```python
class ResponseValidator:
    def validate_response(self, response, context):
        checks = {
            "has_greeting": self._check_greeting(response),
            "addresses_intent": self._check_intent_coverage(response, context),
            "appropriate_confidence": self._check_confidence_language(response, context),
            "includes_next_steps": self._check_actionable(response),
            "length_appropriate": len(response.split()) > 20
        }
        return all(checks.values()), checks
```

### 5. A/B Testing System Implementation

#### Architecture for Shadow Mode Testing

**A. Shadow Request Handler**
```python
# New module: shadow_processor.py
class ShadowProcessor:
    def __init__(self):
        self.shadow_mode = True
        self.log_responses_only = True
        
    def process_shadow_request(self, request_id):
        """Process request without sending to customer"""
        result = process_customer_request(request_id, generate_response=True)
        
        # Log to shadow_responses table
        self.log_shadow_response(request_id, result)
        
        # Compare with actual staff response later
        return result
```

**B. Database Schema Addition**
```sql
CREATE TABLE shadow_responses (
    id INT PRIMARY KEY AUTO_INCREMENT,
    request_id INT,
    generated_response TEXT,
    generation_method VARCHAR(50),
    confidence_score FLOAT,
    products_identified JSON,
    intent_predicted VARCHAR(50),
    processing_time FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (request_id) REFERENCES availability_conversion(id)
);

CREATE TABLE shadow_metrics (
    id INT PRIMARY KEY AUTO_INCREMENT,
    request_id INT,
    response_similarity FLOAT,
    intent_match BOOLEAN,
    product_match BOOLEAN,
    response_length_diff INT,
    customer_satisfaction INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**C. Comparison Pipeline**
```python
# New module: shadow_analyzer.py
class ShadowAnalyzer:
    def compare_responses(self, request_id):
        """Compare AI response with staff response"""
        ai_response = get_shadow_response(request_id)
        staff_response = get_custom_response_data(request_id)
        
        metrics = {
            "similarity": calculate_semantic_similarity(ai_response, staff_response),
            "length_difference": abs(len(ai_response) - len(staff_response)),
            "same_products": compare_products_mentioned(ai_response, staff_response),
            "tone_analysis": analyze_tone_difference(ai_response, staff_response)
        }
        
        return metrics
```

**D. Testing Schedule**
```python
# Configuration for shadow testing
SHADOW_CONFIG = {
    "enabled": True,
    "start_date": "2024-01-15",
    "end_date": "2024-01-22",
    "sample_rate": 1.0,  # Process 100% of requests
    "excluded_intents": ["Product Recommendation", "Compatibility or Upgrade"],
    "minimum_confidence": 0.5,
    "log_level": "detailed"
}
```

### 6. Monitoring & Improvement Pipeline

#### A. Real-time Monitoring Dashboard
```python
# Metrics to track
MONITORING_METRICS = {
    "response_generation": {
        "total_requests": 0,
        "successful_generations": 0,
        "fallback_used": 0,
        "average_confidence": 0.0,
        "average_response_time": 0.0
    },
    "product_identification": {
        "extraction_attempts": 0,
        "successful_extractions": 0,
        "confidence_distribution": {},
        "category_accuracy": 0.0
    },
    "intent_classification": {
        "predictions_made": 0,
        "in_scope_percentage": 0.0,
        "intent_distribution": {}
    }
}
```

#### B. Feedback Loop Implementation
1. **Weekly Analysis Reports**
   - Compare AI vs staff responses
   - Identify systematic errors
   - Track improvement areas

2. **Continuous Learning**
   - Add successful extractions to training data
   - Update category intelligence based on patterns
   - Refine confidence thresholds

3. **Quality Metrics**
   - Response relevance score
   - Information accuracy rate
   - Customer satisfaction proxy

### 7. Implementation Roadmap

#### Phase 1: Confidence System (Week 1)
- [ ] Implement confidence thresholds in product extraction
- [ ] Add confidence-based language to LLM prompts
- [ ] Create response validation layer
- [ ] Test with historical data

#### Phase 2: Response Quality (Week 2)
- [ ] Enhance prompt engineering for each intent
- [ ] Implement dynamic example selection
- [ ] Add fallback strategies for low confidence
- [ ] Improve category intelligence

#### Phase 3: Shadow Testing Setup (Week 3)
- [ ] Create shadow processing pipeline
- [ ] Set up database tables for logging
- [ ] Implement comparison metrics
- [ ] Build monitoring dashboard

#### Phase 4: A/B Testing (Week 4-5)
- [ ] Run shadow mode for one week
- [ ] Collect comprehensive metrics
- [ ] Daily analysis of results
- [ ] Identify improvement areas

#### Phase 5: Refinement (Week 6)
- [ ] Implement fixes based on A/B results
- [ ] Adjust confidence thresholds
- [ ] Update prompt strategies
- [ ] Prepare for production deployment

### 8. Testing Strategies

#### Unit Tests for Confidence System
```python
def test_product_extraction_confidence():
    test_cases = [
        ("I need an RTX 4090", 0.9, True),  # High confidence
        ("graphics card for gaming", 0.4, False),  # Too generic
        ("Seagate 4TB ST4000DM004", 0.95, True),  # Model number
        ("something fast for gaming", 0.2, False)  # No product
    ]
    
    for comment, expected_conf, should_use in test_cases:
        result = extract_products_from_comment(comment)
        assert abs(result['confidence'] - expected_conf) < 0.1
        assert should_use_for_pricing(result) == should_use
```

#### Integration Tests
- Test full pipeline with various intent/product combinations
- Verify fallback behaviors work correctly
- Ensure real-time data integration functions properly

#### Shadow Mode Validation
- Verify no responses are sent to customers
- Confirm all metrics are logged correctly
- Test comparison algorithms accuracy

### 9. Configuration Management

```python
# config/response_engine_config.py
class ResponseEngineConfig:
    # Confidence Settings
    PRODUCT_CONFIDENCE_THRESHOLD = 0.7
    USE_FALLBACK_BELOW = 0.5
    
    # Response Settings
    MIN_RESPONSE_LENGTH = 50
    MAX_RESPONSE_LENGTH = 500
    INCLUDE_CONFIDENCE_LANGUAGE = True
    
    # Shadow Testing
    SHADOW_MODE_ENABLED = False
    SHADOW_LOG_LEVEL = "detailed"
    SHADOW_COMPARISON_ENABLED = True
    
    # Intent Handling
    SUPPORTED_INTENTS = [
        "General Inquiry",
        "Order Assistance",
        "Pricing Inquiry",
        "Quotation Request",
        "Stock Availability",
        "Warranty Inquiry",
        "Returns and Issues",
        "Shipping and Collection"
    ]
    
    REQUIRES_PRODUCT_ID = [
        "Pricing Inquiry",
        "Stock Availability",
        "Quotation Request"
    ]
```

### 10. Success Metrics

#### Primary KPIs
1. **Response Accuracy**: % of responses with correct information
2. **Product Identification Rate**: % successful extractions above threshold
3. **Response Similarity**: Semantic similarity to staff responses
4. **Processing Time**: Average time to generate response
5. **Fallback Rate**: % of requests requiring fallback methods

#### Secondary Metrics
- Intent classification accuracy
- Confidence score distribution
- Category detection precision
- Real-time data usage rate
- Error rates by intent type

### 11. Risk Mitigation

#### Potential Risks & Mitigations
1. **False Product Identification**
   - Mitigation: Strict confidence thresholds
   - Fallback: Ask for clarification

2. **Incorrect Pricing/Stock Info**
   - Mitigation: Only show when product ID confirmed
   - Fallback: General pricing guidance

3. **Poor Response Quality**
   - Mitigation: Response validation layer
   - Fallback: Use high-quality examples

4. **System Overconfidence**
   - Mitigation: Conservative thresholds
   - Fallback: Human escalation path

### 12. Future Enhancements

#### Short Term 
- Implement learning from shadow test results
- Add more sophisticated intent routing
- Enhance product catalog search
- Improve multi-product handling

#### Medium Term 
- Build customer feedback integration
- Implement response personalization
- Add multi-language support
- Create automated quality scoring

#### Long Term 
- Develop recommendation engine
- Add compatibility checking
- Implement proactive suggestions
- Build full conversational AI

## Conclusion

This guide provides a comprehensive roadmap for enhancing the Response Engine with focus on:
1. Improving response confidence and accuracy
2. Implementing proper product identification thresholds
3. Creating a robust A/B testing system
4. Building continuous improvement pipelines

The phased approach ensures systematic improvement while maintaining system stability and allowing for data-driven refinements based on shadow testing results.
