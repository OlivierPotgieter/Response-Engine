Response Engine/
├── app.py                       # Main Flask application
├── .env                         # Environment variables
├── README.md                    # This documentation
├── test_app.py                  # API test suite
├── intent_inference_pipeline.pkl # Trained intent model
├── labeled_full_replies.csv     # Response examples
├── product_intelligence.json     # Product Intelligence
├── category_intelligence.json     # Category Intelligence
├── modules/
│   ├── database/                # Database operations
│   │   ├── __init.py__          #INIT FILE
│   │   ├── main_db.py          # Main database queries
│   │   └── backend_db.py       # Product database queries
│   ├── ai/                     # AI operations
│   │   ├── __init.py__         #INIT file
│   │   ├── intent_classifier.py # Intent prediction
│   │   ├── pinecone_client.py  # Vector search
│   │   └── llm_generator.py    # Response generation
│   ├── processors/             # Business logic
│   │   ├── __init.py__         #INIT file
│   │   ├── data_processor.py   # Data processing
│   │   ├── Product_Identifier.py   # Old solution going to be replaced
│   │   ├── product_identifier_wrapper_temp.py   # New working solution
│   │   ├── product_search.py   # Finds a product
│   │   └── response_builder.py # Response formatting
│   └── product_identification     # product Identifiers
│       ├──__Init.py__          #Init file
│       ├──build_systems.py     #creates systems
│       ├──category_intelligence #defines category intelligence
│       ├──confing.py               #Configuration file
│       ├──database_connector.py    #Database connector
│       ├──embedding_service.py     #Embedding service
│       ├──pinecone_manager.py      #Pinecone manager
│       ├──product_extractor.py     #Product extractor 
│       ├──product_intelligence_builder.py  #Product intelligence
│       └──smart_product_extractor.py       #Smart product extrator


└── test_*.py                   # Individual module tests****
