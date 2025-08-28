# Data Quality Rule Recommendation Engine

An advanced data quality rule recommendation system that leverages GPT-4 and vector embeddings to dynamically generate, analyze, and enhance data quality rules based on your data, business context, and metadata.

## Key Features

### 🤖 AI-Powered Analysis
- **GPT-4 Integration**: Intelligent rule generation using OpenAI's GPT-4
- **Vector Similarity**: ChromaDB for efficient semantic search
- **Hybrid Approach**: Combines traditional rules with AI recommendations
- **Context-Aware**: Incorporates business glossary, technical metadata, and data lineage
- **Automated Learning**: Improves recommendations based on data patterns

### 📊 Comprehensive Analysis
- **Multi-level Analysis**:
  - Column-level patterns and constraints
  - Dataset-level relationships
  - Cross-column dependencies
  - Business context integration
- **Statistical Analysis**:
  - Distribution patterns
  - Outlier detection
  - Seasonality analysis
  - Format consistency
  - Correlation analysis
- **YData Profiling**: Detailed statistical profiling

### 🏗️ Modern Architecture
- **Flask Backend**: High-performance API server
- **Vector Database**: ChromaDB for semantic search
- **LLM Integration**: GPT-4 via OpenAI API
- **Modular Design**: Easily extensible architecture
- **Error Handling**: Robust error management and logging

## Quick Start

### Prerequisites
- Python 3.8 or higher
- OpenAI API key
- pip (Python package installer)
- Virtual environment (recommended)

### Installation & Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd data-quality-poc
```

2. **Create and activate virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Windows
set OPENAI_API_KEY=your_api_key_here

# Linux/Mac
export OPENAI_API_KEY=your_api_key_here
```

5. **Run the application**
```bash
python app.py
```

6. **Access the application**
Open your browser and navigate to: `http://localhost:5000`

## LLM Integration Details

### GPT-4 Implementation

The system uses GPT-4 for intelligent rule generation in two ways:

1. **Column-Level Analysis**:
```python
prompt = f"""
Column: {column_name}
Data Type: {data_type}
Sample Values: {sample_values}
Technical Metadata: {technical_metadata}
Business Context: {business_terms}

Recommend appropriate data quality rules for this column.
"""
```

2. **Dataset-Level Analysis**:
```python
prompt = f"""
Dataset Info: {dataset_stats}
Column Relationships: {correlations}
Business Rules: {business_rules}

Recommend dataset-level quality rules.
"""
```

### Prompt Engineering

The system uses carefully crafted prompts that:
1. Focus on specific aspects of data quality
2. Include relevant context and metadata
3. Request structured JSON responses
4. Handle edge cases and errors

Example response format:
```json
{
    "rule_type": "validity",
    "rule_name": "Email Format Validation",
    "description": "Ensures email addresses follow standard format",
    "severity": "HIGH",
    "sql_rule": "regexp_matches(email, '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$')",
    "business_impact": "Critical for customer communication"
}
```

### Comprehensive Analysis
- **Statistical Pattern Detection**: Advanced algorithms for pattern recognition
- **Distribution Analysis**: Understands data distributions and anomalies
- **Seasonality Detection**: Identifies time-based patterns and trends
- **Format Consistency**: Validates data format adherence
- **Inter-Column Relationships**: Analyzes relationships between columns

### Modern Architecture
- **FastAPI Backend**: High-performance, async-capable API
- **React Frontend**: Modern, responsive user interface
- **Vector Database**: ChromaDB for efficient similarity search
- **YData Profiling**: Comprehensive data profiling capabilities
- **Spark SQL Generation**: Automated conversion to Spark SQL

## Project Structure

```
data-quality-poc/
├── app.py                    # Main Flask application with GPT-4 integration
├── requirements.txt          # Python dependencies
├── templates/               # Flask templates
│   ├── base.html           # Base template with common structure
│   └── index.html          # Main application interface
├── static/                 # Static assets
│   ├── css/
│   │   └── style.css      # Custom styles
│   └── js/
│       └── main.js        # Frontend JavaScript
├── data/                  # Sample and configuration data
│   ├── sample_customer_data.csv   # Example dataset
│   ├── business_glossary.json     # Business terms for vector search
│   ├── technical_metadata.json    # Schema and constraints
│   ├── data_lineage.json         # Data flow information
│   └── standard_rules.json       # Rule templates
├── chromadb/              # Vector database storage
├── tests/                 # Unit and integration tests
│   ├── test_app.py       # Application tests
│   ├── test_llm.py       # LLM integration tests
│   └── test_rules.py     # Rule generation tests
└── README.md             # This documentation

## Performance Optimization

### 1. Batch Processing
- Process multiple columns in parallel
- Cache similar rule recommendations
- Reuse vector embeddings

### 2. Memory Management
- Stream large CSV files
- Limit sample sizes for analysis
- Clear cache periodically

## Roadmap

### Phase 1: Enhanced AI Integration
- [ ] Integration with Claude/GPT-4 Turbo
- [ ] Fine-tuned models for specific industries
- [ ] Active learning from user feedback

### Phase 2: Advanced Features
- [ ] Real-time data quality monitoring
- [ ] Automated rule adjustment
- [ ] Custom rule template generator

### Phase 3: Enterprise Features
- [ ] Multi-user support
- [ ] Role-based access control
- [ ] Audit logging
- [ ] API rate limiting

## Security Considerations

1. **Data Privacy**:
   - Data masking for sensitive fields
   - PII detection and handling
   - Secure API communication

2. **API Security**:
   - Rate limiting
   - Authentication
   - Input validation
   - CORS configuration
```

## Configuration Files

### business_glossary.json
Contains business terms and their definitions used for context-aware rule generation.
```json
{
    "glossary": {
        "terms": [
            {
                "id": "term_001",
                "name": "Customer ID",
                "definition": "Unique identifier for customer",
                "business_rules": ["Must be unique", "Cannot be null"],
                "domain": "Customer Data"
            }
        ]
    }
}
```

## Advanced Usage Examples

### 1. Custom Rule Templates

Add custom rule templates to `standard_rules.json`:
```json
{
    "rule_categories": {
        "address_validation": {
            "description": "Address format validation rules",
            "rules": [
                {
                    "rule_name": "Postal Code Format",
                    "description": "Validates postal code format",
                    "rule_type": "format",
                    "sql_template": "regexp_matches({column}, '^\\d{5}(-\\d{4})?$')",
                    "applicable_data_types": ["string", "varchar"]
                }
            ]
        }
    }
}
```

### 2. Custom LLM Prompts

Modify the system prompt for specific use cases:
```python
system_prompt = """
You are a data quality expert specializing in financial data.
Focus on:
1. Regulatory compliance
2. Financial accuracy
3. Audit requirements
...
"""
```

### 3. Vector Search Integration

Use ChromaDB for semantic search:
```python
results = business_glossary_collection.query(
    query_texts=[f"{column_name} {data_type}"],
    n_results=3
)
```

## Error Handling and Logging

### 1. LLM Error Recovery
```python
try:
    response = get_llm_recommendations(column)
except Exception as e:
    logger.error(f"LLM error: {str(e)}")
    return fallback_recommendations(column)
```

### 2. Data Type Serialization
```python
def convert_to_serializable(obj):
    """Handle complex data types for JSON serialization"""
    if hasattr(obj, 'tolist'):  # numpy arrays
        return obj.tolist()
    elif hasattr(obj, 'item'):  # numpy scalars
        return obj.item()
    # ... more handlers
```

## Basic Usage

1. **Upload CSV File**: Select and upload your CSV dataset
2. **Automated Analysis**: The system will:
   - Profile the data using YData Profiling
   - Analyze schema and data patterns
   - Match against business glossary terms
   - Search for relevant standard rules using vector similarity
3. **View Recommendations**: Get comprehensive data quality rule recommendations with:
   - Rule descriptions and SQL implementations
   - Severity levels (Critical, High, Medium, Low)
   - Business impact assessment
   - Violation counts and thresholds

## Example Files and Templates

### Core Configuration Files
- `business_glossary.json`: Business domain definitions and terms
- `technical_metadata.json`: Schema and constraint information
- `data_lineage.json`: Data flow and transformation details
- `standard_rules.json`: Standard data quality rule templates
- `ydata_profile.json`: Automated profiling results

### Example Templates
The `examples/` directory contains detailed example files that demonstrate the expected structure and format:

#### example_business_glossary.json
Shows how to define business terms with:
- Detailed definitions and rules
- Domain categorization
- Data sensitivity levels
- Aliases and data types

#### example_technical_metadata.json
Demonstrates technical specifications including:
- Column definitions and constraints
- Validation patterns
- PII information
- Indexing strategy
- Partitioning and retention policies

#### example_data_lineage.json
Illustrates data flow documentation with:
- Source and target systems
- Data transformations
- Update frequencies
- System dependencies
- Impact analysis

These examples serve as templates for creating your own configuration files. Review them in the `examples/` directory for detailed formats and best practices.

### Test Data
- `sample_customer_data.csv`: Sample dataset with intentional data quality issues for testing

## Technology Stack

- **Backend**: Flask (Python)
- **Vector Database**: ChromaDB
- **Data Profiling**: YData Profiling
- **Frontend**: Bootstrap 5, HTML5, CSS3, JavaScript
- **Data Processing**: Pandas, NumPy

## API Endpoints

- `GET /`: Main application interface
- `POST /upload`: Upload CSV file and get recommendations
- `GET /api/rules/<rule_type>`: Get standard rules by type
- `GET /api/glossary/<term_id>`: Get business glossary term

## Rule Categories

### 1. Completeness
- Null value checks
- Missing data percentage analysis
- Required field validation

### 2. Validity
- Format validation (email, phone, dates)
- Range checks for numeric data
- Domain value validation

### 3. Consistency
- Uniqueness constraints
- Referential integrity checks
- Cross-field consistency

### 4. Accuracy
- Statistical outlier detection
- Business rule compliance
- Data standard conformance

## Customization

### Adding New Rule Types
1. Update `standard_rules.json` with new rule definitions
2. Modify the `analyze_column` method in `app.py`
3. Add corresponding UI elements for the new rule type

### Extending Business Glossary
1. Add new terms to `business_glossary.json`
2. Restart the application to reindex the vector database

### Custom Metadata Schemas
1. Modify `technical_metadata.json` structure
2. Update the metadata parsing logic in the recommendation engine

## Performance Considerations

- **File Size Limit**: 16MB maximum for CSV uploads
- **Vector Database**: ChromaDB provides fast similarity search
- **Memory Usage**: Optimized for datasets up to 1M rows
- **Response Time**: Typical analysis completes in 5-15 seconds

## Future Enhancements

1. **LLM Integration**: Direct integration with OpenAI/Anthropic APIs
2. **Rule Execution**: Automated rule execution and monitoring
3. **Advanced Lineage**: Graph-based lineage visualization
4. **Rule Templates**: Industry-specific rule template libraries
5. **Real-time Monitoring**: Continuous data quality monitoring

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For questions and support, please create an issue in the repository.
