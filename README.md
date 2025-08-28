# Data Quality Rule Recommendation Engine

An advanced data quality rule recommendation system that uses Large Language Models (LLM) to dynamically generate, analyze, and enhance data quality rules based on your data, business context, and metadata.

## Quick Start

### Prerequisites
- Python 3.8 or higher
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

4. **Run the application**
```bash
python app.py
```

5. **Access the application**
- Open your browser and navigate to: `http://localhost:5000`
- Upload a CSV file to get data quality recommendations

## Features

### Dynamic Rule Generation
- **AI-Powered Analysis**: Intelligent rule generation using advanced algorithms
- **Context-Aware Recommendations**: Incorporates business glossary, technical metadata, and data lineage
- **Automatic Rule Generation**: Identifies patterns and suggests appropriate data quality rules

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
├── app.py                    # Main Flask application
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
│   ├── business_glossary.json     # Business terms
│   ├── technical_metadata.json    # Schema information
│   ├── data_lineage.json         # Data flow info
│   └── standard_rules.json       # Rule templates
└── README.md              # This documentation
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

## Usage

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
