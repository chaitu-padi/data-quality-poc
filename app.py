import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import chromadb
from chromadb.config import Settings
import uuid
from typing import Any, Dict, List, Union
# OpenAI integration
import openai

def convert_to_serializable(obj: Any) -> Union[Dict, List, str, int, float, bool, None]:
    """Convert complex objects to JSON serializable types."""
    if hasattr(obj, 'tolist'):  # numpy arrays
        return obj.tolist()
    elif hasattr(obj, 'item'):  # numpy scalars
        return obj.item()
    elif pd.isna(obj):  # pandas NA/NaN/None
        return None
    elif isinstance(obj, pd.Series):
        return obj.to_list()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif hasattr(obj, '__dict__'):  # custom objects
        return str(obj)
    return obj
openai.api_key = os.environ.get('open_api_key')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('data', exist_ok=True)

class DataQualityRuleRecommendationEngine:
    def __init__(self):
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chromadb")

        # Create collections for different types of knowledge
        try:
            self.business_glossary_collection = self.client.get_collection("business_glossary")
        except:
            self.business_glossary_collection = self.client.create_collection("business_glossary")

        try:
            self.standard_rules_collection = self.client.get_collection("standard_rules")
        except:
            self.standard_rules_collection = self.client.create_collection("standard_rules")

        # Load and index knowledge base
        self.load_knowledge_base()

    def load_knowledge_base(self):
        """Load business glossary and standard rules into vector database"""
        errors = []
        
        # Load business glossary
        try:
            with open('business_glossary.json', 'r') as f:
                business_glossary = json.load(f)
        except FileNotFoundError:
            errors.append("Business glossary file not found: business_glossary.json")
            business_glossary = {"glossary": {"terms": []}}
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in business glossary: {str(e)}")
            business_glossary = {"glossary": {"terms": []}}
        
        try:
            # Index business glossary terms
            for term in business_glossary['glossary']['terms']:
                    doc_text = f"""
                    Term: {term['name']}
                    Definition: {term['definition']}
                    Business Rules: {' '.join(term.get('business_rules', []))}
                    Domain: {term.get('domain', '')}
                    """

                    # Convert metadata to compatible format
                    metadata = {
                        'id': term['id'],
                        'name': term['name'],
                        'definition': term['definition'],
                        'domain': term.get('domain', ''),
                        'business_rules': '; '.join(term.get('business_rules', [])),
                        'data_steward': term.get('data_steward', ''),
                        'synonyms': '; '.join(term.get('synonyms', [])),
                        'related_terms': '; '.join(term.get('related_terms', []))
                    }
                    
                    # Remove any None values
                    metadata = {k: v for k, v in metadata.items() if v is not None}

                    try:
                        self.business_glossary_collection.add(
                            documents=[doc_text],
                            metadatas=[metadata],
                            ids=[term['id']]
                        )
                    except Exception as e:
                        errors.append(f"Failed to add business glossary term '{term['name']}' to ChromaDB: {str(e)}")            # Load standard rules
            try:
                with open('standard_rules.json', 'r') as f:
                    standard_rules = json.load(f)
            except FileNotFoundError:
                errors.append("Standard rules file not found: standard_rules.json")
                standard_rules = {"rule_categories": {}}
            except json.JSONDecodeError as e:
                errors.append(f"Invalid JSON in standard rules: {str(e)}")
                standard_rules = {"rule_categories": {}}

            # Index standard rules
            rule_id = 0
            for category, category_info in standard_rules['rule_categories'].items():
                for rule in category_info['rules']:
                    doc_text = f"""
                    Rule Category: {category}
                    Rule Name: {rule['rule_name']}
                    Description: {rule['description']}
                    Rule Type: {rule['rule_type']}
                    Applicable Data Types: {', '.join(rule.get('applicable_data_types', []))}
                    SQL Template: {rule.get('sql_template', '')}
                    """

                    # Convert metadata to compatible format
                    metadata = {
                        'rule_name': rule['rule_name'],
                        'description': rule['description'],
                        'rule_type': rule['rule_type'],
                        'category': category,
                        'applicable_data_types': ', '.join(rule.get('applicable_data_types', [])),
                        'sql_template': rule.get('sql_template', '')
                    }

                    # Remove any None values
                    metadata = {k: v for k, v in metadata.items() if v is not None}

                    try:
                        self.standard_rules_collection.add(
                            documents=[doc_text],
                            metadatas=[metadata],
                            ids=[f"rule_{rule_id}"]
                        )
                        rule_id += 1
                    except Exception as e:
                        errors.append(f"Failed to add standard rule '{rule['rule_name']}' to ChromaDB: {str(e)}")

            if errors:
                print("Knowledge base loaded with warnings:")
                for error in errors:
                    print(f"- {error}")
            else:
                print("Knowledge base loaded successfully!")
        except Exception as e:
            print(f"Critical error loading knowledge base: {e}")
            if errors:
                print("Previous warnings:")
                for error in errors:
                    print(f"- {error}")

    def analyze_data_and_recommend_rules(self, df, technical_metadata, data_lineage):
        """
        Analyze the dataset and recommend ALL rules (standard and custom) using OpenAI LLM inference only.
        """
        recommendations = []
        try:
            # Get basic dataset statistics
            dataset_info = self._get_dataset_info(df)

            # For each column, use LLM to recommend all possible rules
            for column in df.columns:
                col_data = df[column]
                col_dtype = str(col_data.dtype)
                sample_values = col_data.dropna().astype(str).unique()[:5].tolist()
                # Gather technical metadata for the column
                tech_metadata = self._get_column_tech_metadata(column, technical_metadata)
                # Gather lineage info for the column
                lineage_info = data_lineage.get(column, {}) if data_lineage else {}
                # Gather business glossary info for the column
                glossary_terms = []
                if hasattr(self, 'business_glossary_collection'):
                    try:
                        glossary_results = self.business_glossary_collection.query(
                            query_texts=[f"{column} {col_dtype}"],
                            n_results=2
                        )
                        glossary_terms = glossary_results.get('documents', [])
                    except Exception:
                        pass
                # Gather standard rules
                standard_rules = []
                try:
                    with open('standard_rules.json', 'r') as f:
                        sr_json = json.load(f)
                    for category, category_info in sr_json.get('rule_categories', {}).items():
                        for rule in category_info.get('rules', []):
                            standard_rules.append(rule)
                except Exception:
                    pass
                prompt = (
                    f"Column name: {column}\n"
                    f"Data type: {col_dtype}\n"
                    f"Sample values: {sample_values}\n"
                    f"Technical metadata: {json.dumps(tech_metadata)}\n"
                    f"Lineage info: {json.dumps(lineage_info)}\n"
                    f"Business glossary terms: {json.dumps(glossary_terms)}\n"
                    f"Standard rules: {json.dumps(standard_rules)}\n"
                )
                system_prompt = (
                    "You are a data quality expert. For each column, recommend ALL relevant data quality rules that should be applied to this field, based on its metadata, business meaning, lineage, and standard practices. "
                    "Do NOT validate or analyze the data itself. Instead, suggest what checks (e.g., null check, format, range, uniqueness, allowed values, referential integrity, etc.) should be applied to this column to ensure data quality. "
                    "For each rule, provide a name, description, SQL condition (as a single line string), severity, and business impact. "
                    "Return your answer as a JSON array of objects, each with keys: rule_type, rule_name, description, sql_rule, severity, business_impact. "
                    "Respond ONLY with valid JSON, no explanation or markdown."
                )
                try:
                    response = openai.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=600
                    )
                    llm_content = response.choices[0].message.content
                    try:
                        # Attempt to fix common JSON issues (e.g., unterminated strings)
                        import re
                        fixed_content = llm_content
                        # Remove any newlines inside SQL strings
                        fixed_content = re.sub(r'("sql_rule"\s*:\s*")([^"]*?)(\n)([^"]*?")', lambda m: m.group(1) + m.group(2).replace('\n', ' ') + m.group(4), fixed_content)
                        # Remove triple quotes if present
                        fixed_content = fixed_content.replace('"""', '"')
                        # (Removed unnecessary backslash replacement for valid JSON escapes)
                        llm_json = json.loads(fixed_content)
                        for rule in llm_json:
                            recommendations.append({
                                'rule_type': rule.get('rule_type', 'custom'),
                                'rule_name': rule.get('rule_name', f'LLM Suggested Rule for {column}'),
                                'description': rule.get('description', ''),
                                'severity': rule.get('severity', 'INFO'),
                                'column': column,
                                'sql_rule': rule.get('sql_rule', ''),
                                'business_impact': rule.get('business_impact', ''),
                                'recommendation_reason': 'OpenAI GPT-4 LLM-based suggestion'
                            })
                    except Exception as json_err:
                        print(f"LLM raw output for column {column}: {llm_content}")
                        print(f"OpenAI LLM JSON error for column {column}: {json_err}")
                        # Fallback: recommend generic rules if LLM output is invalid
                        recommendations.append({
                            'rule_type': 'generic',
                            'rule_name': f'Null Check for {column}',
                            'description': f'Recommend checking for nulls in {column}.',
                            'severity': 'MEDIUM',
                            'column': column,
                            'sql_rule': f'SELECT COUNT(*) FROM {{table}} WHERE {column} IS NULL',
                            'business_impact': 'Missing values can impact data quality.',
                            'recommendation_reason': 'Fallback: LLM output not valid JSON.'
                        })
                        recommendations.append({
                            'rule_type': 'generic',
                            'rule_name': f'Uniqueness Check for {column}',
                            'description': f'Recommend checking for duplicate values in {column}.',
                            'severity': 'LOW',
                            'column': column,
                            'sql_rule': f'SELECT {column}, COUNT(*) FROM {{table}} GROUP BY {column} HAVING COUNT(*) > 1',
                            'business_impact': 'Duplicates may violate uniqueness constraints.',
                            'recommendation_reason': 'Fallback: LLM output not valid JSON.'
                        })
                except Exception as e:
                    print(f"OpenAI LLM error for column {column}: {e}")
                    recommendations.append({
                        'rule_type': 'custom',
                        'rule_name': f'LLM Suggested Rule for {column}',
                        'description': f'Could not get LLM recommendation: {e}',
                        'severity': 'INFO',
                        'column': column,
                        'sql_rule': '',
                        'business_impact': '',
                        'recommendation_reason': 'OpenAI GPT-4 LLM error'
                    })

            # Dataset-level rules via LLM
            dataset_prompt = (
                f"You are a data quality expert. Given the following dataset info, recommend ALL relevant dataset-level data quality rules (including standard and custom), their descriptions, and SQL checks.\n"
                f"Dataset info: {json.dumps(dataset_info)}\n"
                f"Return your answer as a JSON array of objects, each with keys: rule_type, rule_name, description, sql_rule, severity, business_impact."
            )
            try:
                response = openai.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": dataset_prompt}],
                    max_tokens=600
                )
                llm_content = response.choices[0].message.content
                llm_json = json.loads(llm_content)
                for rule in llm_json:
                    recommendations.append({
                        'rule_type': rule.get('rule_type', 'custom'),
                        'rule_name': rule.get('rule_name', 'LLM Dataset Rule'),
                        'description': rule.get('description', ''),
                        'severity': rule.get('severity', 'INFO'),
                        'column': 'ALL_COLUMNS',
                        'sql_rule': rule.get('sql_rule', ''),
                        'business_impact': rule.get('business_impact', ''),
                        'recommendation_reason': 'OpenAI GPT-4 LLM-based suggestion'
                    })
            except Exception as e:
                print(f"OpenAI LLM error for dataset-level rules: {e}")
                recommendations.append({
                    'rule_type': 'custom',
                    'rule_name': 'LLM Dataset Rule',
                    'description': f'Could not get LLM dataset-level recommendation: {e}',
                    'severity': 'INFO',
                    'column': 'ALL_COLUMNS',
                    'sql_rule': '',
                    'business_impact': '',
                    'recommendation_reason': 'OpenAI GPT-4 LLM error'
                })

            print(f"Total LLM rules recommended: {len(recommendations)}")
            return recommendations
        except Exception as e:
            print(f"Error in LLM recommendation engine: {e}")
            return []

    def _get_dataset_info(self, df):
        """Extract key dataset characteristics with detailed error logging"""
        try:
            # Convert dtypes to strings to avoid serialization issues
            dtype_dict = {str(k): str(v) for k, v in df.dtypes.to_dict().items()}
            
            dataset_info = {
                'shape': tuple(int(x) for x in df.shape),  # Convert shape to regular tuple of ints
                'columns': list(str(col) for col in df.columns),  # Convert column names to strings
                'dtypes': dtype_dict,
                'missing_values': {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()},
                'duplicate_rows': int(df.duplicated().sum()),
                'numeric_columns': [str(col) for col in df.select_dtypes(include=[np.number]).columns],
                'categorical_columns': [str(col) for col in df.select_dtypes(include=['object']).columns],
                'date_columns': [str(col) for col in df.select_dtypes(include=['datetime64']).columns]
            }
            return dataset_info
        except Exception as e:
            print(f"Error in _get_dataset_info: {str(e)}")
            print("Detailed error information:")
            for k, v in locals().items():
                try:
                    json.dumps({k: v})
                except TypeError as json_err:
                    print(f"Non-serializable object found - Key: {k}, Type: {type(v)}, Error: {str(json_err)}")
            return {}

    def _analyze_column(self, column, df, technical_metadata, data_lineage, dataset_info):
        """Analyze individual column and recommend rules"""
        recommendations = []

        # Get column data
        col_data = df[column]
        col_dtype = str(col_data.dtype)
        missing_count = col_data.isnull().sum()
        missing_percentage = (missing_count / len(df)) * 100

        # Find relevant business glossary terms
        glossary_results = self.business_glossary_collection.query(
            query_texts=[f"column name {column} data field"],
            n_results=3
        )

        # Get technical metadata for this column
        tech_metadata = self._get_column_tech_metadata(column, technical_metadata)

        # 1. Completeness Rules
        if missing_count > 0:
            if tech_metadata.get('nullable', True) == False:
                # Critical: Non-nullable field has missing values
                recommendations.append({
                    'rule_type': 'completeness',
                    'rule_name': f'Null Check - {column}',
                    'description': f'Column {column} is defined as NOT NULL but has {missing_count} missing values ({missing_percentage:.1f}%)',
                    'severity': 'CRITICAL',
                    'column': column,
                    'sql_rule': f"SELECT COUNT(*) as violations FROM {{table}} WHERE {column} IS NULL",
                    'expected_threshold': 0,
                    'current_violation_count': missing_count,
                    'business_impact': 'High - violates database constraints',
                    'recommendation_reason': 'Technical metadata indicates this field should not be null'
                })
            elif missing_percentage > 20:
                # High missing percentage
                recommendations.append({
                    'rule_type': 'completeness', 
                    'rule_name': f'High Missing Rate - {column}',
                    'description': f'Column {column} has high missing rate of {missing_percentage:.1f}%',
                    'severity': 'HIGH',
                    'column': column,
                    'sql_rule': f"SELECT (COUNT(*) - COUNT({column})) * 100.0 / COUNT(*) as missing_percentage FROM {{table}}",
                    'expected_threshold': 20.0,
                    'current_violation_count': missing_count,
                    'business_impact': 'Medium - may impact analysis quality',
                    'recommendation_reason': 'High missing percentage affects data usability'
                })

        # 2. Validity Rules based on data type and patterns
        if col_dtype == 'object':  # String columns
            if 'email' in column.lower():
                # Email validation
                invalid_emails = self._count_invalid_emails(col_data)
                if invalid_emails > 0:
                    recommendations.append({
                        'rule_type': 'validity',
                        'rule_name': f'Email Format Validation - {column}',
                        'description': f'Column {column} contains {invalid_emails} invalid email formats',
                        'severity': 'HIGH',
                        'column': column,
                        'sql_rule': f"SELECT COUNT(*) as violations FROM {{table}} WHERE {column} IS NOT NULL AND {column} NOT REGEXP '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{{2,}}$'",
                        'expected_threshold': 0,
                        'current_violation_count': invalid_emails,
                        'business_impact': 'High - invalid emails affect communication',
                        'recommendation_reason': 'Email field detected with format violations'
                    })

            # Check for allowed values if defined in technical metadata
            allowed_values = tech_metadata.get('allowed_values', [])
            if allowed_values:
                invalid_values = col_data.dropna()[~col_data.dropna().isin(allowed_values)]
                if len(invalid_values) > 0:
                    recommendations.append({
                        'rule_type': 'validity',
                        'rule_name': f'Domain Value Validation - {column}',
                        'description': f'Column {column} contains {len(invalid_values)} values outside allowed domain',
                        'severity': 'MEDIUM',
                        'column': column,
                        'sql_rule': f"SELECT COUNT(*) as violations FROM {{table}} WHERE {column} NOT IN ({', '.join(["'" + v + "'" for v in allowed_values])})",
                        'expected_threshold': 0,
                        'current_violation_count': len(invalid_values),
                        'business_impact': 'Medium - data consistency issues',
                        'recommendation_reason': 'Technical metadata defines allowed values'
                    })

        elif col_dtype in ['int64', 'float64']:  # Numeric columns
            # Range validation
            min_val = tech_metadata.get('min_value')
            max_val = tech_metadata.get('max_value')

            if min_val is not None or max_val is not None:
                actual_min = col_data.min()
                actual_max = col_data.max()

                violations = 0
                range_issues = []

                if min_val is not None and actual_min < min_val:
                    violations += (col_data < min_val).sum()
                    range_issues.append(f'values below {min_val}')

                if max_val is not None and actual_max > max_val:
                    violations += (col_data > max_val).sum()
                    range_issues.append(f'values above {max_val}')

                if violations > 0:
                    recommendations.append({
                        'rule_type': 'validity',
                        'rule_name': f'Range Validation - {column}',
                        'description': f'Column {column} has {violations} values outside expected range: {", ".join(range_issues)}',
                        'severity': 'MEDIUM',
                        'column': column,
                        'sql_rule': f"SELECT COUNT(*) as violations FROM {{table}} WHERE {column} IS NOT NULL AND ({column} < {min_val or 'NULL'} OR {column} > {max_val or 'NULL'})",
                        'expected_threshold': 0,
                        'current_violation_count': violations,
                        'business_impact': 'Medium - values outside business rules',
                        'recommendation_reason': 'Technical metadata defines value ranges'
                    })

            # Statistical outlier detection for numeric data
            if len(col_data.dropna()) > 10:  # Need sufficient data points
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                outlier_threshold = 1.5 * iqr
                outliers = col_data[(col_data < q1 - outlier_threshold) | (col_data > q3 + outlier_threshold)]

                if len(outliers) > 0:
                    recommendations.append({
                        'rule_type': 'accuracy',
                        'rule_name': f'Outlier Detection - {column}',
                        'description': f'Column {column} contains {len(outliers)} statistical outliers',
                        'severity': 'LOW',
                        'column': column,
                        'sql_rule': f"WITH quartiles AS (SELECT PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {column}) as q1, PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {column}) as q3 FROM {{table}}) SELECT COUNT(*) as violations FROM {{table}}, quartiles WHERE {column} < q1 - 1.5*(q3-q1) OR {column} > q3 + 1.5*(q3-q1)",
                        'expected_threshold': len(df) * 0.05,  # 5% threshold
                        'current_violation_count': len(outliers),
                        'business_impact': 'Low - may indicate data entry errors',
                        'recommendation_reason': 'Statistical analysis detected unusual values'
                    })

        # 3. Consistency Rules
        if tech_metadata.get('primary_key') or tech_metadata.get('unique'):
            # Uniqueness check
            duplicates = col_data.duplicated().sum()
            if duplicates > 0:
                recommendations.append({
                    'rule_type': 'consistency',
                    'rule_name': f'Uniqueness Validation - {column}',
                    'description': f'Column {column} should be unique but has {duplicates} duplicate values',
                    'severity': 'CRITICAL',
                    'column': column,
                    'sql_rule': f"SELECT {column}, COUNT(*) as duplicate_count FROM {{table}} GROUP BY {column} HAVING COUNT(*) > 1",
                    'expected_threshold': 0,
                    'current_violation_count': duplicates,
                    'business_impact': 'High - violates uniqueness constraints',
                    'recommendation_reason': 'Technical metadata indicates this field should be unique'
                })

        return recommendations

    def _get_dataset_level_recommendations(self, df, dataset_info):
        """Generate dataset-level recommendations"""
        recommendations = []

        # Check for duplicate rows
        if dataset_info['duplicate_rows'] > 0:
            recommendations.append({
                'rule_type': 'consistency',
                'rule_name': 'Duplicate Row Detection',
                'description': f'Dataset contains {dataset_info["duplicate_rows"]} duplicate rows',
                'severity': 'MEDIUM',
                'column': 'ALL_COLUMNS',
                'sql_rule': "SELECT COUNT(*) - COUNT(DISTINCT *) as duplicate_rows FROM {table}",
                'expected_threshold': 0,
                'current_violation_count': dataset_info['duplicate_rows'],
                'business_impact': 'Medium - may skew analysis results',
                'recommendation_reason': 'Duplicate rows detected in dataset'
            })

        return recommendations

    def _get_column_tech_metadata(self, column, technical_metadata):
        """Get technical metadata for a specific column"""
        if not technical_metadata:
            return {}

        columns_metadata = technical_metadata.get('columns', [])
        for col_meta in columns_metadata:
            if col_meta.get('name') == column:
                return col_meta
        return {}

    def _count_invalid_emails(self, email_series):
        """Count invalid email addresses in a series"""
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

        valid_emails = 0
        for email in email_series.dropna():
            if isinstance(email, str) and email.strip():
                if re.match(email_pattern, email.strip()):
                    valid_emails += 1

        total_non_null = email_series.dropna().count()
        return total_non_null - valid_emails

# Initialize the recommendation engine
recommendation_engine = DataQualityRuleRecommendationEngine()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and file.filename.lower().endswith('.csv'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the uploaded file
        return process_csv_file(filepath, filename)

    return jsonify({'error': 'Invalid file type. Please upload a CSV file.'})

def convert_all_numpy(obj):
    """Recursively convert all numpy and pandas types in a data structure to JSON serializable types"""
    if isinstance(obj, dict):
        return {k: convert_all_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_all_numpy(v) for v in obj]
    else:
        return convert_to_serializable(obj)

def process_csv_file(filepath, filename):
    try:
        print(f"Processing file: {filename}")
        # Read the CSV file
        df = pd.read_csv(filepath)
        print(f"Successfully read CSV file with shape: {df.shape}")

        # Load supporting metadata files
        technical_metadata = load_json_file('technical_metadata.json')
        data_lineage = load_json_file('data_lineage.json')
        business_glossary = load_json_file('business_glossary.json')
        print("Successfully loaded metadata files")

        # Generate profiling report using ydata-profiling
        from ydata_profiling import ProfileReport
        profile = ProfileReport(df, title=f"Profiling Report - {filename}", minimal=True)

        # Get data quality rule recommendations
        recommendations = recommendation_engine.analyze_data_and_recommend_rules(
            df, technical_metadata, data_lineage
        )
        
        # Log recommendations
        print(f"\nGenerated {len(recommendations)} recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"\nRecommendation {i}:")
            print(f"Type: {rec['rule_type']}")
            print(f"Name: {rec['rule_name']}")
            print(f"Severity: {rec['severity']}")
            print(f"Column: {rec['column']}")
            print(f"Description: {rec['description']}")

        print("\nPreparing response data")
        # Convert all numpy types to Python native types
        dataset_info = {
            'rows': int(len(df)),
            'columns': int(len(df.columns)),
            'column_names': df.columns.tolist(),
            'data_types': {k: str(v) for k, v in df.dtypes.items()},
            'missing_values': {k: convert_to_serializable(v) for k, v in df.isnull().sum().items()},
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB"
        }
        # Prepare response data
        response_data = {
            'filename': filename,
            'dataset_info': dataset_info,
            'recommendations': recommendations,
            'recommendation_summary': {
                'total_rules': len(recommendations),
                'critical_rules': len([r for r in recommendations if r['severity'] == 'CRITICAL']),
                'high_rules': len([r for r in recommendations if r['severity'] == 'HIGH']),
                'medium_rules': len([r for r in recommendations if r['severity'] == 'MEDIUM']),
                'low_rules': len([r for r in recommendations if r['severity'] == 'LOW'])
            }
        }
        # Recursively convert all numpy types in the response data
        response_data = convert_all_numpy(response_data)
        print("Successfully prepared response data")
        return jsonify(response_data)

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error processing file: {str(e)}")
        print(f"Full traceback:\n{error_trace}")
        return jsonify({
            'error': f'Error processing file: {str(e)}',
            'traceback': error_trace
        })

def load_json_file(filename):
    """Load JSON file with error handling"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {filename} not found")
        return None
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

@app.route('/api/rules/<rule_type>')
def get_rules_by_type(rule_type):
    """API endpoint to get standard rules by type"""
    standard_rules = load_json_file('standard_rules.json')
    if standard_rules and rule_type in standard_rules.get('rule_categories', {}):
        return jsonify(standard_rules['rule_categories'][rule_type])
    return jsonify({'error': 'Rule type not found'})

@app.route('/api/glossary/<term_id>')
def get_glossary_term(term_id):
    """API endpoint to get business glossary term"""
    business_glossary = load_json_file('business_glossary.json')
    if business_glossary:
        for term in business_glossary.get('glossary', {}).get('terms', []):
            if term['id'] == term_id:
                return jsonify(term)
    return jsonify({'error': 'Term not found'})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
