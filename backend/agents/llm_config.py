from langchain.prompts import PromptTemplate
from langchain import LLMChain
from langchain.llms import LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
import os
from dotenv import load_dotenv

load_dotenv()

# LLM Templates
RULE_GENERATION_TEMPLATE = """
Analyze the given data and metadata to recommend data quality rules. Consider the following aspects:
1. Column data type and statistics
2. Business glossary terms and definitions
3. Technical metadata constraints
4. Standard industry rules
5. Data lineage information

Context Information:
Column Name: {column_name}
Data Type: {data_type}
Statistics: {statistics}
Business Context: {business_context}
Technical Metadata: {technical_metadata}
Data Lineage: {data_lineage}

Generate comprehensive data quality rules that include:
1. Rule name and type
2. Description and rationale
3. SQL validation query
4. Business impact
5. Severity level (CRITICAL, HIGH, MEDIUM, LOW)
6. Expected threshold
7. Implementation considerations

Response should be structured and focused on practical, implementable rules.

Rules:"""

CUSTOM_RULE_TEMPLATE = """
Based on the analysis of data patterns and business context, generate custom data quality rules.

Data Analysis:
{data_analysis}

Business Context:
{business_context}

Consider:
1. Unusual patterns or anomalies
2. Business domain specific requirements
3. Complex inter-column relationships
4. Domain value distributions
5. Time-based patterns

Generate innovative but practical custom rules that go beyond standard checks.

Custom Rules:"""

RULE_ENHANCEMENT_TEMPLATE = """
Enhance and refine the following data quality rule based on available context:

Original Rule:
{original_rule}

Additional Context:
{context}

Consider:
1. Rule effectiveness
2. Performance implications
3. Business relevance
4. Implementation complexity
5. False positive/negative rates

Provide enhanced version with:
1. Refined logic
2. Optimized SQL
3. Better thresholds
4. Additional context considerations

Enhanced Rule:"""

def initialize_llm(model_path: str = None):
    """Initialize the LLM with specified configuration"""
    if not model_path:
        model_path = os.getenv('LLM_MODEL_PATH', 'models/llama-2-7b.Q4_K_M.gguf')
    
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.2,
        max_tokens=2000,
        n_ctx=2048,
        callback_manager=callback_manager,
        verbose=True,
        n_batch=512,
        top_p=0.95,
    )
    
    return llm

def create_rule_chain(llm):
    """Create LangChain chain for rule generation"""
    prompt = PromptTemplate(
        template=RULE_GENERATION_TEMPLATE,
        input_variables=[
            "column_name",
            "data_type",
            "statistics",
            "business_context",
            "technical_metadata",
            "data_lineage"
        ]
    )
    
    return LLMChain(llm=llm, prompt=prompt)

def create_custom_rule_chain(llm):
    """Create LangChain chain for custom rule generation"""
    prompt = PromptTemplate(
        template=CUSTOM_RULE_TEMPLATE,
        input_variables=["data_analysis", "business_context"]
    )
    
    return LLMChain(llm=llm, prompt=prompt)

def create_enhancement_chain(llm):
    """Create LangChain chain for rule enhancement"""
    prompt = PromptTemplate(
        template=RULE_ENHANCEMENT_TEMPLATE,
        input_variables=["original_rule", "context"]
    )
    
    return LLMChain(llm=llm, prompt=prompt)
