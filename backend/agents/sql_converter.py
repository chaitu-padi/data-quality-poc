from typing import List, Dict
import json
from llama_cpp import Llama

class SQLConverterAgent:
    def __init__(self, model_path: str = "models/llama-2-7b.Q4_K_M.gguf"):
        """Initialize the SQL converter agent with a local LLaMA model"""
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_batch=512,
        )

    def convert_to_spark_sql(self, recommendation: Dict) -> str:
        """Convert a data quality rule recommendation to Spark SQL"""
        prompt = self._create_conversion_prompt(recommendation)
        
        response = self.llm(
            prompt,
            max_tokens=512,
            stop=["```"],
            temperature=0.1
        )
        
        return self._extract_sql_from_response(response['choices'][0]['text'])

    def _create_conversion_prompt(self, recommendation: Dict) -> str:
        """Create a prompt for converting the recommendation to Spark SQL"""
        return f"""Convert this data quality rule to PySpark SQL. 
        The rule should be efficient and follow Spark best practices.

        Rule Information:
        - Type: {recommendation['rule_type']}
        - Name: {recommendation['rule_name']}
        - Description: {recommendation['description']}
        - Base SQL: {recommendation['sql_rule']}
        - Column: {recommendation['column']}
        - Expected Threshold: {recommendation.get('expected_threshold', 'N/A')}

        Return only the PySpark SQL code without any explanation.
        Start the response with ```sql and end with ```
        """

    def _extract_sql_from_response(self, response: str) -> str:
        """Extract the SQL code from the LLM response"""
        sql_code = response.strip()
        if sql_code.startswith("```sql"):
            sql_code = sql_code[6:]
        if sql_code.endswith("```"):
            sql_code = sql_code[:-3]
        return sql_code.strip()

    def batch_convert_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """Convert a batch of recommendations to Spark SQL"""
        for rec in recommendations:
            spark_sql = self.convert_to_spark_sql(rec)
            rec['spark_sql'] = spark_sql
        return recommendations
