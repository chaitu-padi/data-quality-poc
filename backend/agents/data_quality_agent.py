from typing import List, Dict
from abc import ABC, abstractmethod
from .sql_converter import SQLConverterAgent
from .rag_agent import RAGAgent

class BaseAgent(ABC):
    @abstractmethod
    def process_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        pass

class DataQualityAgent:
    def __init__(self, model_path: str = "models/llama-2-7b.Q4_K_M.gguf"):
        """Initialize the main data quality agent with sub-agents"""
        self.sql_converter = SQLConverterAgent(model_path)
        self.rag_agent = RAGAgent(model_path)

    def process_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """Process recommendations through the agent pipeline"""
        # Step 1: Enhance recommendations with RAG
        enhanced_recs = self.rag_agent.enhance_recommendations(recommendations)
        
        # Step 2: Convert to Spark SQL
        final_recs = self.sql_converter.batch_convert_recommendations(enhanced_recs)
        
        return final_recs

    def get_explanation(self, recommendation: Dict) -> str:
        """Get human-readable explanation of a recommendation"""
        prompt = f"""
        Explain this data quality rule in simple terms:
        
        Rule: {recommendation['rule_name']}
        Type: {recommendation['rule_type']}
        Description: {recommendation['description']}
        Impact: {recommendation['business_impact']}
        
        Enhanced Context: {recommendation.get('enhanced_context', '')}
        """
        
        # Use RAG agent's LLM to generate explanation
        explanation = self.rag_agent.llm(prompt, max_tokens=256)
        return explanation['choices'][0]['text'].strip()

    def suggest_fixes(self, recommendation: Dict) -> List[str]:
        """Suggest potential fixes for a data quality issue"""
        prompt = f"""
        Suggest fixes for this data quality issue:
        
        Rule: {recommendation['rule_name']}
        Description: {recommendation['description']}
        Current Violations: {recommendation['current_violation_count']}
        Enhanced Context: {recommendation.get('enhanced_context', '')}
        
        Return a list of specific, actionable fixes.
        """
        
        response = self.rag_agent.llm(prompt, max_tokens=512)
        fixes = response['choices'][0]['text'].strip().split('\n')
        return [fix.strip('- ') for fix in fixes if fix.strip()]
