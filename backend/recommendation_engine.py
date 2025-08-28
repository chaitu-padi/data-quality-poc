import json
import pandas as pd
from typing import List, Dict, Any
from .agents.rule_recommender import DataQualityRuleRecommender

class DataQualityRuleRecommendationEngine:
    def __init__(self, model_path: str = None):
        """Initialize the recommendation engine with LLM-based recommender"""
        self.recommender = DataQualityRuleRecommender(model_path)
        self.business_glossary = None
        self.technical_metadata = None
        self.data_lineage = None
        self.load_knowledge_base()

    def load_knowledge_base(self):
        """Load knowledge base from JSON files"""
        try:
            # Load business glossary
            with open('business_glossary.json', 'r') as f:
                self.business_glossary = json.load(f)
            
            # Load technical metadata
            with open('technical_metadata.json', 'r') as f:
                self.technical_metadata = json.load(f)
            
            # Load data lineage
            with open('data_lineage.json', 'r') as f:
                self.data_lineage = json.load(f)
                
            print("Knowledge base loaded successfully!")
        except Exception as e:
            print(f"Error loading knowledge base: {e}")

    def analyze_data_and_recommend_rules(
        self,
        df: pd.DataFrame,
        technical_metadata: Dict,
        data_lineage: Dict
    ) -> List[Dict]:
        """
        Analyze the dataset and recommend data quality rules using LLM
        """
        recommendations = []
        
        try:
            # Update metadata if provided
            if technical_metadata:
                self.technical_metadata = technical_metadata
            if data_lineage:
                self.data_lineage = data_lineage
            
            # Process each column
            for column in df.columns:
                column_recommendations = self.recommender.analyze_and_recommend(
                    df=df,
                    column_name=column,
                    business_glossary=self.business_glossary,
                    technical_metadata=self.technical_metadata,
                    data_lineage=self.data_lineage
                )
                recommendations.extend(column_recommendations)
            
            # Sort recommendations by severity
            severity_order = {
                'CRITICAL': 0,
                'HIGH': 1,
                'MEDIUM': 2,
                'LOW': 3
            }
            recommendations.sort(key=lambda x: severity_order.get(x['severity'], 999))
            
            return recommendations
        except Exception as e:
            print(f"Error in recommendation engine: {e}")
            return []

    def get_column_metadata(self, column_name: str) -> Dict[str, Any]:
        """Get combined metadata for a column"""
        metadata = {
            'business_glossary': self._get_business_glossary_info(column_name),
            'technical_metadata': self._get_technical_metadata_info(column_name),
            'data_lineage': self._get_data_lineage_info(column_name)
        }
        return metadata

    def _get_business_glossary_info(self, column_name: str) -> Dict:
        """Extract business glossary information for a column"""
        if not self.business_glossary:
            return {}
        
        for term in self.business_glossary.get('glossary', {}).get('terms', []):
            if column_name.lower() in term['name'].lower():
                return term
        return {}

    def _get_technical_metadata_info(self, column_name: str) -> Dict:
        """Extract technical metadata information for a column"""
        if not self.technical_metadata:
            return {}
        
        for col in self.technical_metadata.get('columns', []):
            if col.get('name') == column_name:
                return col
        return {}

    def _get_data_lineage_info(self, column_name: str) -> Dict:
        """Extract data lineage information for a column"""
        if not self.data_lineage:
            return {}
        
        return self.data_lineage.get(column_name, {})
