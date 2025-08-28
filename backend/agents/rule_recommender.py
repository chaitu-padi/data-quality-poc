from typing import List, Dict, Any
import json
import numpy as np
import pandas as pd
from .llm_config import (
    initialize_llm,
    create_rule_chain,
    create_custom_rule_chain,
    create_enhancement_chain
)

class DataQualityRuleRecommender:
    def __init__(self, model_path: str = None):
        """Initialize the LLM-based rule recommendation engine"""
        self.llm = initialize_llm(model_path)
        self.rule_chain = create_rule_chain(self.llm)
        self.custom_rule_chain = create_custom_rule_chain(self.llm)
        self.enhancement_chain = create_enhancement_chain(self.llm)
        
    def analyze_and_recommend(
        self,
        df: pd.DataFrame,
        column_name: str,
        business_glossary: Dict,
        technical_metadata: Dict,
        data_lineage: Dict
    ) -> List[Dict]:
        """Generate data quality rules for a specific column using LLM"""
        
        # Prepare column statistics
        stats = self._get_column_statistics(df[column_name])
        
        # Get business context
        business_context = self._extract_business_context(
            column_name, business_glossary
        )
        
        # Get technical metadata
        tech_metadata = self._extract_technical_metadata(
            column_name, technical_metadata
        )
        
        # Generate base rules
        rules = self._generate_base_rules(
            column_name=column_name,
            data_type=str(df[column_name].dtype),
            statistics=json.dumps(stats),
            business_context=business_context,
            technical_metadata=tech_metadata,
            data_lineage=json.dumps(data_lineage.get(column_name, {}))
        )
        
        # Generate custom rules based on data patterns
        custom_rules = self._generate_custom_rules(
            df[column_name],
            business_context
        )
        
        # Combine and enhance rules
        all_rules = rules + custom_rules
        enhanced_rules = self._enhance_rules(all_rules, df[column_name])
        
        return enhanced_rules

    def _get_column_statistics(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate comprehensive statistics for a column"""
        stats = {
            "count": len(series),
            "null_count": series.isnull().sum(),
            "null_percentage": (series.isnull().sum() / len(series)) * 100,
            "unique_count": series.nunique(),
            "unique_percentage": (series.nunique() / len(series)) * 100
        }
        
        if np.issubdtype(series.dtype, np.number):
            stats.update({
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "quartiles": series.quantile([0.25, 0.5, 0.75]).to_dict()
            })
        elif series.dtype == 'object' or series.dtype == 'string':
            value_counts = series.value_counts()
            stats.update({
                "top_values": value_counts.head(5).to_dict(),
                "avg_length": series.str.len().mean(),
                "max_length": series.str.len().max()
            })
        
        return stats

    def _extract_business_context(
        self,
        column_name: str,
        business_glossary: Dict
    ) -> str:
        """Extract relevant business context from glossary"""
        context = []
        
        if not business_glossary:
            return "No business glossary available"
            
        for term in business_glossary.get('glossary', {}).get('terms', []):
            if (
                column_name.lower() in term['name'].lower() or
                any(column_name.lower() in alias.lower() 
                    for alias in term.get('aliases', []))
            ):
                context.append(f"Term: {term['name']}")
                context.append(f"Definition: {term['definition']}")
                if 'business_rules' in term:
                    context.append("Business Rules: " + 
                                 "; ".join(term['business_rules']))
                if 'domain' in term:
                    context.append(f"Domain: {term['domain']}")
                context.append("---")
                
        return "\n".join(context) if context else "No direct business context found"

    def _extract_technical_metadata(
        self,
        column_name: str,
        technical_metadata: Dict
    ) -> str:
        """Extract technical metadata for the column"""
        if not technical_metadata:
            return "No technical metadata available"
            
        for col in technical_metadata.get('columns', []):
            if col.get('name') == column_name:
                metadata = []
                for key, value in col.items():
                    if key != 'name':
                        metadata.append(f"{key}: {value}")
                return "\n".join(metadata)
                
        return "No specific technical metadata found"

    def _generate_base_rules(self, **kwargs) -> List[Dict]:
        """Generate base rules using LLM"""
        try:
            result = self.rule_chain.run(kwargs)
            return self._parse_llm_rules_response(result)
        except Exception as e:
            print(f"Error generating base rules: {e}")
            return []

    def _generate_custom_rules(
        self,
        series: pd.Series,
        business_context: str
    ) -> List[Dict]:
        """Generate custom rules based on data patterns"""
        try:
            # Analyze data patterns
            analysis = self._analyze_data_patterns(series)
            
            result = self.custom_rule_chain.run({
                "data_analysis": json.dumps(analysis),
                "business_context": business_context
            })
            
            return self._parse_llm_rules_response(result)
        except Exception as e:
            print(f"Error generating custom rules: {e}")
            return []

    def _analyze_data_patterns(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze complex patterns in the data"""
        patterns = {}
        
        if np.issubdtype(series.dtype, np.number):
            # Analyze numerical patterns
            patterns['distribution_type'] = self._detect_distribution(series)
            patterns['seasonality'] = self._detect_seasonality(series)
            patterns['outliers'] = self._detect_outliers(series)
        elif series.dtype == 'object' or series.dtype == 'string':
            # Analyze string patterns
            patterns['common_prefixes'] = self._detect_common_prefixes(series)
            patterns['common_formats'] = self._detect_common_formats(series)
            patterns['pattern_consistency'] = self._check_pattern_consistency(series)
            
        return patterns

    def _enhance_rules(
        self,
        rules: List[Dict],
        series: pd.Series
    ) -> List[Dict]:
        """Enhance rules with additional context and optimizations"""
        enhanced_rules = []
        
        for rule in rules:
            try:
                # Create enhancement context
                context = {
                    "data_sample": series.head(100).to_dict(),
                    "value_distribution": series.value_counts().head(10).to_dict(),
                    "current_violations": self._check_current_violations(rule, series)
                }
                
                result = self.enhancement_chain.run({
                    "original_rule": json.dumps(rule),
                    "context": json.dumps(context)
                })
                
                enhanced_rule = self._parse_llm_rules_response(result)[0]
                enhanced_rules.append(enhanced_rule)
            except Exception as e:
                print(f"Error enhancing rule: {e}")
                enhanced_rules.append(rule)
                
        return enhanced_rules

    def _parse_llm_rules_response(self, response: str) -> List[Dict]:
        """Parse LLM response into structured rules"""
        try:
            # Basic cleanup
            response = response.strip()
            if response.startswith('```') and response.endswith('```'):
                response = response[3:-3]
                
            # Try to parse as JSON first
            try:
                return json.loads(response)
            except:
                pass
            
            # If not JSON, parse structured text
            rules = []
            current_rule = {}
            
            for line in response.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith('Rule ') or line.startswith('- Rule '):
                    if current_rule:
                        rules.append(current_rule)
                    current_rule = {'rule_name': line.split(':', 1)[1].strip()}
                elif ': ' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace(' ', '_')
                    current_rule[key] = value.strip()
                    
            if current_rule:
                rules.append(current_rule)
                
            return rules
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return []

    def _detect_distribution(self, series: pd.Series) -> str:
        """Detect the statistical distribution of numerical data"""
        # Implementation for distribution detection
        return "normal"  # Placeholder

    def _detect_seasonality(self, series: pd.Series) -> Dict:
        """Detect seasonal patterns in time series data"""
        # Implementation for seasonality detection
        return {"has_seasonality": False}  # Placeholder

    def _detect_outliers(self, series: pd.Series) -> Dict:
        """Detect outliers using statistical methods"""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        
        return {
            "count": len(outliers),
            "percentage": (len(outliers) / len(series)) * 100,
            "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)}
        }

    def _detect_common_prefixes(self, series: pd.Series) -> Dict:
        """Detect common prefixes in string data"""
        prefixes = series.str[:3].value_counts()
        return prefixes.head(5).to_dict()

    def _detect_common_formats(self, series: pd.Series) -> Dict:
        """Detect common string formats/patterns"""
        # Implementation for format detection
        return {"formats": []}  # Placeholder

    def _check_pattern_consistency(self, series: pd.Series) -> Dict:
        """Check consistency of patterns in string data"""
        # Implementation for pattern consistency checking
        return {"consistency_score": 0.0}  # Placeholder

    def _check_current_violations(self, rule: Dict, series: pd.Series) -> Dict:
        """Check current violations for a rule"""
        # Implementation for violation checking
        return {"violation_count": 0}  # Placeholder
