from pydantic import BaseModel
from typing import List, Optional, Dict

class Recommendation(BaseModel):
    rule_type: str
    rule_name: str
    description: str
    severity: str
    column: str
    sql_rule: str
    expected_threshold: Optional[float]
    current_violation_count: Optional[int]
    business_impact: Optional[str]
    recommendation_reason: Optional[str]

class DatasetInfo(BaseModel):
    rows: int
    columns: int
    column_names: List[str]
    data_types: Dict[str, str]
    missing_values: Dict[str, int]
    memory_usage: str

class RecommendationSummary(BaseModel):
    total_rules: int
    critical_rules: int
    high_rules: int
    medium_rules: int
    low_rules: int
