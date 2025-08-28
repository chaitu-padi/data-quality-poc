from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
from .recommendation_engine import DataQualityRuleRecommendationEngine
from .data_loader import load_csv, load_json_file, generate_profile
from .models import Recommendation, DatasetInfo, RecommendationSummary

router = APIRouter()
engine = DataQualityRuleRecommendationEngine()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.csv'):
        return JSONResponse(content={"error": "Invalid file type. Please upload a CSV file."})
    contents = await file.read()
    filepath = f"uploads/{file.filename}"
    with open(filepath, "wb") as f:
        f.write(contents)
    df = load_csv(filepath)
    technical_metadata = load_json_file('technical_metadata.json')
    data_lineage = load_json_file('data_lineage.json')
    business_glossary = load_json_file('business_glossary.json')
    profile = generate_profile(df, title=f"Profiling Report - {file.filename}")
    recommendations = engine.analyze_data_and_recommend_rules(df, technical_metadata, data_lineage)
    response_data = {
        "filename": file.filename,
        "dataset_info": {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB"
        },
        "recommendations": recommendations,
        "recommendation_summary": {
            "total_rules": len(recommendations),
            "critical_rules": len([r for r in recommendations if r['severity'] == 'CRITICAL']),
            "high_rules": len([r for r in recommendations if r['severity'] == 'HIGH']),
            "medium_rules": len([r for r in recommendations if r['severity'] == 'MEDIUM']),
            "low_rules": len([r for r in recommendations if r['severity'] == 'LOW'])
        }
    }
    return JSONResponse(content=response_data)
