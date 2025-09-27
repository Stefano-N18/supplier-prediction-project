from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
from model_handler import SupplierRecommender

# Inicializar FastAPI
app = FastAPI(
    title="Dewatering Solutions - Supplier Recommender API",
    description="Sistema de recomendación de proveedores basado en árboles de decisión",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar el sistema de recomendación
recommender = None

@app.on_event("startup")
async def startup_event():
    global recommender
    try:
        recommender = SupplierRecommender()
        print("✅ Sistema de recomendación cargado correctamente")
    except Exception as e:
        print(f"❌ Error al cargar sistema: {e}")
        raise e

# Modelos Pydantic para requests/responses
class RecommendationRequest(BaseModel):
    product_type: str
    urgency: str
    quantity: int
    budget: float = 10000

class SupplierRecommendation(BaseModel):
    supplier_name: str
    country: str
    quality_rating: float
    price_usd: float
    total_cost: float
    delivery_days: int
    payment_terms_days: int
    probability_score: float
    final_score: float
    recommendation_level: str
    within_budget: bool

class RecommendationResponse(BaseModel):
    product_type: str
    urgency: str
    quantity: int
    budget: float
    recommendations: List[SupplierRecommendation]
    total_options: int

# ENDPOINTS

@app.get("/")
async def root():
    return {
        "message": "Dewatering Solutions - Supplier Recommender API",
        "version": "1.0.0",
        "status": "running",
        "author": "Dewatering Solutions Team"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": recommender is not None}

@app.get("/products")
async def get_available_products():
    """Obtener lista de productos disponibles"""
    if not recommender:
        raise HTTPException(status_code=500, detail="Sistema no inicializado")
    
    try:
        products = recommender.get_available_products()
        return {
            "filtration_products": products["filtration"],
            "sensor_products": products["sensors"],
            "total_products": len(products["filtration"]) + len(products["sensors"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo productos: {str(e)}")

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_suppliers(request: RecommendationRequest):
    """Obtener recomendaciones de proveedores"""
    if not recommender:
        raise HTTPException(status_code=500, detail="Sistema no inicializado")
    
    # Validar urgencia
    valid_urgencies = ['Low', 'Medium', 'High', 'Critical']
    if request.urgency not in valid_urgencies:
        raise HTTPException(
            status_code=400, 
            detail=f"Urgencia debe ser una de: {valid_urgencies}"
        )
    
    # Validar parámetros
    if request.quantity <= 0:
        raise HTTPException(status_code=400, detail="Cantidad debe ser mayor a 0")
    
    if request.budget <= 0:
        raise HTTPException(status_code=400, detail="Presupuesto debe ser mayor a 0")
    
    try:
        result = recommender.recommend_suppliers(
            request.product_type,
            request.urgency, 
            request.quantity,
            request.budget
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en recomendación: {str(e)}")

@app.get("/test-scenarios")
async def get_test_scenarios():
    """Obtener escenarios de prueba predefinidos"""
    scenarios = [
        {
            "name": "Rollos Premium vs Económicos",
            "params": {
                "product_type": "filter_cloth_roll",
                "urgency": "Low",
                "quantity": 2,
                "budget": 8000
            }
        },
        {
            "name": "Filtros Prensa Estándar", 
            "params": {
                "product_type": "filter_press_cloth_set",
                "urgency": "Medium",
                "quantity": 30,
                "budget": 800
            }
        },
        {
            "name": "Válvulas Urgentes",
            "params": {
                "product_type": "valve_butterfly",
                "urgency": "Critical",
                "quantity": 1,
                "budget": 600
            }
        }
    ]
    return {"test_scenarios": scenarios}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)