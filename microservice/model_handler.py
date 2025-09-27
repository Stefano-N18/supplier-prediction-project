import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os

class SupplierRecommender:
    def __init__(self):
        self.model = None
        self.label_encoders = None
        self.target_encoder = None
        self.feature_names = None
        self.df = None
        self.load_system()
    
    def load_system(self):
        """Cargar modelo y datos"""
        try:
            # Rutas de archivos
            model_path = 'models/decision_tree_model_fixed.pkl'
            encoders_path = 'models/label_encoders_fixed.pkl'
            target_path = 'models/target_encoder_fixed.pkl'
            features_path = 'models/feature_names_fixed.pkl'
            data_path = 'data/dewatering_realistic_supplier_dataset.csv'
            
            # Verificar que existen los archivos
            for path in [model_path, encoders_path, target_path, features_path, data_path]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Archivo requerido no encontrado: {path}")
            
            # Cargar modelo y encoders
            self.model = joblib.load(model_path)
            self.label_encoders = joblib.load(encoders_path)
            self.target_encoder = joblib.load(target_path)
            self.feature_names = joblib.load(features_path)
            
            # Cargar dataset
            self.df = pd.read_csv(data_path)
            
            print("✅ Sistema cargado correctamente")
            
        except Exception as e:
            print(f"❌ Error cargando sistema: {e}")
            raise e
    
    def get_available_products(self):
        """Obtener productos disponibles organizados por categoría"""
        products = self.df['product_type'].unique()
        
        # Separar por categoría
        sensor_types = ['pressure_sensor_analog', 'pressure_sensor_digital', 
                       'temperature_sensor_bimetal', 'sensor_inductivo', 'transmisor_presion']
        
        filtration_products = []
        sensor_products = []
        
        for product in products:
            if product in sensor_types:
                sensor_products.append(product)
            else:
                filtration_products.append(product)
        
        return {
            'filtration': sorted(filtration_products),
            'sensors': sorted(sensor_products)
        }
    
    def recommend_suppliers(self, product_type, urgency, quantity, budget):
        """Función principal de recomendación"""
        # Validar inputs
        valid_urgencies = ['Low', 'Medium', 'High', 'Critical']
        if urgency not in valid_urgencies:
            return {"error": f"Urgencia debe ser una de: {valid_urgencies}"}
        
        # Obtener productos disponibles
        available_products = self.df[self.df['product_type'] == product_type]
        if available_products.empty:
            available_types = self.df['product_type'].unique()
            return {"error": f"Producto no encontrado. Tipos disponibles: {list(available_types)}"}
        
        recommendations = []
        
        for supplier_name in available_products['supplier_name'].unique():
            supplier_data = available_products[available_products['supplier_name'] == supplier_name].iloc[0]
            
            # Crear input para el modelo
            input_data = pd.DataFrame({
                'price_usd': [supplier_data['price_usd']],
                'delivery_days': [supplier_data['delivery_days']],
                'payment_terms_days': [supplier_data['payment_terms_days']],
                'shipping_included': [int(supplier_data['shipping_included'])],
                'express_available': [int(supplier_data['express_available'])],
                'order_urgency': [urgency],
                'quantity_needed': [quantity],
                'budget_available': [budget],
                'product_type': [product_type],
                'incoterms': [supplier_data['incoterms']],
                'month': [datetime.now().month],
                'quarter': [f"Q{(datetime.now().month-1)//3 + 1}"]
            })
            
            # Aplicar encoding
            for feature in self.label_encoders.keys():
                if feature in input_data.columns:
                    try:
                        input_data[feature] = self.label_encoders[feature].transform(input_data[feature].astype(str))
                    except ValueError:
                        input_data[feature] = 0
            
            # Reordenar columnas
            input_data = input_data[self.feature_names]
            
            # Predicción
            probabilities = self.model.predict_proba(input_data)[0]
            supplier_idx = np.where(self.target_encoder.classes_ == supplier_name)[0]
            
            if len(supplier_idx) > 0:
                probability = probabilities[supplier_idx[0]]
            else:
                probability = 0.0
            
            # Calcular costos
            total_cost = supplier_data['price_usd'] * quantity
            if not supplier_data['shipping_included']:
                total_cost += 300
            
            # Crear recomendación
            recommendation = {
                'supplier_name': supplier_name,
                'country': supplier_data['country'],
                'quality_rating': float(supplier_data['quality_rating']),
                'price_usd': float(supplier_data['price_usd']),
                'total_cost': round(float(total_cost), 2),
                'delivery_days': int(supplier_data['delivery_days']),
                'payment_terms_days': int(supplier_data['payment_terms_days']),
                'probability_score': round(float(probability), 3),
                'within_budget': bool(total_cost <= budget),
                'shipping_included': bool(supplier_data['shipping_included']),
                'express_available': bool(supplier_data['express_available'])
            }
            
            # Score final
            quality_score = recommendation['quality_rating'] / 5.0
            price_score = 1.0 if recommendation['within_budget'] else 0.5
            delivery_score = 1.0 if recommendation['delivery_days'] <= 15 else 0.7
            payment_score = 1.0 if recommendation['payment_terms_days'] > 0 else 0.8
            
            final_score = (
                recommendation['probability_score'] * 0.4 +
                quality_score * 0.25 +
                price_score * 0.2 +
                delivery_score * 0.1 +
                payment_score * 0.05
            )
            
            recommendation['final_score'] = round(final_score, 3)
            
            # Nivel de recomendación
            if final_score >= 0.75:
                recommendation['recommendation_level'] = "Altamente Recomendado"
            elif final_score >= 0.6:
                recommendation['recommendation_level'] = "Recomendado"
            elif final_score >= 0.45:
                recommendation['recommendation_level'] = "Aceptable"
            else:
                recommendation['recommendation_level'] = "No Recomendado"
            
            recommendations.append(recommendation)
        
        # Ordenar por score
        recommendations.sort(key=lambda x: x['final_score'], reverse=True)
        
        return {
            'product_type': product_type,
            'urgency': urgency,
            'quantity': quantity,
            'budget': budget,
            'recommendations': recommendations,
            'total_options': len(recommendations)
        }