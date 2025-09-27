# PROGRAMA INTERACTIVO DE RECOMENDACIÓN DE PROVEEDORES
# Dewatering Solutions - Sistema de Predicción

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os
import sys

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
            # Cargar modelo corregido
            self.model = joblib.load('models/decision_tree_model_fixed.pkl')
            self.label_encoders = joblib.load('models/label_encoders_fixed.pkl')
            self.target_encoder = joblib.load('models/target_encoder_fixed.pkl')
            self.feature_names = joblib.load('models/feature_names_fixed.pkl')
            
            # Cargar dataset
            self.df = pd.read_csv('data/dewatering_realistic_supplier_dataset.csv')
            
            print("✅ Sistema cargado correctamente")
            
        except FileNotFoundError as e:
            print("❌ Error: Archivos del modelo no encontrados")
            print("   Asegúrate de estar en la carpeta del proyecto")
            print("   Y de haber ejecutado el notebook 02b_fix_overfitting.ipynb")
            sys.exit(1)
    
    def get_available_products(self):
        """Obtener productos disponibles organizados por categoría"""
        products = self.df['product_type'].unique()
        
        # Separar por categoría
        filtration_products = []
        sensor_products = []
        
        sensor_types = ['pressure_sensor_analog', 'pressure_sensor_digital', 
                       'temperature_sensor_bimetal', 'sensor_inductivo', 'transmisor_presion']
        
        for product in products:
            if product in sensor_types:
                sensor_products.append(product)
            else:
                filtration_products.append(product)
        
        return {
            'filtration': sorted(filtration_products),
            'sensors': sorted(sensor_products)
        }
    
    def show_product_menu(self):
        """Mostrar menú de productos"""
        products = self.get_available_products()
        
        print("\n" + "="*60)
        print("PRODUCTOS DISPONIBLES")
        print("="*60)
        
        option_num = 1
        product_map = {}
        
        print("\n🔧 PRODUCTOS DE FILTRACIÓN:")
        for product in products['filtration']:
            # Mostrar productos con proveedores disponibles
            suppliers = self.df[self.df['product_type'] == product]['supplier_name'].unique()
            print(f"{option_num:2d}. {product}")
            print(f"    Proveedores: {', '.join(suppliers)}")
            product_map[option_num] = product
            option_num += 1
        
        print("\n⚡ PRODUCTOS DE SENSORES:")
        for product in products['sensors']:
            suppliers = self.df[self.df['product_type'] == product]['supplier_name'].unique()
            print(f"{option_num:2d}. {product}")
            print(f"    Proveedores: {', '.join(suppliers)}")
            product_map[option_num] = product
            option_num += 1
        
        return product_map
    
    def get_user_input(self):
        """Obtener parámetros del usuario"""
        print("\n" + "="*60)
        print("CONFIGURAR BÚSQUEDA DE PROVEEDORES")
        print("="*60)
        
        # 1. Seleccionar producto
        product_map = self.show_product_menu()
        
        while True:
            try:
                choice = int(input(f"\nSelecciona un producto (1-{len(product_map)}): "))
                if choice in product_map:
                    selected_product = product_map[choice]
                    break
                else:
                    print("❌ Opción no válida")
            except ValueError:
                print("❌ Por favor ingresa un número")
        
        # 2. Seleccionar urgencia
        print("\n📋 NIVEL DE URGENCIA:")
        urgency_options = {
            1: "Low",
            2: "Medium", 
            3: "High",
            4: "Critical"
        }
        
        for num, urgency in urgency_options.items():
            print(f"{num}. {urgency}")
        
        while True:
            try:
                choice = int(input("\nSelecciona urgencia (1-4): "))
                if choice in urgency_options:
                    selected_urgency = urgency_options[choice]
                    break
                else:
                    print("❌ Opción no válida")
            except ValueError:
                print("❌ Por favor ingresa un número")
        
        # 3. Cantidad
        while True:
            try:
                quantity = int(input("\n📦 Cantidad requerida: "))
                if quantity > 0:
                    break
                else:
                    print("❌ La cantidad debe ser mayor a 0")
            except ValueError:
                print("❌ Por favor ingresa un número válido")
        
        # 4. Presupuesto
        while True:
            try:
                budget = float(input("💰 Presupuesto disponible (USD): "))
                if budget > 0:
                    break
                else:
                    print("❌ El presupuesto debe ser mayor a 0")
            except ValueError:
                print("❌ Por favor ingresa un número válido")
        
        return selected_product, selected_urgency, quantity, budget
    
    def recommend_suppliers(self, product_type, urgency, quantity, budget):
        """Función de recomendación (copiada del notebook)"""
        # Validar inputs
        valid_urgencies = ['Low', 'Medium', 'High', 'Critical']
        if urgency not in valid_urgencies:
            return {"error": f"Urgencia debe ser una de: {valid_urgencies}"}
        
        # Obtener productos disponibles del tipo solicitado
        available_products = self.df[self.df['product_type'] == product_type]
        if available_products.empty:
            available_types = self.df['product_type'].unique()
            return {"error": f"Producto no encontrado. Tipos disponibles: {list(available_types)}"}
        
        # Generar recomendaciones para cada proveedor que ofrece el producto
        recommendations = []
        
        for supplier_name in available_products['supplier_name'].unique():
            supplier_data = available_products[available_products['supplier_name'] == supplier_name].iloc[0]
            
            # FEATURES LIMPIAS (sin country/quality_rating)
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
            
            # Reordenar columnas según el modelo
            input_data = input_data[self.feature_names]
            
            # Obtener probabilidades de predicción
            probabilities = self.model.predict_proba(input_data)[0]
            supplier_idx = np.where(self.target_encoder.classes_ == supplier_name)[0]
            
            if len(supplier_idx) > 0:
                probability = probabilities[supplier_idx[0]]
            else:
                probability = 0.0
            
            # Calcular costos
            total_cost = supplier_data['price_usd'] * quantity
            if not supplier_data['shipping_included']:
                total_cost += 300  # Costo estimado de envío
            
            # Crear recomendación
            recommendation = {
                'supplier_name': supplier_name,
                'country': supplier_data['country'],
                'quality_rating': supplier_data['quality_rating'],
                'price_usd': supplier_data['price_usd'],
                'total_cost': round(total_cost, 2),
                'delivery_days': supplier_data['delivery_days'],
                'payment_terms_days': supplier_data['payment_terms_days'],
                'probability_score': round(probability, 3),
                'within_budget': total_cost <= budget,
                'shipping_included': supplier_data['shipping_included'],
                'express_available': supplier_data['express_available']
            }
            
            # Calcular score final
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
                recommendation['recommendation_level'] = "⭐⭐⭐ Altamente Recomendado"
            elif final_score >= 0.6:
                recommendation['recommendation_level'] = "⭐⭐ Recomendado"
            elif final_score >= 0.45:
                recommendation['recommendation_level'] = "⭐ Aceptable"
            else:
                recommendation['recommendation_level'] = "❌ No Recomendado"
            
            recommendations.append(recommendation)
        
        # Ordenar por score final
        recommendations.sort(key=lambda x: x['final_score'], reverse=True)
        
        return {
            'product_type': product_type,
            'urgency': urgency,
            'quantity': quantity,
            'budget': budget,
            'recommendations': recommendations,
            'total_options': len(recommendations)
        }
    
    def display_results(self, result):
        """Mostrar resultados de manera clara"""
        if 'error' in result:
            print(f"\n❌ ERROR: {result['error']}")
            return
        
        print("\n" + "="*80)
        print("RESULTADOS DE LA RECOMENDACIÓN")
        print("="*80)
        
        print(f"📦 Producto: {result['product_type']}")
        print(f"⚡ Urgencia: {result['urgency']}")
        print(f"🔢 Cantidad: {result['quantity']}")
        print(f"💰 Presupuesto: ${result['budget']:,.2f}")
        print(f"🏭 Opciones encontradas: {result['total_options']}")
        
        print(f"\n🏆 RANKING DE PROVEEDORES:")
        print("-" * 80)
        
        for i, supplier in enumerate(result['recommendations'], 1):
            status_color = "🟢" if supplier['within_budget'] else "🔴"
            
            print(f"\n{i}. {supplier['supplier_name']} ({supplier['country']})")
            print(f"   {supplier['recommendation_level']}")
            print(f"   💯 Score Final: {supplier['final_score']}")
            print(f"   🤖 ML Score: {supplier['probability_score']}")
            print(f"   ⭐ Calidad: {supplier['quality_rating']}/5.0")
            print(f"   💵 Precio Unitario: ${supplier['price_usd']}")
            print(f"   💰 Costo Total: ${supplier['total_cost']:,.2f} {status_color}")
            print(f"   🚚 Delivery: {supplier['delivery_days']} días")
            print(f"   💳 Pago: {supplier['payment_terms_days']} días")
            print(f"   📦 Envío incluido: {'Sí' if supplier['shipping_included'] else 'No'}")
            print(f"   ⚡ Express: {'Disponible' if supplier['express_available'] else 'No disponible'}")
        
        # Mostrar recomendación principal
        best_supplier = result['recommendations'][0]
        print(f"\n" + "="*80)
        print("🎯 RECOMENDACIÓN PRINCIPAL")
        print("="*80)
        print(f"Proveedor: {best_supplier['supplier_name']}")
        print(f"Razón: {best_supplier['recommendation_level']}")
        print(f"Costo Total: ${best_supplier['total_cost']:,.2f}")
        print(f"Calidad: {best_supplier['quality_rating']}/5.0")
        print(f"Delivery: {best_supplier['delivery_days']} días")
    
    def run(self):
        """Ejecutar programa principal"""
        print("="*80)
        print("🏭 SISTEMA DE RECOMENDACIÓN DE PROVEEDORES")
        print("   Dewatering Solutions - Modelo Predictivo")
        print("="*80)
        
        while True:
            try:
                # Obtener parámetros del usuario
                product, urgency, quantity, budget = self.get_user_input()
                
                # Ejecutar recomendación
                print("\n🔄 Procesando recomendación...")
                result = self.recommend_suppliers(product, urgency, quantity, budget)
                
                # Mostrar resultados
                self.display_results(result)
                
                # Preguntar si continuar
                print("\n" + "="*80)
                continue_choice = input("¿Quieres hacer otra búsqueda? (s/n): ").lower()
                
                if continue_choice not in ['s', 'sí', 'si', 'y', 'yes']:
                    print("\n👋 ¡Gracias por usar el sistema de recomendación!")
                    break
                    
            except KeyboardInterrupt:
                print("\n\n👋 Programa interrumpido por el usuario. ¡Hasta luego!")
                break
            except Exception as e:
                print(f"\n❌ Error inesperado: {e}")
                print("Intenta nuevamente...")

# PROGRAMA PRINCIPAL
if __name__ == "__main__":
    # Verificar que estamos en el directorio correcto
    if not os.path.exists('models') or not os.path.exists('data'):
        print("❌ Error: Ejecuta este programa desde la carpeta principal del proyecto")
        print("   Estructura esperada:")
        print("   - models/")
        print("   - data/")
        print("   - supplier_recommender.py")
        sys.exit(1)
    
    # Crear y ejecutar el sistema
    recommender = SupplierRecommender()
    recommender.run()