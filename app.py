import streamlit as st
import pandas as pd
import numpy as np
from pulp import *
from scipy.spatial.distance import cdist
import plotly.graph_objects as go
from supabase import create_client, Client
import google.generativeai as genai

#Initialization of Supabase and Gemini clients
url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(url, key)

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

def initialize_secure_model():
    """
    Busca dinámicamente un modelo disponible para evitar el error 404.
    """
    try:
        #List models that support content generation
        modelos_disponibles = [
            m.name for m in genai.list_models() 
            if 'generateContent' in m.supported_generation_methods
        ]
        
        if not modelos_disponibles:
            return None
            
        #Try ti prioritize 'flash' models, if available
        for name in modelos_disponibles:
            if 'flash' in name.lower():
                return genai.GenerativeModel(name)
        
        return genai.GenerativeModel(modelos_disponibles[0])
    except Exception as e:
        st.sidebar.error(f"Error de conexión con Google: {e}")
        return None

#Initialize Gemini model
model_gemini = initialize_secure_model()

#Page COnfiguration
st.set_page_config(page_title="Marketplace Optimizer", layout="wide")
st.title("Marketplace Optimización & IA")

#Optimization Model
def run_optimization(n_orders, n_couriers):
    #Sintetic data generation
    orders = pd.DataFrame({'id': range(n_orders), 'x': np.random.rand(n_orders)*10, 'y': np.random.rand(n_orders)*10})
    couriers = pd.DataFrame({'id': range(n_couriers), 'x': np.random.rand(n_couriers)*10, 'y': np.random.rand(n_couriers)*10, 'capacity': [5]*n_couriers    })

    #Matrix of distances
    dist_matrix = cdist(orders[['x', 'y']], couriers[['x', 'y']], metric='euclidean')

    #dynamic_capacity = math.ceil(n_orders / n_couriers) + 1
    #couriers['capacity'] = [dynamic_capacity] * n_couriers
    
    #Definition of linear programming model
    prob = LpProblem("Delivery_Optimization", LpMinimize)
    x = LpVariable.dicts("assign", (orders.index, couriers.index), 0, 1, LpBinary)

    #Objective function
    prob += lpSum([dist_matrix[i][j] * x[i][j] for i in orders.index for j in couriers.index])

    #Restrictions
    for i in orders.index: prob += lpSum([x[i][j] for j in couriers.index]) == 1 #Each order assigned to one courier
    for j in couriers.index: prob += lpSum([x[i][j] for i in orders.index]) <= couriers.loc[j, 'capacity'] #Courier capacity

    prob.solve(PULP_CBC_CMD(msg=0))

    assignments = []
    if LpStatus[prob.status] == 'Optimal':
        for i in orders.index:
            for j in couriers.index:
                if value(x[i][j]) == 1:
                    assignments.append({
                        'order_id': f"Order_{i}", 
                        'courier_id': f"Courier_{j}",
                        })
        
    df_results = pd.DataFrame(assignments)

    return value(prob.objective), LpStatus[prob.status], df_results

#Gemini integration
def get_gemini_insights(total_dist, orders, couriers):
    """
    Traduce resultados técnicos a insights de negocio para aplicaciones de delivery.
    """
    prompt = (f"Actúa como un Operations Manager en ena empresa de delivery. "
              f"Acabamos de correr una optimización: Distancia total: {total_dist:.2f}, "
              f"Pedidos: {orders}, Repartidores: {couriers}. "
              f"Dame 3 recomendaciones estratégicas para mejorar la eficiencia operativa.")
    
    try:
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Análisis no disponible temporalmente. (Error: {str(e)})"

#Generate assignment graph
def generate_assignment_graph(df_orders, df_couriers, df_result):
    """
    Crea un mapa visual de las asignaciones optimizadas.
    Demuestra capacidad de 'Visualización de Datos' y 'Análisis de Redes'.
    """
    fig = go.Figure()

    #Draw assignation lines
    for _, row in df_result.iterrows():
        #Get order and courier coordinates
        order = df_orders[df_orders["order_id"] == row["order_id"]].iloc[0]
        courier = df_couriers[df_couriers["courier_id"] == row["courier_id"]].iloc[0]

        fig.add_trace(go.Scatter(
            x=[order['x'], courier['x']],
            y=[order['y'], courier['y']],
            mode='lines',
            line=dict(color='gray', width=1, dash='dot'),
            showlegend=False
        ))

    #Draw orders (Red points)
    fig.add_trace(go.Scatter(
        x=df_orders['x'], y=df_orders['y'],
        mode='markers',
        marker=dict(size=10, color='red', symbol='circle'),
        name='Pedidos',
        text=df_orders['order_id']
    ))

    #Draw couriers (Blue squares)
    fig.add_trace(go.Scatter(
        x=df_couriers['x'], y=df_couriers['y'],
        mode='markers',
        marker=dict(size=15, color='blue', symbol='square'),
        name='Repartidores (' \
        ')',
        text=df_couriers['courier_id']
    ))

    fig.update_layout(
        title="Visualización de Asignación de Última Milla",
        xaxis_title="Coordenada X (km)",
        yaxis_title="Coordenada Y (km)",
        template="plotly_white"
    )
    
    return fig

#UI and data base
st.sidebar.header("Parámetros de Simulación")
orders_input = st.sidebar.number_input("Cantidad de Pedidos", 5, 50, 15)
couriers_input = st.sidebar.number_input("Cantidad de Repartidores", 2, 10, 3)

if st.button("Optimizar y Analizar con Gemini"):
    #Generate data
    np.random.seed(42)
    df_orders = pd.DataFrame({
        'order_id': [f'Order_{i}' for i in range(orders_input)],
        'x': np.random.uniform(0, 10, orders_input),
        'y': np.random.uniform(0, 10, orders_input)
    })
    df_couriers = pd.DataFrame({
        'courier_id': [f'Courier_{i}' for i in range(couriers_input)],
        'x': np.random.uniform(0, 10, couriers_input),
        'y': np.random.uniform(0, 10, couriers_input),
        'capacity': [5] * couriers_input
    })

    #Execute mathematical optimization
    total_distance, status, df_assignments = run_optimization(orders_input, couriers_input)
    
    if status == 'Optimal':
        st.subheader("Mapa de asignacion optimizada")

        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        avg_distance = total_distance / orders_input

        col_kpi1.metric("Distancia total", f"{total_distance:.2f} km")
        col_kpi2.metric("Distancia promedio por pedido", f"{avg_distance:.2f} km")
        col_kpi3.metric("Eficiencia de asignación", f"{(orders_input / (couriers_input * 5)) * 100:.1f}%")

        col_map, col_table = st.columns([2, 1])
        with col_map:
            fig = generate_assignment_graph(df_orders, df_couriers, df_assignments)
            st.plotly_chart(fig, use_container_width=True)

        with col_table:
            st.dataframe(df_assignments, use_container_width=True, hide_index=True)  # Show assignment details

        #Generate insights with Gemini
        with st.spinner("Gemini está analizando la operación..."):
            ai_insight = get_gemini_insights(total_distance, orders_input, couriers_input)

            #Supabase persistence
            try:
                supabase.table("optimization_logs").insert({
                    "total_distance": total_distance,
                    "num_orders": orders_input,
                    "ai_insight": ai_insight
                }).execute()
            except Exception as e:
                st.error(f"Error al guardar en DB: {e}")

            #Show results
            st.markdown("### Recomendiaciones de IA (Gemini):")
            st.write(ai_insight)            
    else:
        st.error("No se encontró una solución óptima. Intenta ajustar los parámetros o revisa la capacidad de los repartidores.")
        
#Visualization of past optimizations
st.divider()
st.subheader("📊 Historial de Decisiones (Supabase Cloud)")

logs = supabase.table("optimization_logs").select("created_at, total_distance, num_orders").order("created_at", desc=True).limit(5).execute()

if logs.data:
    st.dataframe(pd.DataFrame(logs.data))