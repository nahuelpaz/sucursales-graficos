import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuración de la app
st.set_page_config(layout="wide")

# Clase CalculadorDeMétricas reutilizada del segundo código
class CalculadorDeMétricas:
    @staticmethod
    def calcular_métricas_de_negocio(datos: pd.DataFrame) -> pd.DataFrame:
        analiticas = datos.copy()
        analiticas['precio_unitario'] = analiticas['ingresos'] / analiticas['unidades']
        analiticas['margen_de_ganancia'] = (analiticas['ingresos'] - analiticas['costo']) / analiticas['ingresos']
        
        return (analiticas.groupby('producto', as_index=False)
                .agg({
                    'precio_unitario': 'mean',
                    'margen_de_ganancia': 'mean',
                    'unidades': 'sum',
                    'secuencia_original': 'min'
                })
                .sort_values('secuencia_original'))

# Función para mostrar la información del alumno
def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('*Legajo:* 59074')
        st.markdown('*Nombre:* Teseyra, Juan Ignacio')
        st.markdown('*Comisión:* C9')

# Función para cargar y preparar datos
def cargar_datos(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = ['sucursal', 'producto', 'año', 'mes', 'unidades', 'ingresos', 'costo']
            df['secuencia_original'] = range(len(df))  # Guardamos el orden original
            df['periodo'] = df['año'].astype(str) + "-" + df['mes'].astype(str).str.zfill(2)
            productos_orden_original = df['producto'].unique()
            df = df.sort_values('periodo')
            return df, productos_orden_original
        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")
    return None, None

# Función para calcular variaciones y métricas
def calcular_variaciones(df, productos_orden_original):
    calculador = CalculadorDeMétricas()
    datos_métricas = calculador.calcular_métricas_de_negocio(df)
    
    resultados = []
    for producto in productos_orden_original:
        if producto in df['producto'].unique():
            # Filtrar datos por producto
            grupo = df[df['producto'] == producto].sort_values('periodo')
            if len(grupo) > 1:  # Necesitamos al menos dos periodos para calcular variaciones
                precio_anterior = grupo.iloc[-2]['ingresos'] / grupo.iloc[-2]['unidades']
                precio_actual = grupo.iloc[-1]['ingresos'] / grupo.iloc[-1]['unidades']
                margen_anterior = (grupo.iloc[-2]['ingresos'] - grupo.iloc[-2]['costo']) / grupo.iloc[-2]['ingresos']
                margen_actual = (grupo.iloc[-1]['ingresos'] - grupo.iloc[-1]['costo']) / grupo.iloc[-1]['ingresos']
                unidades_anteriores = grupo.iloc[-2]['unidades']
                unidades_actuales = grupo.iloc[-1]['unidades']

                delta_precio = ((precio_actual - precio_anterior) / precio_anterior) * 100
                delta_margen = ((margen_actual - margen_anterior) / margen_anterior) * 100
                delta_unidades = ((unidades_actuales - unidades_anteriores) / unidades_anteriores) * 100
            else:
                delta_precio = delta_margen = delta_unidades = None

            producto_data = datos_métricas[datos_métricas['producto'] == producto].iloc[0]
            resultados.append({
                'Producto': producto,
                'Precio_promedio': producto_data['precio_unitario'],
                'Margen_promedio': producto_data['margen_de_ganancia'],
                'Unidades_vendidas': producto_data['unidades'],
                'Delta_precio': delta_precio,
                'Delta_margen': delta_margen,
                'Delta_unidades': delta_unidades
            })
    
    return pd.DataFrame(resultados)

# Función para crear gráfico de ventas
def crear_grafico_ventas(df, producto):
    ventas = df.groupby(['año', 'mes']).agg({'ingresos': 'sum', 'unidades': 'sum'}).reset_index()
    ventas['fecha'] = pd.to_datetime(ventas['año'].astype(str) + '-' + ventas['mes'].astype(str).str.zfill(2) + '-01')
    ventas.sort_values('fecha', inplace=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.grid(True, linestyle='-', alpha=0.3, color='gray')

    ax.plot(ventas['fecha'], ventas['unidades'], color='#2563EB', linewidth=1.5, label=producto)

    z = np.polyfit(np.arange(len(ventas)), ventas['unidades'], 1)
    p = np.poly1d(z)
    ax.plot(ventas['fecha'], p(np.arange(len(ventas))), color='red', linestyle='--', linewidth=1.5, label='Tendencia')

    ax.set_title(f'Evolución de Ventas: {producto}', fontsize=12)
    ax.set_xlabel('Fecha', fontsize=10)
    ax.set_ylabel('Unidades Vendidas', fontsize=10)
    ax.legend(loc='best', fontsize=10)
    ax.tick_params(axis='x', rotation=45)

    return fig

# Aplicación principal
uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

if uploaded_file:
    df, productos_orden_original = cargar_datos(uploaded_file)
    if df is not None and productos_orden_original is not None:
        sucursales = ['Todas'] + list(df['sucursal'].unique())
        sucursal_seleccionada = st.sidebar.selectbox("Filtrar por Sucursal", sucursales)

        if sucursal_seleccionada == 'Todas':
            st.title("Análisis de ventas - Todas las sucursales")
            df_filtrado = df
        else:
            st.title(f"Análisis de ventas - Sucursal {sucursal_seleccionada}")
            df_filtrado = df[df['sucursal'] == sucursal_seleccionada]

        metricas = calcular_variaciones(df_filtrado, productos_orden_original)

        for producto in productos_orden_original:
            if producto in df_filtrado['producto'].unique():
                producto_data = metricas[metricas['Producto'] == producto].iloc[0]
                with st.container(border=True):
                    st.subheader(producto)

                    col1, col2 = st.columns([2, 3])
                    with col1:
                            st.metric("Precio Promedio", f"${producto_data['Precio_promedio']:.2f}", 
                                 f"{producto_data['Delta_precio']:.2f}%" if producto_data['Delta_precio'] is not None else "N/A")
                            st.metric("Margen Promedio", f"{producto_data['Margen_promedio'] * 100:.2f}%", 
                                 f"{producto_data['Delta_margen']:.2f}%" if producto_data['Delta_margen'] is not None else "N/A")
                            st.metric("Unidades Vendidas", f"{producto_data['Unidades_vendidas']:,}", 
                                 f"{producto_data['Delta_unidades']:.2f}%" if producto_data['Delta_unidades'] is not None else "N/A")
                    with col2:
                        datos_producto = df_filtrado[df_filtrado['producto'] == producto]
                        grafico = crear_grafico_ventas(datos_producto, producto)
                        st.pyplot(grafico)
    else:
        st.error("No se pudo procesar el archivo.")
else:
    st.title("**Por favor sube un archivo CSV para comenzar**")
    mostrar_informacion_alumno()
