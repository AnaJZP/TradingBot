import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Añadir el directorio raíz al path para importar los otros módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators.moving_averages import MovingAverages
from trading.signals import CrossoverSignals
from trading.backtest import Backtest

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Medias Móviles",
    page_icon="📈",
    layout="wide"
)

# Título de la aplicación
st.title("📊 Análisis de Medias Móviles para Trading")
st.markdown("### Una herramienta educativa para análisis bursátil")

# Barra lateral para configuración
with st.sidebar:
    st.header("Configuración")
    
    # Selección de símbolo
    ticker = st.text_input("Símbolo", value="AAPL")
    
    # Selección de fecha inicial
    start_date = st.date_input(
        "Fecha inicial",
        datetime.now() - timedelta(days=365)
    )
    
    # Parámetros de medias móviles
    st.subheader("Medias Móviles")
    
    # SMA
    sma_periods = st.multiselect(
        "Períodos SMA",
        options=[5, 10, 20, 50, 100, 200],
        default=[20, 50]
    )
    
    # EMA
    ema_periods = st.multiselect(
        "Períodos EMA",
        options=[5, 10, 12, 20, 26, 50, 100, 200],  # Añadido 12 y 26 a las opciones
        default=[12, 26]
    )
    
    # Optimización
    st.subheader("Optimización")
    optimize = st.checkbox("Optimizar medias móviles", value=False)
    
    if optimize:
        optimization_metric = st.selectbox(
            "Métrica de optimización",
            options=["profit", "sharpe", "drawdown"],
            index=1
        )
        
        col1, col2 = st.columns(2)
        with col1:
            min_sma = st.number_input("SMA mínimo", value=5, min_value=2)
            max_sma = st.number_input("SMA máximo", value=100, min_value=min_sma)
        
        with col2:
            min_ema = st.number_input("EMA mínimo", value=5, min_value=2)
            max_ema = st.number_input("EMA máximo", value=100, min_value=min_ema)
        
        sma_range = list(range(min_sma, max_sma + 1, 5))  # Incrementos de 5 para reducir combinaciones
        ema_range = list(range(min_ema, max_ema + 1, 5))
    
    # Parámetros de backtest
    st.subheader("Backtest")
    initial_capital = st.number_input("Capital inicial ($)", value=10000, min_value=1000)
    position_size = st.slider("Tamaño de posición (%)", min_value=10, max_value=100, value=100) / 100
    commission_pct = st.slider("Comisión (%)", min_value=0.0, max_value=2.0, value=0.1) / 100
    
    # Botón para ejecutar
    run_analysis = st.button("Ejecutar Análisis", type="primary")

# Función principal para cargar datos y calcular indicadores
@st.cache_data(ttl=3600)  # Cache por 1 hora
def load_data(ticker, start_date):
    try:
        # Método recomendado con la nueva versión de yfinance
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(start=start_date)
        
        # Eliminar zona horaria del índice
        if data.index.tzinfo is not None:
            data.index = data.index.tz_localize(None)
        
        # Verificar y eliminar valores nulos
        na_count = data.isna().sum().sum()
        if na_count > 0:
            st.warning(f"Se encontraron {na_count} valores NA en los datos. Se eliminarán automáticamente.")
            data = data.dropna()
        
        if data.empty:
            st.error(f"No se encontraron datos para el símbolo {ticker}")
            return None
            
        # Verificar que tenemos suficientes datos
        if len(data) < 30:
            st.warning("ADVERTENCIA: Pocos datos disponibles. Los resultados pueden no ser confiables.")
            
        return data
    except Exception as e:
        st.error(f"Error al cargar datos: {str(e)}")
        return None

# Ejecutar análisis cuando se pulsa el botón
if run_analysis:
    # Mostrar spinner durante la carga
    with st.spinner("Cargando datos..."):
        # Cargar datos
        data = load_data(ticker, start_date)
        
        if data is not None:
            # Tab layout para diferentes secciones
            tab1, tab2, tab3 = st.tabs(["📈 Análisis Técnico", "🔍 Optimización", "💰 Backtest"])
            
            # Inicializar objetos
            ma = MovingAverages(data)
            
            with tab1:
                st.header(f"Análisis Técnico para {ticker}")
                
                # Añadir medias móviles al dataframe
                df_indicators = ma.add_indicators(sma_periods, ema_periods)
                
                # Crear gráfico con Plotly
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                   vertical_spacing=0.03, subplot_titles=('Precio y Medias Móviles', 'Volumen'),
                                   row_heights=[0.7, 0.3])
                
                # Añadir precio
                fig.add_trace(
                    go.Candlestick(
                        x=df_indicators.index,
                        open=df_indicators['Open'] if 'Open' in df_indicators.columns else df_indicators['open'],
                        high=df_indicators['High'] if 'High' in df_indicators.columns else df_indicators['high'],
                        low=df_indicators['Low'] if 'Low' in df_indicators.columns else df_indicators['low'],
                        close=df_indicators['Close'] if 'Close' in df_indicators.columns else df_indicators['close'],
                        name="OHLC"
                    ),
                    row=1, col=1
                )
                
                # Añadir SMAs
                for period in sma_periods:
                    fig.add_trace(
                        go.Scatter(
                            x=df_indicators.index,
                            y=df_indicators[f'SMA_{period}'],
                            name=f'SMA {period}',
                            line=dict(width=2)
                        ),
                        row=1, col=1
                    )
                
                # Añadir EMAs
                for period in ema_periods:
                    fig.add_trace(
                        go.Scatter(
                            x=df_indicators.index,
                            y=df_indicators[f'EMA_{period}'],
                            name=f'EMA {period}',
                            line=dict(width=2, dash='dash')
                        ),
                        row=1, col=1
                    )
                
                # Añadir volumen
                volume_col = 'Volume' if 'Volume' in df_indicators.columns else 'volume'
                if volume_col in df_indicators.columns:
                    fig.add_trace(
                        go.Bar(
                            x=df_indicators.index,
                            y=df_indicators[volume_col],
                            name='Volumen',
                            marker_color='rgba(0, 150, 255, 0.6)'
                        ),
                        row=2, col=1
                    )
                
                # Ajustar layout
                fig.update_layout(
                    height=600,
                    title_text=f"{ticker} - Análisis Técnico",
                    xaxis_rangeslider_visible=False,
                    template="plotly_white"
                )
                
                # Mostrar gráfico
                st.plotly_chart(fig, use_container_width=True)
                
                # Generar señales con los cruces
                if len(sma_periods) > 0 and len(ema_periods) > 0:
                    st.subheader("Señales de Trading (SMA-EMA)")
                    signals = CrossoverSignals(df_indicators)
                    
                    # Tomar el primer SMA y EMA para generar señales
                    df_signals = signals.sma_ema_crossover(sma_periods[0], ema_periods[0])
                    
                    # Filtrar solo las filas con señales
                    signal_rows = df_signals[df_signals['signal'] != 0].copy()
                    signal_rows['signal_type'] = signal_rows['signal'].apply(lambda x: "COMPRA" if x > 0 else "VENTA")
                    
                    if not signal_rows.empty:
                        # Mostrar tabla de señales
                        signal_display = signal_rows[['Close', 'signal_type']].rename(
                            columns={'Close': 'Precio', 'signal_type': 'Señal'}
                        )
                        st.dataframe(signal_display, use_container_width=True)
                    else:
                        st.info("No se generaron señales con los parámetros actuales")
                
                # Mostrar tabla de datos recientes
                with st.expander("Ver datos recientes"):
                    st.dataframe(df_indicators.tail(10), use_container_width=True)
            
            with tab2:
                st.header("Optimización de Medias Móviles")
                
                if optimize:
                    with st.spinner("Optimizando parámetros..."):
                        try:
                            # Ejecutar optimización
                            optimization_results = ma.find_optimal_combination(
                                sma_range=sma_range,
                                ema_range=ema_range,
                                metric=optimization_metric
                            )
                            
                            # Mostrar resultados
                            best_sma = optimization_results['combination']['sma']
                            best_ema = optimization_results['combination']['ema']
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Mejor SMA", best_sma)
                            with col2:
                                st.metric("Mejor EMA", best_ema)
                            with col3:
                                metric_value = optimization_results['metric_value']
                                if optimization_metric == 'profit':
                                    formatted_metric = f"${metric_value:.2f}"
                                elif optimization_metric == 'sharpe':
                                    formatted_metric = f"{metric_value:.2f}"
                                else:  # drawdown
                                    formatted_metric = f"{metric_value*100:.2f}%"
                                
                                st.metric(
                                    f"Mejor {optimization_metric.capitalize()}", 
                                    formatted_metric
                                )
                            
                            # Añadir medias móviles optimizadas al dataframe
                            df_optimized = ma.add_indicators([best_sma], [best_ema])
                            
                            # Generar señales con los parámetros óptimos
                            signals = CrossoverSignals(df_optimized)
                            df_signals = signals.sma_ema_crossover(best_sma, best_ema)
                            
                            # Gráfico con parámetros optimizados
                            fig = make_subplots(rows=1, cols=1)
                            
                            # Añadir precio
                            fig.add_trace(
                                go.Candlestick(
                                    x=df_optimized.index,
                                    open=df_optimized['Open'],
                                    high=df_optimized['High'],
                                    low=df_optimized['Low'],
                                    close=df_optimized['Close'],
                                    name="Precio"
                                )
                            )
                            
                            # Añadir medias móviles optimizadas
                            fig.add_trace(
                                go.Scatter(
                                    x=df_optimized.index,
                                    y=df_optimized[f'SMA_{best_sma}'],
                                    name=f'SMA {best_sma}',
                                    line=dict(width=2, color='blue')
                                )
                            )
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=df_optimized.index,
                                    y=df_optimized[f'EMA_{best_ema}'],
                                    name=f'EMA {best_ema}',
                                    line=dict(width=2, color='red', dash='dash')
                                )
                            )
                            
                            # Añadir señales
                            price_col = 'Close' if 'Close' in df_signals.columns else 'close'
                            buy_signals = df_signals[df_signals['signal'] > 0]
                            sell_signals = df_signals[df_signals['signal'] < 0]
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=buy_signals.index,
                                    y=buy_signals[price_col],
                                    mode='markers',
                                    name='Compra',
                                    marker=dict(
                                        symbol='triangle-up',
                                        size=15,
                                        color='green',
                                    )
                                )
                            )
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=sell_signals.index,
                                    y=sell_signals[price_col],
                                    mode='markers',
                                    name='Venta',
                                    marker=dict(
                                        symbol='triangle-down',
                                        size=15,
                                        color='red',
                                    )
                                )
                            )
                            
                            # Actualizar layout
                            fig.update_layout(
                                title=f'Estrategia Optimizada: SMA({best_sma})-EMA({best_ema})',
                                xaxis_title='Fecha',
                                yaxis_title='Precio',
                                height=500,
                                xaxis_rangeslider_visible=False,
                                template="plotly_white"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Mostrar algunas métricas adicionales
                            results = optimization_results['results']
                            
                            st.subheader("Métricas de Rendimiento")
                            metrics_cols = st.columns(4)
                            
                            with metrics_cols[0]:
                                st.metric("Retorno Total", f"{results['total_return']*100:.2f}%")
                            with metrics_cols[1]:
                                st.metric("Retorno Anualizado", f"{results['annualized_return']*100:.2f}%")
                            with metrics_cols[2]:
                                st.metric("Ratio de Sharpe", f"{results['sharpe_ratio']:.2f}")
                            with metrics_cols[3]:
                                st.metric("Max Drawdown", f"{results['max_drawdown']*100:.2f}%")
                                
                        except Exception as e:
                            st.error(f"Error durante la optimización: {str(e)}")
                else:
                    st.info("Activa la opción 'Optimizar medias móviles' en la barra lateral para encontrar la mejor combinación de parámetros.")
            
            with tab3:
                st.header("Backtest de la Estrategia")
                
                # Selección de estrategia para backtest
                st.subheader("Configura tu estrategia")
                
                strategy_type = st.selectbox(
                    "Tipo de estrategia",
                    options=["SMA-EMA Crossover", "Double SMA Crossover", "Double EMA Crossover"]
                )
                
                # Usar parámetros optimizados si están disponibles
                if optimize and 'optimization_results' in locals():
                    use_optimized = st.checkbox("Usar parámetros optimizados", value=True)
                    
                    if use_optimized:
                        if strategy_type == "SMA-EMA Crossover":
                            param1 = best_sma
                            param2 = best_ema
                        else:
                            # Para las otras estrategias, usar valores predeterminados
                            param1 = 20
                            param2 = 50
                    else:
                        # Configuración manual
                        col1, col2 = st.columns(2)
                        with col1:
                            param1 = st.selectbox("Primer parámetro", options=[5, 10, 20, 50, 100, 200], index=1)
                        with col2:
                            param2 = st.selectbox("Segundo parámetro", options=[5, 10, 20, 50, 100, 200], index=3)
                else:
                    # Configuración manual si no hay optimización
                    col1, col2 = st.columns(2)
                    with col1:
                        param1 = st.selectbox("Primer parámetro", options=[5, 10, 20, 50, 100, 200], index=1)
                    with col2:
                        param2 = st.selectbox("Segundo parámetro", options=[5, 10, 20, 50, 100, 200], index=3)
                
                # Ejecutar backtest
                with st.spinner("Ejecutando backtest..."):
                    try:
                        # Preparar datos según estrategia seleccionada
                        if strategy_type == "SMA-EMA Crossover":
                            df_indicators = ma.add_indicators([param1], [param2])
                            signals = CrossoverSignals(df_indicators)
                            df_signals = signals.sma_ema_crossover(param1, param2)
                            strategy_name = f"SMA({param1})-EMA({param2}) Crossover"
                        
                        elif strategy_type == "Double SMA Crossover":
                            df_indicators = ma.add_indicators([param1, param2], [])
                            signals = CrossoverSignals(df_indicators)
                            df_signals = signals.double_sma_crossover(param1, param2)
                            strategy_name = f"SMA({param1})-SMA({param2}) Crossover"
                        
                        else:  # Double EMA Crossover
                            df_indicators = ma.add_indicators([], [param1, param2])
                            signals = CrossoverSignals(df_indicators)
                            df_signals = signals.double_ema_crossover(param1, param2)
                            strategy_name = f"EMA({param1})-EMA({param2}) Crossover"
                        
                        # Ejecutar backtest
                        backtest = Backtest(df_signals)
                        results = backtest.run(
                            initial_capital=initial_capital,
                            position_size=position_size,
                            commission=commission_pct
                        )
                        
                        # Mostrar resultados
                        st.subheader(f"Resultados del Backtest: {strategy_name}")
                        
                        # Métricas principales
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Capital Inicial", f"${initial_capital:,.2f}")
                            st.metric("Capital Final", f"${results['final_capital']:,.2f}")
                            st.metric("Retorno Total", f"{results['total_return']*100:.2f}%")
                            st.metric("Operaciones Totales", results['total_trades'])
                        
                        with col2:
                            st.metric("Retorno Anualizado", f"{results['annualized_return']*100:.2f}%")
                            st.metric("Ratio de Sharpe", f"{results['sharpe_ratio']:.2f}")
                            st.metric("Máximo Drawdown", f"{results['max_drawdown']*100:.2f}%")
                            st.metric("Comisiones Pagadas", f"${results['total_commission']:,.2f}")
                        
                        # Gráfico de curva de equidad
                        equity_data = results['backtest_data']
                        
                        fig = go.Figure()
                        
                        # Curva de equidad
                        fig.add_trace(
                            go.Scatter(
                                x=equity_data.index,
                                y=equity_data['capital'],
                                mode='lines',
                                name='Capital',
                                line=dict(color='blue', width=2)
                            )
                        )
                        
                        # Señales de compra
                        buy_signals = equity_data[equity_data['signal'] > 0]
                        if not buy_signals.empty:
                            fig.add_trace(
                                go.Scatter(
                                    x=buy_signals.index,
                                    y=buy_signals['capital'],
                                    mode='markers',
                                    name='Compra',
                                    marker=dict(
                                        symbol='triangle-up',
                                        size=12,
                                        color='green',
                                    )
                                )
                            )
                        
                        # Señales de venta
                        sell_signals = equity_data[equity_data['signal'] < 0]
                        if not sell_signals.empty:
                            fig.add_trace(
                                go.Scatter(
                                    x=sell_signals.index,
                                    y=sell_signals['capital'],
                                    mode='markers',
                                    name='Venta',
                                    marker=dict(
                                        symbol='triangle-down',
                                        size=12,
                                        color='red',
                                    )
                                )
                            )
                        
                        # Actualizar layout
                        fig.update_layout(
                            title='Curva de Equidad',
                            xaxis_title='Fecha',
                            yaxis_title='Capital ($)',
                            height=500,
                            template="plotly_white"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Tabla de operaciones
                        st.subheader("Registro de Operaciones")
                        trade_history = equity_data[equity_data['trade_value'] != 0].copy()
                        
                        if not trade_history.empty:
                            trade_history['tipo'] = trade_history['trade_value'].apply(
                                lambda x: "COMPRA" if x > 0 else "VENTA"
                            )
                            
                            trade_display = trade_history[[
                                'tipo', 'trade_price', 'trade_value', 'commission_paid', 'capital'
                            ]].rename(
                                columns={
                                    'tipo': 'Tipo',
                                    'trade_price': 'Precio',
                                    'trade_value': 'Valor',
                                    'commission_paid': 'Comisión',
                                    'capital': 'Capital'
                                }
                            )
                            
                            st.dataframe(trade_display, use_container_width=True)
                        else:
                            st.info("No se realizaron operaciones durante el período analizado.")
                        
                    except Exception as e:
                        st.error(f"Error durante el backtest: {str(e)}")
        else:
            st.warning("No se pudieron cargar los datos para realizar el análisis.")
else:
    # Mostrar instrucciones cuando se inicia la aplicación
    st.info("👈 Configura los parámetros en la barra lateral y presiona 'Ejecutar Análisis' para comenzar.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("¿Qué son las Medias Móviles?")
        st.write("""
        Las medias móviles son indicadores técnicos que suavizan los datos de precios para crear una línea de tendencia.
        Son útiles para identificar la dirección de la tendencia y potenciales puntos de entrada y salida.
        
        Tipos principales:
        - **SMA (Simple Moving Average)**: Promedio simple de precios durante un período específico.
        - **EMA (Exponential Moving Average)**: Da más peso a los precios recientes, reaccionando más rápido a los cambios.
        """)
    
    with col2:
        st.subheader("Estrategias de Cruce")
        st.write("""
        Las estrategias de cruce de medias móviles son técnicas populares para generar señales de trading:
        
        - **SMA-EMA Crossover**: Compra cuando EMA cruza por encima de SMA, vende cuando cruza por debajo.
        - **Double SMA Crossover**: Usa dos SMAs de diferentes períodos para generar señales.
        - **Double EMA Crossover**: Similar, pero usando dos EMAs para mayor sensibilidad.
        """)