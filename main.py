#!/usr/bin/env python3
"""
Módulo principal para ejecutar el análisis de medias móviles.
Este script permite ejecutar diferentes funcionalidades del sistema de análisis bursátil
desde la línea de comandos.
"""

import argparse
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Añadir el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar módulos del proyecto
from indicators.moving_averages import MovingAverages
from trading.signals import CrossoverSignals
from trading.backtest import Backtest


def download_data(ticker, start_date, output_file=None):
    """
    Descarga datos históricos para un símbolo dado hasta la fecha actual.
    
    Args:
        ticker: Símbolo a descargar
        start_date: Fecha de inicio (formato YYYY-MM-DD)
        output_file: Archivo de salida para guardar los datos (opcional)
    
    Returns:
        DataFrame con los datos descargados
    """
    print(f"Descargando datos para {ticker} desde {start_date} hasta hoy...")
    
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
            print(f"Se encontraron {na_count} valores NA en los datos. Eliminando...")
            data = data.dropna()
        
        if data.empty:
            print(f"No se encontraron datos para {ticker}")
            return None
        
        # Verificar que tenemos suficientes datos
        if len(data) < 30:
            print("ADVERTENCIA: Pocos datos disponibles. Los resultados pueden no ser confiables.")
        
        print(f"Datos procesados con éxito. {len(data)} observaciones.")
        print(f"Rango: {data.index.min().strftime('%Y-%m-%d')} a {data.index.max().strftime('%Y-%m-%d')}")
        
        if output_file:
            # Crear directorio si no existe
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            data.to_csv(output_file)
            print(f"Datos guardados en {output_file}")
        
        return data
        
    except Exception as e:
        print(f"Error al obtener datos: {str(e)}")
        return None


def run_analysis(data, sma_periods, ema_periods, output_file=None):
    """
    Ejecuta el análisis de medias móviles.
    
    Args:
        data: DataFrame con datos de mercado
        sma_periods: Lista de períodos para SMA
        ema_periods: Lista de períodos para EMA
        output_file: Ruta donde guardar los resultados (opcional)
    
    Returns:
        DataFrame con indicadores añadidos
    """
    print("Ejecutando análisis de medias móviles...")
    
    # Verificar y mostrar las columnas del DataFrame para depuración
    print(f"Columnas en el DataFrame: {data.columns.tolist()}")
    
    try:
        ma = MovingAverages(data)
        df_indicators = ma.add_indicators(sma_periods, ema_periods)
        
        # Guardar resultados si se proporcionó una ruta de salida
        if output_file:
            # Crear el directorio si no existe
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Directorio creado: {output_dir}")
            
            df_indicators.to_csv(output_file)
            print(f"Resultados guardados en {output_file}")
            
        return df_indicators
    except Exception as e:
        print(f"Error durante el análisis: {str(e)}")
        raise


def run_optimization(data, sma_range, ema_range, metric='sharpe', output_file=None):
    """
    Ejecuta la optimización de medias móviles.
    
    Args:
        data: DataFrame con datos de mercado
        sma_range: Lista de períodos de SMA a probar
        ema_range: Lista de períodos de EMA a probar
        metric: Métrica a optimizar ('profit', 'sharpe', 'drawdown')
        output_file: Ruta donde guardar los resultados (opcional)
    
    Returns:
        Diccionario con los resultados de la optimización
    """
    print(f"Optimizando medias móviles ({metric})...")
    ma = MovingAverages(data)
    results = ma.find_optimal_combination(sma_range, ema_range, metric=metric)
    
    print(f"Mejor combinación: SMA({results['combination']['sma']}), EMA({results['combination']['ema']})")
    print(f"Valor de {metric}: {results['metric_value']}")
    
    # Guardar resultados si se especifica un archivo
    if output_file:
        # Crear directorio si no existe
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Directorio creado: {output_dir}")
            
        # Guardar resultados en CSV
        results_df = pd.DataFrame({
            'Métrica': ['SMA', 'EMA', metric],
            'Valor': [
                results['combination']['sma'],
                results['combination']['ema'],
                results['metric_value']
            ]
        })
        
        results_df.to_csv(output_file, index=False)
        print(f"Resultados guardados en {output_file}")
    
    return results


def run_backtest(data, strategy_type, param1, param2, initial_capital=10000,
               position_size=1.0, commission=0.001, plot=False, output_file=None):
    """
    Ejecuta el backtest de una estrategia.
    
    Args:
        data: DataFrame con datos de mercado
        strategy_type: Tipo de estrategia ('sma_ema', 'double_sma', 'double_ema')
        param1: Primer parámetro (depende de la estrategia)
        param2: Segundo parámetro (depende de la estrategia)
        initial_capital: Capital inicial
        position_size: Tamaño de la posición (0-1)
        commission: Comisión por operación
        plot: Si es True, muestra el gráfico
        output_file: Archivo para guardar resultados del backtest
    
    Returns:
        Diccionario con los resultados del backtest
    """
    print(f"Ejecutando backtest de estrategia {strategy_type}...")
    
    ma = MovingAverages(data)
    
    if strategy_type == 'sma_ema':
        df_indicators = ma.add_indicators([param1], [param2])
        signals = CrossoverSignals(df_indicators)
        df_signals = signals.sma_ema_crossover(param1, param2)
        strategy_name = f"SMA({param1})-EMA({param2}) Crossover"
    
    elif strategy_type == 'double_sma':
        df_indicators = ma.add_indicators([param1, param2], [])
        signals = CrossoverSignals(df_indicators)
        df_signals = signals.double_sma_crossover(param1, param2)
        strategy_name = f"SMA({param1})-SMA({param2}) Crossover"
    
    elif strategy_type == 'double_ema':
        df_indicators = ma.add_indicators([], [param1, param2])
        signals = CrossoverSignals(df_indicators)
        df_signals = signals.double_ema_crossover(param1, param2)
        strategy_name = f"EMA({param1})-EMA({param2}) Crossover"
    
    else:
        raise ValueError(f"Estrategia {strategy_type} no reconocida")
    
    backtest = Backtest(df_signals)
    results = backtest.run(
        initial_capital=initial_capital,
        position_size=position_size,
        commission=commission
    )
    
    # Imprimir resultados
    print(f"Resultados para {strategy_name}:")
    print(f"Capital inicial: ${initial_capital:.2f}")
    print(f"Capital final: ${results['final_capital']:.2f}")
    print(f"Retorno total: {results['total_return']*100:.2f}%")
    print(f"Ratio de Sharpe: {results['sharpe_ratio']:.2f}")
    print(f"Máximo drawdown: {results['max_drawdown']*100:.2f}%")
    print(f"Operaciones totales: {results['total_trades']}")
    
    # Guardar resultados si se especifica un archivo
    if output_file:
        # Crear directorio si no existe
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Directorio creado: {output_dir}")
            
        # Guardar resultados en CSV
        results_df = pd.DataFrame({
            'Métrica': [
                'Capital Inicial', 'Capital Final', 'Retorno Total', 'Retorno Anualizado',
                'Volatilidad Anualizada', 'Ratio de Sharpe', 'Máximo Drawdown',
                'Días de Trading', 'Operaciones Totales', 'Comisiones Totales'
            ],
            'Valor': [
                f"${results['initial_capital']:.2f}",
                f"${results['final_capital']:.2f}",
                f"{results['total_return']*100:.2f}%",
                f"{results['annualized_return']*100:.2f}%",
                f"{results['annualized_volatility']*100:.2f}%",
                f"{results['sharpe_ratio']:.2f}",
                f"{results['max_drawdown']*100:.2f}%",
                results['trading_days'],
                results['total_trades'],
                f"${results['total_commission']:.2f}"
            ]
        })
        
        results_df.to_csv(output_file, index=False)
        print(f"Resultados guardados en {output_file}")
    
    # Mostrar gráfico si se solicita
    if plot:
        fig = backtest.plot_equity_curve(results)
        plt.title(f"Curva de Equidad - {strategy_name}")
        plt.show()
    
    return results


def download_multiple(tickers, start_date, output_dir):
    """
    Descarga datos históricos para múltiples símbolos.
    
    Args:
        tickers: Lista de símbolos a descargar
        start_date: Fecha de inicio (formato YYYY-MM-DD)
        output_dir: Directorio de salida para guardar los datos
    
    Returns:
        Diccionario con los DataFrames descargados
    """
    print(f"Descargando datos para {len(tickers)} símbolos desde {start_date}...")
    
    # Crear directorio si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results = {}
    
    for ticker in tickers:
        try:
            output_file = os.path.join(output_dir, f"{ticker}.csv")
            data = download_data(ticker, start_date, output_file)
            
            if data is not None:
                results[ticker] = data
            
        except Exception as e:
            print(f"Error al descargar {ticker}: {str(e)}")
    
    print(f"Descarga completada. Se descargaron datos para {len(results)} de {len(tickers)} símbolos.")
    return results


def run_dashboard():
    """
    Ejecuta el dashboard de Streamlit.
    """
    dashboard_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "dashboard", "app.py"
    )
    
    if not os.path.exists(dashboard_path):
        print(f"Error: No se encontró el archivo {dashboard_path}")
        return
    
    print("Ejecutando dashboard...")
    os.system(f"streamlit run {dashboard_path}")


def main():
    """Función principal que procesa los argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description='Análisis de Medias Móviles para Trading')
    
    # Crear subparsers para los diferentes comandos
    subparsers = parser.add_subparsers(dest='command', help='Comando a ejecutar')
    
    # Comando: download
    download_parser = subparsers.add_parser('download', help='Descargar datos históricos hasta hoy')
    download_parser.add_argument('--ticker', type=str, help='Símbolo (o lista separada por comas)')
    download_parser.add_argument('--file', type=str, help='Archivo de texto con símbolos (uno por línea)')
    download_parser.add_argument('--start', type=str, default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                               help='Fecha de inicio (YYYY-MM-DD)')
    download_parser.add_argument('--output', type=str, help='Archivo de salida o directorio (para múltiples símbolos)')
    
    # Comando: analyze
    analyze_parser = subparsers.add_parser('analyze', help='Ejecutar análisis de medias móviles')
    analyze_parser.add_argument('--data', type=str, required=True, help='Archivo CSV con datos')
    analyze_parser.add_argument('--sma', type=str, default='20,50', help='Períodos de SMA (separados por comas)')
    analyze_parser.add_argument('--ema', type=str, default='12,26', help='Períodos de EMA (separados por comas)')
    analyze_parser.add_argument('--output', type=str, help='Archivo de salida (CSV)')
    
    # Comando: optimize
    optimize_parser = subparsers.add_parser('optimize', help='Optimizar medias móviles')
    optimize_parser.add_argument('--data', type=str, required=True, help='Archivo CSV con datos')
    optimize_parser.add_argument('--sma-min', type=int, default=5, help='SMA mínimo')
    optimize_parser.add_argument('--sma-max', type=int, default=100, help='SMA máximo')
    optimize_parser.add_argument('--ema-min', type=int, default=5, help='EMA mínimo')
    optimize_parser.add_argument('--ema-max', type=int, default=100, help='EMA máximo')
    optimize_parser.add_argument('--step', type=int, default=5, help='Paso para los rangos')
    optimize_parser.add_argument('--metric', type=str, default='sharpe', 
                               choices=['profit', 'sharpe', 'drawdown'], help='Métrica a optimizar')
    optimize_parser.add_argument('--output', type=str, help='Archivo de salida (CSV)')
    
    # Comando: backtest
    backtest_parser = subparsers.add_parser('backtest', help='Ejecutar backtest')
    backtest_parser.add_argument('--data', type=str, required=True, help='Archivo CSV con datos')
    backtest_parser.add_argument('--strategy', type=str, required=True, 
                               choices=['sma_ema', 'double_sma', 'double_ema'], help='Estrategia')
    backtest_parser.add_argument('--param1', type=int, required=True, help='Primer parámetro')
    backtest_parser.add_argument('--param2', type=int, required=True, help='Segundo parámetro')
    backtest_parser.add_argument('--capital', type=float, default=10000, help='Capital inicial')
    backtest_parser.add_argument('--size', type=float, default=1.0, help='Tamaño de posición (0-1)')
    backtest_parser.add_argument('--commission', type=float, default=0.001, help='Comisión por operación')
    backtest_parser.add_argument('--plot', action='store_true', help='Mostrar gráfico')
    backtest_parser.add_argument('--output', type=str, help='Archivo de salida (CSV)')
    
    # Comando: dashboard
    subparsers.add_parser('dashboard', help='Ejecutar dashboard de Streamlit')
    
    # Procesar argumentos
    args = parser.parse_args()
    
    # Si no hay comando, mostrar ayuda
    if not args.command:
        parser.print_help()
        return
    
    # Ejecutar comando correspondiente
    if args.command == 'download':
        # Verificar si tenemos un ticker o un archivo con tickers
        if args.ticker and args.file:
            print("ERROR: No puedes especificar --ticker y --file simultáneamente")
            return
        elif not args.ticker and not args.file:
            print("ERROR: Debes especificar --ticker o --file")
            return
        
        # Procesar múltiples tickers
        if args.file:
            try:
                with open(args.file, 'r') as f:
                    tickers = [line.strip() for line in f if line.strip()]
            except Exception as e:
                print(f"Error al leer el archivo {args.file}: {str(e)}")
                return
                
            if not tickers:
                print(f"No se encontraron símbolos en el archivo {args.file}")
                return
                
            download_multiple(tickers, args.start, args.output or 'datos')
        elif ',' in args.ticker:
            # Lista de tickers separados por comas
            tickers = [t.strip() for t in args.ticker.split(',')]
            download_multiple(tickers, args.start, args.output or 'datos')
        else:
            # Un solo ticker
            download_data(args.ticker, args.start, args.output)
    
    elif args.command == 'analyze':
        # Cargar datos
        data = pd.read_csv(args.data, index_col=0, parse_dates=True)
        
        # Convertir argumentos string a listas
        sma_periods = [int(x) for x in args.sma.split(',')]
        ema_periods = [int(x) for x in args.ema.split(',')]
        
        # Ejecutar análisis
        df_indicators = run_analysis(data, sma_periods, ema_periods, args.output)
    
    elif args.command == 'optimize':
        # Cargar datos
        data = pd.read_csv(args.data, index_col=0, parse_dates=True)
        
        # Crear rangos
        sma_range = list(range(args.sma_min, args.sma_max + 1, args.step))
        ema_range = list(range(args.ema_min, args.ema_max + 1, args.step))
        
        # Ejecutar optimización
        results = run_optimization(data, sma_range, ema_range, args.metric, args.output)
    
    elif args.command == 'backtest':
        # Cargar datos
        data = pd.read_csv(args.data, index_col=0, parse_dates=True)
        
        # Ejecutar backtest
        run_backtest(
            data, 
            args.strategy, 
            args.param1, 
            args.param2, 
            args.capital, 
            args.size, 
            args.commission, 
            args.plot, 
            args.output
        )
    
    elif args.command == 'dashboard':
        run_dashboard()


if __name__ == "__main__":
    main()