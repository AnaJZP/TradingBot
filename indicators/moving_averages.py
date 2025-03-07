import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Union

class MovingAverages:
    """
    Clase para calcular y optimizar medias móviles simples y exponenciales.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Inicializa el objeto con los datos del mercado.
        
        Args:
            df: DataFrame con datos de mercado. Debe contener al menos una columna 'close' o 'Close'.
        """
        self.df = df.copy()
        
        # Verificar si existe la columna 'close' o 'Close' y estandarizar nombres
        if 'close' in self.df.columns:
            self.price_col = 'close'
        elif 'Close' in self.df.columns:
            # Renombrar columna para mantener consistencia
            self.df['close'] = self.df['Close']
            self.price_col = 'close'
        else:
            raise ValueError("El DataFrame debe contener una columna 'close' o 'Close'")
            
        # También estandarizar otros nombres de columnas si existen
        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Volume': 'volume',
            'Adj Close': 'adj_close'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in self.df.columns and new_name not in self.df.columns:
                self.df[new_name] = self.df[old_name]
    
    def simple_moving_average(self, period: int, column: str = None) -> pd.Series:
        """
        Calcula la media móvil simple (SMA).
        
        Args:
            period: Número de períodos para la media móvil
            column: Columna sobre la que calcular la media móvil (por defecto usa self.price_col)

        Returns:
            Serie con la media móvil simple
        """
        col = column if column is not None else self.price_col
        return self.df[col].rolling(window=period).mean()
    
    def exponential_moving_average(self, period: int, column: str = None) -> pd.Series:
        """
        Calcula la media móvil exponencial (EMA).
        
        Args:
            period: Número de períodos para la media móvil
            column: Columna sobre la que calcular la media móvil (por defecto usa self.price_col)

        Returns:
            Serie con la media móvil exponencial
        """
        col = column if column is not None else self.price_col
        return self.df[col].ewm(span=period, adjust=False).mean()
    
    def add_indicators(self, sma_periods: List[int], ema_periods: List[int]) -> pd.DataFrame:
        """
        Añade múltiples SMAs y EMAs al DataFrame.
        
        Args:
            sma_periods: Lista de períodos para calcular SMAs
            ema_periods: Lista de períodos para calcular EMAs
            
        Returns:
            DataFrame con los indicadores añadidos
        """
        df_result = self.df.copy()
        
        for period in sma_periods:
            df_result[f'SMA_{period}'] = self.simple_moving_average(period)
            
        for period in ema_periods:
            df_result[f'EMA_{period}'] = self.exponential_moving_average(period)
            
        return df_result
    
    def find_optimal_combination(self, 
                                sma_range: List[int], 
                                ema_range: List[int], 
                                test_period: int = 252,  # Aproximadamente un año de trading
                                metric: str = 'profit') -> Dict:
        """
        Encuentra la combinación óptima de SMAs y EMAs basada en una estrategia de cruce.
        
        Args:
            sma_range: Lista de períodos de SMA a probar
            ema_range: Lista de períodos de EMA a probar
            test_period: Período de testeo en días
            metric: Métrica a optimizar ('profit', 'sharpe', 'drawdown')
            
        Returns:
            Diccionario con la combinación óptima y sus resultados
        """
        from trading.signals import CrossoverSignals
        from trading.backtest import Backtest
        
        best_metric = -np.inf if metric != 'drawdown' else np.inf
        best_combination = None
        best_results = None
        
        # Asegurar que tenemos suficientes datos
        if len(self.df) < max(max(sma_range), max(ema_range)) + test_period:
            raise ValueError("No hay suficientes datos para la optimización")
        
        # Iterar sobre todas las combinaciones posibles
        for sma_period in sma_range:
            for ema_period in ema_range:
                # Evitar combinaciones donde SMA y EMA tienen el mismo período
                if sma_period == ema_period:
                    continue
                
                # Añadir los indicadores
                df_with_indicators = self.add_indicators([sma_period], [ema_period])
                
                # Generar señales
                signals = CrossoverSignals(df_with_indicators)
                df_signals = signals.sma_ema_crossover(sma_period, ema_period)
                
                # Ejecutar backtest
                backtest = Backtest(df_signals)
                results = backtest.run(initial_capital=10000)
                
                # Evaluar métrica
                current_metric = None
                if metric == 'profit':
                    current_metric = results['final_capital'] - 10000
                elif metric == 'sharpe':
                    current_metric = results['sharpe_ratio']
                elif metric == 'drawdown':
                    current_metric = results['max_drawdown']
                
                # Actualizar mejor combinación si es necesario
                if ((metric != 'drawdown' and current_metric > best_metric) or 
                    (metric == 'drawdown' and current_metric < best_metric)):
                    best_metric = current_metric
                    best_combination = {'sma': sma_period, 'ema': ema_period}
                    best_results = results
        
        return {
            'combination': best_combination,
            'results': best_results,
            'metric_value': best_metric,
            'metric_name': metric
        }