# TradingBot
TradingBot para capturar señales de compra y venta de activos financieros a través de Yahoo Finance

# Análisis de Medias Móviles para Trading

Este proyecto implementa un sistema completo para el análisis de medias móviles, optimización de parámetros y backtesting de estrategias de trading. Diseñado como herramienta educativa para cursos de análisis bursátil.

## Características Principales

- **Indicadores Técnicos**: Implementación de medias móviles simples (SMA) y exponenciales (EMA)
- **Señales de Trading**: Generación de señales basadas en cruces de diferentes tipos de medias móviles
- **Optimización de Parámetros**: Encontrar la combinación óptima de períodos para maximizar rentabilidad o Sharpe Ratio
- **Backtesting**: Simulación completa de operaciones con cálculo de métricas de rendimiento
- **Dashboard Interactivo**: Visualización dinámica de datos, indicadores y resultados con Streamlit
- **Interfaz de Línea de Comandos**: Ejecución fácil de diferentes funcionalidades desde la terminal

## Estructura del Proyecto

```
proyecto_analisis_bursatil/
│
├── main.py                   # Módulo principal para ejecutar todo el sistema
├── indicators/
│   └── moving_averages.py    # Implementación de SMA y EMA con optimización
├── trading/
│   ├── signals.py            # Generación de señales de compra/venta
│   └── backtest.py           # Módulo para simular pérdidas y ganancias
├── dashboard/
│   └── app.py                # Aplicación Streamlit para visualizar resultados
├── data/                     # Directorio para almacenar datos descargados
├── results/                  # Directorio para guardar resultados de análisis
└── requirements.txt          # Dependencias del proyecto
```

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/proyecto_analisis_bursatil.git
cd proyecto_analisis_bursatil
```

2. Crear un entorno virtual (opcional pero recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### Descargar Datos

Para un solo símbolo:
```bash
python main.py download --ticker AAPL --start 2022-01-01 --output data/aapl.csv
```

Para múltiples símbolos:
```bash
python main.py download --ticker "AAPL,MSFT,GOOGL" --start 2022-01-01 --output data/
```

Desde un archivo con lista de símbolos:
```bash
python main.py download --file simbolos.txt --start 2022-01-01 --output data/
```

### Análisis con Medias Móviles

```bash
python main.py analyze --data data/aapl.csv --sma 20,50 --ema 12,26 --output results/analysis.csv
```

### Optimización de Parámetros

```bash
python main.py optimize --data data/aapl.csv --sma-min 5 --sma-max 100 --ema-min 5 --ema-max 100 --step 5 --metric sharpe --output results/optimizacion.csv
```

### Ejecutar Backtest

```bash
python main.py backtest --data data/aapl.csv --strategy sma_ema --param1 20 --param2 50 --capital 10000 --commission 0.001 --plot --output results/backtest.csv
```

### Ejecutar Dashboard Interactivo

```bash
python main.py dashboard
```

O directamente:
```bash
streamlit run dashboard/app.py
```

## Métricas de Evaluación

El sistema utiliza varias métricas para evaluar el rendimiento de las estrategias:

### Ratio de Sharpe

El Ratio de Sharpe mide el rendimiento ajustado por riesgo de una inversión. Se calcula como:

```
Ratio de Sharpe = (Retorno Anualizado - Tasa Libre de Riesgo) / Volatilidad Anualizada
```

Donde:
- **Retorno Anualizado**: Ganancia porcentual anual de la estrategia
- **Tasa Libre de Riesgo**: Rendimiento de una inversión sin riesgo (en nuestro sistema se asume 0 para simplificar)
- **Volatilidad Anualizada**: Desviación estándar anualizada de los retornos diarios

**Interpretación**:
- Ratio > 1.0: Bueno
- Ratio > 2.0: Muy bueno
- Ratio > 3.0: Excelente

Un Ratio de Sharpe más alto indica mejor rendimiento ajustado por riesgo.

### Drawdown Máximo

El Drawdown es la caída desde un máximo hasta el punto más bajo antes de alcanzar un nuevo máximo. El Drawdown Máximo representa la pérdida máxima histórica desde un pico hasta un valle, expresado como porcentaje.

```
Drawdown(t) = (Valor(t) / Máximo Valor Previo) - 1
Drawdown Máximo = Mínimo de todos los Drawdowns
```

**Interpretación**:
- Mide el riesgo de pérdida de una estrategia
- Indica la caída más grande que un inversor habría experimentado
- Valor más pequeño (menos negativo) es mejor

### Retorno Total

El Retorno Total mide la ganancia porcentual total durante todo el período de prueba:

```
Retorno Total = (Capital Final / Capital Inicial) - 1
```

### Retorno Anualizado

El Retorno Anualizado convierte el retorno total a una tasa anual equivalente:

```
Retorno Anualizado = (1 + Retorno Total)^(1 / Años) - 1
```

Donde Años = Días de Trading / 252 (asumiendo 252 días de trading por año).

## Estrategias Implementadas

El sistema implementa tres estrategias principales:

1. **SMA-EMA Crossover**: Genera señales de compra cuando EMA cruza por encima de SMA, y señales de venta cuando cruza por debajo.

2. **Double SMA Crossover**: Genera señales basadas en el cruce de dos SMAs de diferentes períodos (uno corto y uno largo).

3. **Double EMA Crossover**: Similar a la anterior, pero usando dos EMAs para mayor sensibilidad a los cambios de precio.

## Recomendaciones para Optimización

Para obtener mejores resultados en la optimización:

1. **Datos Suficientes**: Usar al menos 2-3 años de datos históricos para resultados más confiables.

2. **Períodos Adecuados**: Para SMA y EMA cortos, probar rangos de 5-50 días. Para largos, 20-200 días.

3. **Métrica Adecuada**: 
   - Para estrategias de alto rendimiento: optimizar por 'profit'
   - Para riesgo controlado: optimizar por 'sharpe'
   - Para minimizar pérdidas: optimizar por 'drawdown'

4. **Validación**: Siempre validar los resultados optimizados en diferentes períodos de tiempo o símbolos.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, siéntete libre de fork el repositorio y enviar pull requests.

## Licencia

Este proyecto no tiene licencia, simplemente me gusta el Trading.

## Contacto

Para preguntas o sugerencias, por favor contacta a [ana.jzp@gmail.com].

---

Desarrollado como herramienta educativa para cursos de análisis bursátil y trading.
