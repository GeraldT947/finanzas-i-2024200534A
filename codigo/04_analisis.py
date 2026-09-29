# Torres Condor Gerald Luis
# 2024200534A
# Tema 44: Analisis con 5 variables
# Fecha: 2026-09-28

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

BASE = "/content/drive/MyDrive/finanzas-i-2024200534A"
logging.basicConfig(filename=f'{BASE}/log_ejecucion.txt', level=logging.INFO)
CODIGO_MATRICULA = "2024200534A"
CARPETA_PROCESADOS = f"{BASE}/datos_procesados"
CARPETA_SALIDAS = f"{BASE}/salidas"
os.makedirs(CARPETA_SALIDAS, exist_ok=True)


def cargar_datos():
    ruta = f"{CARPETA_PROCESADOS}/datos_procesados_{CODIGO_MATRICULA}.csv"
    df = pd.read_csv(ruta, parse_dates=['fecha'])
    df = df.set_index('fecha')
    logging.info(f"Datos cargados: {len(df)} filas, {len(df.columns)} columnas")
    return df


def tabla_descriptivos(df):
    desc = df.describe().T
    desc = desc[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    desc.columns = ['Obs', 'Media', 'Desv.Est.', 'Min', 'P25', 'Mediana', 'P75', 'Max']
    desc.to_csv(f"{CARPETA_SALIDAS}/tabla_descriptivos.csv")
    desc.to_latex(f"{CARPETA_SALIDAS}/tabla_descriptivos.tex", float_format="%.2f")
    logging.info("Tabla descriptivos guardada")
    print(desc)


def matriz_correlacion(df):
    corr = df.corr()
    corr.to_csv(f"{CARPETA_SALIDAS}/tabla_correlacion.csv")
    corr.to_latex(f"{CARPETA_SALIDAS}/tabla_correlacion.tex", float_format="%.3f")
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".3f", vmin=-1, vmax=1)
    plt.title("Matriz de correlaciones")
    plt.tight_layout()
    plt.savefig(f"{CARPETA_SALIDAS}/fig_correlacion.png", dpi=150)
    plt.close()
    logging.info("Matriz correlacion guardada")
    print(corr)


def grafico_series(df):
    fig, axes = plt.subplots(5, 1, figsize=(10, 14), sharex=True)
    axes[0].plot(df.index, df['indice_bvl'], color='navy')
    axes[0].set_ylabel('Indice BVL')
    axes[0].set_title('Indice General BVL, 2015-2024')
    axes[0].grid(alpha=0.3)
    axes[1].plot(df.index, df['tipo_cambio'], color='green')
    axes[1].set_ylabel('S/ por US$')
    axes[1].set_title('Tipo de cambio')
    axes[1].grid(alpha=0.3)
    axes[2].plot(df.index, df['precio_cobre'], color='darkorange')
    axes[2].set_ylabel('US$/lb')
    axes[2].set_title('Precio del cobre')
    axes[2].grid(alpha=0.3)
    axes[3].plot(df.index, df['tasa_referencia'], color='red')
    axes[3].set_ylabel('%')
    axes[3].set_title('Tasa de referencia BCRP')
    axes[3].grid(alpha=0.3)
    axes[4].plot(df.index, df['ipc'], color='purple')
    axes[4].set_ylabel('Indice IPC')
    axes[4].set_title('Indice de Precios al Consumidor')
    axes[4].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{CARPETA_SALIDAS}/fig_series_tiempo.png", dpi=150)
    plt.close()
    logging.info("Grafico de series guardado")


def regresion_ols(df):
    # Modelo en niveles
    X = df[['tipo_cambio', 'precio_cobre']]
    X = sm.add_constant(X)
    y = df['indice_bvl']
    modelo = sm.OLS(y, X).fit()
    logging.info(f"R2 del modelo en niveles: {modelo.rsquared:.4f}")
    with open(f"{CARPETA_SALIDAS}/regresion_resumen.txt", 'w') as f:
        f.write(modelo.summary().as_text())

    # Modelo en log-diferencias
    df_ret = np.log(df).diff().dropna()
    X2 = df_ret[['tipo_cambio', 'precio_cobre']]
    X2 = sm.add_constant(X2)
    y2 = df_ret['indice_bvl']
    modelo2 = sm.OLS(y2, X2).fit()
    logging.info(f"R2 del modelo en log-diferencias: {modelo2.rsquared:.4f}")
    with open(f"{CARPETA_SALIDAS}/regresion_log_diff.txt", 'w') as f:
        f.write(modelo2.summary().as_text())

    # Modelo ampliado con las 4 variables
    X3 = df_ret[['tipo_cambio', 'precio_cobre', 'tasa_referencia', 'ipc']]
    X3 = sm.add_constant(X3)
    modelo3 = sm.OLS(y2, X3).fit()
    with open(f"{CARPETA_SALIDAS}/regresion_ampliada.txt", 'w') as f:
        f.write(modelo3.summary().as_text())

    print(modelo.summary())
    print(modelo2.summary())
    print(modelo3.summary())


def graficos_dispersion(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0,0].scatter(df['tipo_cambio'], df['indice_bvl'], alpha=0.6, color='green')
    axes[0,0].set_xlabel('Tipo de cambio')
    axes[0,0].set_ylabel('Indice BVL')
    axes[0,0].grid(alpha=0.3)
    axes[0,1].scatter(df['precio_cobre'], df['indice_bvl'], alpha=0.6, color='darkorange')
    axes[0,1].set_xlabel('Precio del cobre')
    axes[0,1].set_ylabel('Indice BVL')
    axes[0,1].grid(alpha=0.3)
    axes[1,0].scatter(df['tasa_referencia'], df['indice_bvl'], alpha=0.6, color='red')
    axes[1,0].set_xlabel('Tasa de referencia')
    axes[1,0].set_ylabel('Indice BVL')
    axes[1,0].grid(alpha=0.3)
    axes[1,1].scatter(df['ipc'], df['indice_bvl'], alpha=0.6, color='purple')
    axes[1,1].set_xlabel('IPC')
    axes[1,1].set_ylabel('Indice BVL')
    axes[1,1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{CARPETA_SALIDAS}/fig_dispersion.png", dpi=150)
    plt.close()
    logging.info("Graficos de dispersion guardados")


if __name__ == "__main__":
    logging.info("=== INICIO ANALISIS ===")
    df = cargar_datos()
    tabla_descriptivos(df)
    matriz_correlacion(df)
    grafico_series(df)
    regresion_ols(df)
    graficos_dispersion(df)
    logging.info("=== FIN ANALISIS ===")
    print("\nAnalisis completado.")
