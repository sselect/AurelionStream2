"""Análisis exploratorio y reportes para las ventas.

Lee los archivos Excel (clientes.xlsx, detalle_ventas.xlsx, productos.xlsx, ventas.xlsx)
esperados en la carpeta `DataBase/` (o en la raíz) y genera:
- estadísticas descriptivas
- detección de outliers
- correlaciones
- gráficos (guardados en outputs/figures)
- archivo de resumen en outputs/report.txt

Uso:
    python src/analysis.py

Requiere: pandas, numpy, matplotlib, seaborn, openpyxl
"""
from pathlib import Path
import os
import sys
import unicodedata
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_DIR = PROJECT_ROOT / "DataBase"
OUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"



def normalize_col_name(s: str) -> str:
    # lower, strip, remove accents, replace spaces with _
    s = str(s).strip().lower()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = s.replace(' ', '_')
    return s


def find_best_col(df, candidates):
    # intenta encontrar la primera coincidencia razonable en el df.columns
    norm_cols = {normalize_col_name(c): c for c in df.columns}
    for cand in candidates:
        n = normalize_col_name(cand)
        if n in norm_cols:
            return norm_cols[n]
    # fallback: try partial match
    for n, orig in norm_cols.items():
        for cand in candidates:
            if normalize_col_name(cand) in n:
                return orig
    return None


def describe_series(s: pd.Series):
    return {
        'count': int(s.count()),
        'mean': float(s.mean()),
        'median': float(s.median()),
        'std': float(s.std()),
        'min': float(s.min()),
        'max': float(s.max()),
        'skew': float(s.skew())
    }


def detect_bimodal(s: pd.Series, bins=30):
    # detect simple bimodality by counting peaks in the histogram counts
    counts, edges = np.histogram(s.dropna(), bins=bins)
    peaks = 0
    for i in range(1, len(counts)-1):
        if counts[i] > counts[i-1] and counts[i] > counts[i+1]:
            peaks += 1
    return peaks >= 2, peaks


def iqr_outliers(s: pd.Series):
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    mask = (s < low) | (s > high)
    return mask, low, high


def ensure_dirs():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    ensure_dirs()

    # rutas directas a los archivos en DataBase
    files = {
        'clientes': DB_DIR / 'clientes.xlsx',
        'detalle_ventas': DB_DIR / 'detalle_ventas.xlsx',
        'productos': DB_DIR / 'productos.xlsx',
        'ventas': DB_DIR / 'ventas.xlsx'
    }

    dfs = {}
    for key, path in files.items():
        try:
            dfs[key] = pd.read_excel(path, engine='openpyxl')
            print(f"Cargado {path.name} desde {path}")
        except FileNotFoundError as e:
            print(f"Advertencia: No se encontró {path}. Se continuará sin este archivo.")
            dfs[key] = None

    # Preferir detalle_ventas o ventas para columnas numéricas
    # Buscamos columnas: precio_unitario, cantidad, total_venta
    candidate_price = ['precio_unitario', 'precio', 'price', 'unit_price']
    candidate_qty = ['cantidad', 'cantidad_vendida', 'qty', 'quantity']
    candidate_total = ['total_venta', 'total', 'importe', 'total_amount']

    # pick a df that contains relevant columns
    df_source = dfs.get('detalle_ventas') if dfs.get('detalle_ventas') is not None else dfs.get('ventas')
    if df_source is None:
        print("No se encontró `detalle_ventas.xlsx` ni `ventas.xlsx` con datos. Abortando.")
        sys.exit(1)

    # find actual column names
    price_col = find_best_col(df_source, candidate_price)
    qty_col = find_best_col(df_source, candidate_qty)
    total_col = find_best_col(df_source, candidate_total)

    if not any([price_col, qty_col, total_col]):
        print("No se encontraron columnas esperadas (precio/cantidad/total). Revisa los nombres de columna.")
        print("Columnas disponibles:", list(df_source.columns))
        sys.exit(1)

    # Convert to numeric when possible
    if price_col:
        df_source[price_col] = pd.to_numeric(df_source[price_col], errors='coerce')
    if qty_col:
        df_source[qty_col] = pd.to_numeric(df_source[qty_col], errors='coerce')
    if total_col:
        df_source[total_col] = pd.to_numeric(df_source[total_col], errors='coerce')

    target_cols = {}
    if price_col:
        target_cols['precio_unitario'] = price_col
    if qty_col:
        target_cols['cantidad'] = qty_col
    if total_col:
        target_cols['total_venta'] = total_col

    report_lines = []
    report_lines.append('ANÁLISIS DE VENTAS')
    report_lines.append('==================')

    # Estadísticas descriptivas
    report_lines.append('\nEstadísticas descriptivas:')
    for label, col in target_cols.items():
        s = df_source[col]
        stats = describe_series(s)
        report_lines.append(f"\nVariable: {label} (col: {col})")
        for k, v in stats.items():
            report_lines.append(f"  {k}: {v}")

    # Distribuciones: skewness, bimodality heuristic
    report_lines.append('\nDistribuciones y sesgo:')
    for label, col in target_cols.items():
        s = df_source[col]
        skew = float(s.skew())
        bimodal, peaks = detect_bimodal(s.dropna())
        report_lines.append(f"{label}: skew={skew:.3f}, peaks_detected={peaks}, bimodal_guess={bimodal}")

    # Correlaciones
    num_cols = [c for c in target_cols.values()]
    df_corr = df_source[num_cols].dropna()
    corr = df_corr.corr()
    report_lines.append('\nCorrelaciones (matriz):')
    report_lines.append(str(corr))

    # Specific relationships if present
    if 'precio_unitario' in target_cols and 'total_venta' in target_cols:
        r = corr.loc[target_cols['precio_unitario'], target_cols['total_venta']]
        report_lines.append(f"Relación precio_unitario vs total_venta: r = {r:.2f}")
    if 'cantidad' in target_cols and 'total_venta' in target_cols:
        r = corr.loc[target_cols['cantidad'], target_cols['total_venta']]
        report_lines.append(f"Relación cantidad vs total_venta: r = {r:.2f}")

    # Outliers (IQR) por variable
    report_lines.append('\nOutliers detectados (IQR):')
    for label, col in target_cols.items():
        s = df_source[col]
        mask, low, high = iqr_outliers(s.dropna())
        outliers = s.dropna()[mask]
        report_lines.append(f"{label}: {len(outliers)} outliers (low={low:.2f}, high={high:.2f})")
        # list top 10
        if len(outliers) > 0:
            report_lines.append(f"  Top outliers (muestra hasta 10): {outliers.sort_values(ascending=False).head(10).tolist()}")

    # Save report
    report_path = OUT_DIR / 'report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"Reporte guardado en {report_path}")

    # Configuración de estilo mejorada
    sns.set_style("whitegrid")
    colors = sns.color_palette("husl", 3)
    
    # 1. Histogramas mejorados con KDE para todas las variables numéricas
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle('Distribución de Variables Principales por Producto', fontsize=16, y=0.95)
    
    # Agrupar por producto para análisis más detallado
    product_stats = df_source.groupby('id_producto').agg({
        target_cols['precio_unitario']: 'mean',
        target_cols['cantidad']: 'sum',
        target_cols['total_venta']: 'sum'
    }).reset_index()
    
    for idx, (label, col) in enumerate(target_cols.items()):
        sns.histplot(data=product_stats, x=col, kde=True, ax=axes[idx], color=colors[idx])
        axes[idx].set_title(f'Distribución de {label} por Producto')
        axes[idx].set_xlabel(f'{label} (media por producto, skew={float(product_stats[col].skew()):.3f})')
    
    plt.tight_layout()
    p = FIG_DIR / 'distribuciones_por_producto.png'
    plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Guardado gráfico de distribuciones por producto: {p}")

    # 2. Boxplots por producto (TOP 10 productos más vendidos)
    plt.figure(figsize=(15, 7))
    top_products = df_source.groupby('id_producto')[target_cols['total_venta']].sum().nlargest(10).index
    df_top = df_source[df_source['id_producto'].isin(top_products)]
    
    sns.boxplot(data=df_top, x='id_producto', y=target_cols['precio_unitario'])
    plt.title('Distribución de Precios por Producto (Top 10 productos por venta total)', fontsize=12)
    plt.xlabel('ID Producto')
    plt.ylabel('Precio Unitario')
    plt.xticks(rotation=45)
    plt.tight_layout()
    p = FIG_DIR / 'boxplots_por_producto.png'
    plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Guardado boxplot por producto: {p}")

    # 3. Heatmap de correlaciones mejorado
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, 
                mask=mask,
                annot=True,
                cmap='RdYlBu_r',
                fmt='.2f',
                square=True,
                linewidths=1,
                cbar_kws={"shrink": .5})
    plt.title('Matriz de Correlaciones', fontsize=12, pad=20)
    plt.tight_layout()
    p = FIG_DIR / 'heatmap_correlaciones.png'
    plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Guardado heatmap: {p}")

    # 4. Scatter plot de productos
    if 'precio_unitario' in target_cols and 'total_venta' in target_cols:
        plt.figure(figsize=(10, 6))
        # Agregar por producto
        product_performance = df_source.groupby('id_producto').agg({
            target_cols['precio_unitario']: 'mean',
            target_cols['total_venta']: 'sum',
            target_cols['cantidad']: 'sum'
        }).reset_index()
        
        # Tamaño del punto proporcional a la cantidad vendida
        plt.scatter(product_performance[target_cols['precio_unitario']],
                   product_performance[target_cols['total_venta']],
                   s=product_performance[target_cols['cantidad']] * 20,
                   alpha=0.6)
        
        # Añadir ID de producto para los top 5
        top_5 = product_performance.nlargest(5, target_cols['total_venta'])
        for _, row in top_5.iterrows():
            plt.annotate(f'ID: {row["id_producto"]}',
                        (row[target_cols['precio_unitario']], row[target_cols['total_venta']]),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.title('Relación Precio vs Venta Total por Producto', fontsize=12)
        plt.xlabel('Precio Unitario Promedio')
        plt.ylabel('Total de Ventas')
        plt.tight_layout()
        p = FIG_DIR / 'scatter_productos.png'
        plt.savefig(p, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Guardado scatter plot de productos: {p}")

    # 5. Top 10 productos por venta total
    if 'total_venta' in target_cols:
        plt.figure(figsize=(12, 6))
        # Agrupar por producto
        top_10_products = df_source.groupby('id_producto').agg({
            target_cols['total_venta']: 'sum',
            'nombre_producto': 'first'  # Tomar el nombre del producto
        }).nlargest(10, target_cols['total_venta'])
        
        plt.figure(figsize=(14, 6))
        bars = plt.bar(range(len(top_10_products)), top_10_products[target_cols['total_venta']])
        
        # Añadir etiquetas con ID y nombre del producto
        plt.xticks(range(len(top_10_products)), 
                  [f'ID: {idx}\n{name[:20]}' for idx, name in 
                   zip(top_10_products.index, top_10_products['nombre_producto'])],
                  rotation=45, ha='right')
        
        # Añadir valores en las barras
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.0f}',
                    ha='center', va='bottom')
        
        plt.title('Top 10 Productos por Venta Total', fontsize=12)
        plt.xlabel('Producto')
        plt.ylabel('Total de Ventas ($)')
        plt.tight_layout()
        p = FIG_DIR / 'top_10_productos.png'
        plt.savefig(p, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Guardado gráfico top 10 productos: {p}")

    # 6. Gráficos de dona para método de pago y ciudad
    if dfs['ventas'] is not None and dfs['clientes'] is not None:
        # Preparar datos
        ventas_df = dfs['ventas']
        clientes_df = dfs['clientes']
        
        # Combinar ventas con detalle_ventas para obtener montos
        ventas_detalle = pd.merge(
            ventas_df,
            df_source.groupby('id_venta')[target_cols['total_venta']].sum().reset_index(),
            on='id_venta'
        )
        
        # Combinar con clientes para obtener ciudad
        ventas_ciudad = pd.merge(
            ventas_detalle,
            clientes_df[['id_cliente', 'ciudad']],
            on='id_cliente'
        )
        
        # Crear figura con dos subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # 1. Gráfico de dona para método de pago
        pago_stats = ventas_detalle.groupby('medio_pago')[target_cols['total_venta']].sum()
        total_ventas = pago_stats.sum()
        pago_pcts = (pago_stats / total_ventas * 100).round(1)
        
        colors_pago = plt.cm.Set3(np.linspace(0, 1, len(pago_stats)))
        wedges, texts, autotexts = ax1.pie(pago_stats,
                                         labels=[f'{idx}\n(${val:,.0f})' for idx, val in pago_stats.items()],
                                         autopct='%1.1f%%',
                                         colors=colors_pago,
                                         pctdistance=0.85)
        # Crear el efecto de dona
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        ax1.add_artist(centre_circle)
        ax1.set_title('Distribución por Método de Pago\n(Total de Ventas)', pad=20)
        
        # 2. Gráfico de dona para ciudad
        ciudad_stats = ventas_ciudad.groupby('ciudad')[target_cols['total_venta']].sum()
        ciudad_pcts = (ciudad_stats / total_ventas * 100).round(1)
        
        colors_ciudad = plt.cm.Set3(np.linspace(0, 1, len(ciudad_stats)))
        wedges, texts, autotexts = ax2.pie(ciudad_stats,
                                         labels=[f'{idx}\n(${val:,.0f})' for idx, val in ciudad_stats.items()],
                                         autopct='%1.1f%%',
                                         colors=colors_ciudad,
                                         pctdistance=0.85)
        # Crear el efecto de dona
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        ax2.add_artist(centre_circle)
        ax2.set_title('Distribución por Ciudad\n(Total de Ventas)', pad=20)
        
        plt.tight_layout()
        p = FIG_DIR / 'distribucion_pago_ciudad.png'
        plt.savefig(p, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Guardado gráfico de distribución por pago y ciudad: {p}")
        
        # Añadir estadísticas al reporte
        report_lines.append("\nDISTRIBUCIÓN POR MÉTODO DE PAGO:")
        for metodo, porcentaje in pago_pcts.items():
            monto = pago_stats[metodo]
            report_lines.append(f"- {metodo}: {porcentaje}% (${monto:,.2f})")
        
        report_lines.append("\nDISTRIBUCIÓN POR CIUDAD:")
        for ciudad, porcentaje in ciudad_pcts.items():
            monto = ciudad_stats[ciudad]
            report_lines.append(f"- {ciudad}: {porcentaje}% (${monto:,.2f})")

    # Análisis adicional para el reporte
    report_lines.append('\nANÁLISIS PROFUNDO:')
    
    # 1. Análisis de concentración de ventas
    if 'total_venta' in target_cols:
        total_ventas = df_source[target_cols['total_venta']].sum()
        top_10_sum = df_source.nlargest(10, target_cols['total_venta'])[target_cols['total_venta']].sum()
        concentracion = (top_10_sum / total_ventas) * 100
        report_lines.append(f"\nConcentración de ventas:")
        report_lines.append(f"- Las top 10 ventas representan el {concentracion:.1f}% del total")
    
    # 2. Análisis de rangos de precios
    if 'precio_unitario' in target_cols:
        price_col = target_cols['precio_unitario']
        quantiles = df_source[price_col].quantile([0.25, 0.5, 0.75])
        report_lines.append(f"\nSegmentación por precio:")
        report_lines.append(f"- Rango económico: < {quantiles[0.25]:.0f}")
        report_lines.append(f"- Rango medio: {quantiles[0.25]:.0f} - {quantiles[0.75]:.0f}")
        report_lines.append(f"- Rango premium: > {quantiles[0.75]:.0f}")
    
    # 3. Patrones de cantidad
    if 'cantidad' in target_cols:
        qty_col = target_cols['cantidad']
        mode_qty = df_source[qty_col].mode().iloc[0]
        avg_qty = df_source[qty_col].mean()
        report_lines.append(f"\nPatrones de compra:")
        report_lines.append(f"- Cantidad más frecuente: {mode_qty:.0f} unidades")
        report_lines.append(f"- Promedio de unidades por venta: {avg_qty:.1f}")
    
    # 4. Interpretación final
    report_lines.append("\nINTERPRETACIÓN INTEGRAL:")
    report_lines.append("1. Distribución de precios:")
    report_lines.append("   - Los precios muestran una distribución relativamente normal con ligero sesgo positivo")
    report_lines.append("   - Existe segmentación clara en rangos de precio (económico, medio, premium)")
    
    report_lines.append("\n2. Patrones de venta:")
    report_lines.append("   - Las ventas muestran alta concentración en algunos tickets de alto valor")
    report_lines.append("   - La cantidad por venta tiende a ser estable, con pocos casos extremos")
    
    report_lines.append("\n3. Correlaciones e impacto:")
    report_lines.append("   - El precio unitario es el principal driver del total de venta")
    report_lines.append("   - La cantidad vendida tiene un impacto moderado pero significativo")
    report_lines.append("   - No hay relación clara entre precio y cantidad, sugiriendo que el precio no determina el volumen")

    print('\nAnálisis completado. Revisa outputs/report.txt y outputs/figures/*.png')


if __name__ == '__main__':
    main()
