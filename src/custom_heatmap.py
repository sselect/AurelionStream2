"""Script personalizado para generar heatmaps de correlaciones con diferentes esquemas de colores.

Este script crea múltiples versiones del heatmap de correlaciones basado en detalle_ventas.xlsx
con diferentes paletas de colores y estilos visuales.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuración de rutas
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_DIR = PROJECT_ROOT / "DataBase"
OUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

# Asegurar que existe el directorio de figuras
FIG_DIR.mkdir(parents=True, exist_ok=True)

def normalize_col_name(s: str) -> str:
    """Normalizar nombres de columnas"""
    import unicodedata
    s = str(s).strip().lower()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = s.replace(' ', '_')
    return s

def find_best_col(df, candidates):
    """Encontrar la mejor coincidencia de columna"""
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

def create_custom_heatmaps():
    """Crear múltiples heatmaps personalizados con diferentes estilos"""
    
    # Cargar datos
    df_source = pd.read_excel(DB_DIR / 'detalle_ventas.xlsx', engine='openpyxl')
    print(f"Datos cargados: {len(df_source)} filas")
    
    # Encontrar columnas relevantes
    candidate_price = ['precio_unitario', 'precio', 'price', 'unit_price']
    candidate_qty = ['cantidad', 'cantidad_vendida', 'qty', 'quantity']
    candidate_total = ['total_venta', 'total', 'importe', 'total_amount']
    
    price_col = find_best_col(df_source, candidate_price)
    qty_col = find_best_col(df_source, candidate_qty)
    total_col = find_best_col(df_source, candidate_total)
    
    # Convertir a numérico
    if price_col:
        df_source[price_col] = pd.to_numeric(df_source[price_col], errors='coerce')
    if qty_col:
        df_source[qty_col] = pd.to_numeric(df_source[qty_col], errors='coerce')
    if total_col:
        df_source[total_col] = pd.to_numeric(df_source[total_col], errors='coerce')
    
    # Crear matriz de correlaciones
    num_cols = [col for col in [price_col, qty_col, total_col] if col is not None]
    df_corr = df_source[num_cols].dropna()
    corr = df_corr.corr()
    
    # Renombrar columnas para mejor visualización
    labels_map = {}
    if price_col:
        labels_map[price_col] = 'Precio Unitario'
    if qty_col:
        labels_map[qty_col] = 'Cantidad'
    if total_col:
        labels_map[total_col] = 'Total Venta'
    
    corr_renamed = corr.rename(index=labels_map, columns=labels_map)
    
    # Configuración de estilo general
    plt.style.use('default')
    sns.set_context("notebook", font_scale=1.2)
    
    # 1. HEATMAP CLÁSICO - Azul y Rojo
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_renamed, dtype=bool))
    
    sns.heatmap(corr_renamed, 
                mask=mask,
                annot=True,
                fmt='.3f',
                cmap='RdBu_r',
                center=0,
                square=True,
                linewidths=2,
                cbar_kws={"shrink": .8, "label": "Coeficiente de Correlación"},
                annot_kws={'size': 14, 'weight': 'bold'})
    
    plt.title('Matriz de Correlaciones - Estilo Clásico\n(Datos de Ventas)', 
              fontsize=16, fontweight='bold', pad=25)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    p1 = FIG_DIR / 'heatmap_clasico.png'
    plt.savefig(p1, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Heatmap clásico guardado: {p1}")
    
    # 2. HEATMAP VIBRANTE - Colores Cálidos
    plt.figure(figsize=(12, 9))
    
    # Sin máscara para mostrar toda la matriz
    sns.heatmap(corr_renamed, 
                annot=True,
                fmt='.3f',
                cmap='plasma',
                center=0,
                square=True,
                linewidths=3,
                linecolor='white',
                cbar_kws={"shrink": .7, "label": "Correlación", "orientation": "horizontal", "pad": 0.1},
                annot_kws={'size': 15, 'weight': 'bold', 'color': 'white'})
    
    plt.title('🔥 Matriz de Correlaciones - Estilo Vibrante 🔥\n(Análisis de Ventas Detallado)', 
              fontsize=18, fontweight='bold', pad=30, color='darkred')
    plt.xticks(rotation=30, ha='right', fontsize=12, fontweight='bold')
    plt.yticks(rotation=0, fontsize=12, fontweight='bold')
    
    # Agregar borde decorativo
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_linewidth(3)
        spine.set_edgecolor('darkred')
    
    plt.tight_layout()
    
    p2 = FIG_DIR / 'heatmap_vibrante.png'
    plt.savefig(p2, dpi=300, bbox_inches='tight', facecolor='#f8f8f8')
    plt.close()
    print(f"✓ Heatmap vibrante guardado: {p2}")
    
    # 3. HEATMAP MINIMALISTA - Estilo Moderno
    plt.figure(figsize=(10, 8))
    
    # Crear una paleta personalizada
    colors = ['#2E4057', '#048A81', '#54C6EB', '#F18F01', '#C73E1D']
    custom_cmap = sns.blend_palette(colors, as_cmap=True)
    
    mask = np.triu(np.ones_like(corr_renamed, dtype=bool))
    
    ax = sns.heatmap(corr_renamed, 
                     mask=mask,
                     annot=True,
                     fmt='.2f',
                     cmap=custom_cmap,
                     center=0,
                     square=True,
                     linewidths=1,
                     linecolor='#333333',
                     cbar_kws={"shrink": .6, "label": "Correlación"},
                     annot_kws={'size': 16, 'weight': 'normal', 'family': 'monospace'})
    
    plt.title('Matriz de Correlaciones - Estilo Minimalista\nAnálisis de Ventas por Producto', 
              fontsize=14, fontweight='300', pad=20, color='#333333')
    plt.xticks(rotation=0, ha='center', fontsize=11, color='#444444')
    plt.yticks(rotation=0, fontsize=11, color='#444444')
    
    # Remover bordes
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    
    p3 = FIG_DIR / 'heatmap_minimalista.png'
    plt.savefig(p3, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Heatmap minimalista guardado: {p3}")
    
    # 4. HEATMAP PROFESIONAL - Estilo Corporativo
    plt.figure(figsize=(12, 8))
    
    # Paleta corporativa en azules y grises
    corporate_colors = ['#1f2937', '#374151', '#6b7280', '#9ca3af', '#d1d5db', '#f3f4f6']
    corporate_cmap = sns.blend_palette(corporate_colors, as_cmap=True)
    
    sns.heatmap(corr_renamed, 
                annot=True,
                fmt='.3f',
                cmap='Blues',
                center=None,
                square=False,
                linewidths=2,
                linecolor='white',
                cbar_kws={"shrink": .8, "label": "Coeficiente de Correlación"},
                annot_kws={'size': 13, 'weight': 'bold', 'color': 'black'})
    
    plt.title('ANÁLISIS DE CORRELACIONES - REPORTE EJECUTIVO\nDatos de Ventas y Rendimiento', 
              fontsize=16, fontweight='bold', pad=25, color='#1f2937')
    plt.xticks(rotation=0, ha='center', fontsize=12, fontweight='600', color='#374151')
    plt.yticks(rotation=0, fontsize=12, fontweight='600', color='#374151')
    
    # Agregar estadísticas en el gráfico
    stats_text = f"N = {len(df_corr)} observaciones\nPeríodo: Análisis Completo"
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, color='#6b7280', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='#f9fafb', alpha=0.8))
    
    plt.tight_layout()
    
    p4 = FIG_DIR / 'heatmap_profesional.png'
    plt.savefig(p4, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Heatmap profesional guardado: {p4}")
    
    # 5. HEATMAP ARTÍSTICO - Gradientes Suaves
    plt.figure(figsize=(11, 9))
    
    # Crear gradiente personalizado
    from matplotlib.colors import LinearSegmentedColormap
    colors_artistic = ['#0d1b2a', '#415a77', '#778da9', '#e0e1dd', '#ffd23f', '#ff6b6b']
    artistic_cmap = LinearSegmentedColormap.from_list("artistic", colors_artistic)
    
    mask = np.triu(np.ones_like(corr_renamed, dtype=bool))
    
    sns.heatmap(corr_renamed, 
                mask=mask,
                annot=True,
                fmt='.2f',
                cmap=artistic_cmap,
                center=0,
                square=True,
                linewidths=0.5,
                linecolor='#2d3436',
                cbar_kws={"shrink": .7, "label": "Fuerza de Correlación", "aspect": 30},
                annot_kws={'size': 14, 'weight': 'bold', 'color': 'white'})
    
    plt.title('🎨 Matriz de Correlaciones - Diseño Artístico 🎨\n"Visualización de Patrones de Venta"', 
              fontsize=16, fontweight='bold', pad=30, color='#2d3436', style='italic')
    plt.xticks(rotation=20, ha='right', fontsize=12, fontweight='600', color='#636e72')
    plt.yticks(rotation=0, fontsize=12, fontweight='600', color='#636e72')
    
    # Agregar sombra al título
    plt.figtext(0.5, 0.95, '🎨 Matriz de Correlaciones - Diseño Artístico 🎨\n"Visualización de Patrones de Venta"', 
                ha='center', fontsize=16, fontweight='bold', color='#ddd', alpha=0.3)
    
    plt.tight_layout()
    
    p5 = FIG_DIR / 'heatmap_artistico.png'
    plt.savefig(p5, dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
    plt.close()
    print(f"✓ Heatmap artístico guardado: {p5}")
    
    # 6. HEATMAP COMPARATIVO - Panel con múltiples paletas
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('COMPARATIVO DE ESTILOS - MATRIZ DE CORRELACIONES\nAnálisis de Ventas con Diferentes Paletas de Color', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    paletas = [
        ('RdYlBu_r', 'Clásico Divergente'),
        ('viridis', 'Viridis Moderno'),
        ('plasma', 'Plasma Energético'),
        ('coolwarm', 'Frío-Cálido'),
        ('Spectral_r', 'Espectral'),
        ('RdBu_r', 'Rojo-Azul')
    ]
    
    for idx, (cmap, title) in enumerate(paletas):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        sns.heatmap(corr_renamed, 
                    annot=True,
                    fmt='.2f',
                    cmap=cmap,
                    center=0,
                    square=True,
                    linewidths=1,
                    cbar=False,
                    ax=ax,
                    annot_kws={'size': 10, 'weight': 'bold'})
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('')
        ax.set_ylabel('')
        if col == 0:
            ax.set_ylabel('Variables', fontsize=12, fontweight='600')
        if row == 1:
            ax.set_xlabel('Variables', fontsize=12, fontweight='600')
    
    plt.tight_layout()
    
    p6 = FIG_DIR / 'heatmap_comparativo.png'
    plt.savefig(p6, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Panel comparativo guardado: {p6}")
    
    # Crear resumen de correlaciones
    print("\n" + "="*60)
    print("📊 RESUMEN DE CORRELACIONES ENCONTRADAS")
    print("="*60)
    
    correlations = []
    for i in range(len(corr_renamed.index)):
        for j in range(i+1, len(corr_renamed.columns)):
            var1 = corr_renamed.index[i]
            var2 = corr_renamed.columns[j]
            corr_val = corr_renamed.iloc[i, j]
            correlations.append((var1, var2, corr_val))
    
    # Ordenar por valor absoluto de correlación
    correlations.sort(key=lambda x: abs(x[2]), reverse=True)
    
    for var1, var2, corr_val in correlations:
        if abs(corr_val) > 0.7:
            strength = "🔴 MUY FUERTE"
        elif abs(corr_val) > 0.5:
            strength = "🟡 FUERTE"
        elif abs(corr_val) > 0.3:
            strength = "🟢 MODERADA"
        else:
            strength = "⚪ DÉBIL"
        
        direction = "POSITIVA" if corr_val > 0 else "NEGATIVA"
        
        print(f"{var1} ↔ {var2}")
        print(f"   Correlación: {corr_val:.3f} ({strength} {direction})")
        print()
    
    print(f"✅ Se generaron 6 estilos diferentes de heatmaps en la carpeta: {FIG_DIR}")
    print("\nArchivos generados:")
    print("- heatmap_clasico.png (Estilo tradicional)")
    print("- heatmap_vibrante.png (Colores energéticos)")
    print("- heatmap_minimalista.png (Diseño limpio)")
    print("- heatmap_profesional.png (Estilo corporativo)")
    print("- heatmap_artistico.png (Gradientes creativos)")
    print("- heatmap_comparativo.png (Panel de comparación)")
    

if __name__ == '__main__':
    create_custom_heatmaps()