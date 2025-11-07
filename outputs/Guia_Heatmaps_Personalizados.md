# 🎨 GUÍA DE HEATMAPS PERSONALIZADOS - ANÁLISIS DE CORRELACIONES

## 📊 Resumen de Datos Analizados
- **Total de transacciones**: 343
- **Variables analizadas**: Precio Unitario, Cantidad, Total Venta
- **Fuente**: detalle_ventas.xlsx

---

## 🎯 CORRELACIONES PRINCIPALES ENCONTRADAS

### 🔴 **Precio Unitario ↔ Total Venta**
- **Correlación: 0.679** (FUERTE POSITIVA)
- **Interpretación**: A mayor precio unitario, mayor es el total de la venta
- **Impacto**: El precio es el principal driver de los ingresos

### 🟡 **Cantidad ↔ Total Venta**
- **Correlación: 0.600** (FUERTE POSITIVA)  
- **Interpretación**: A mayor cantidad vendida, mayor el total
- **Impacto**: El volumen también influye significativamente en las ventas

### ⚪ **Precio Unitario ↔ Cantidad**
- **Correlación: -0.074** (DÉBIL NEGATIVA)
- **Interpretación**: No hay relación clara entre precio y cantidad
- **Impacto**: Los productos caros no necesariamente se venden menos

---

## 🎨 ESTILOS DE HEATMAPS GENERADOS

### 1. **📊 HEATMAP CLÁSICO** (`heatmap_clasico.png`)
- **Colores**: Azul y Rojo tradicional (RdBu_r)
- **Características**: 
  - Máscara triangular superior
  - Formato conservador y profesional
  - Fácil de interpretar
- **Uso recomendado**: Presentaciones académicas o reportes formales

### 2. **🔥 HEATMAP VIBRANTE** (`heatmap_vibrante.png`)
- **Colores**: Paleta Plasma (morados, magentas, amarillos)
- **Características**:
  - Colores intensos y llamativos
  - Matriz completa visible
  - Texto en blanco sobre fondo colorido
- **Uso recomendado**: Presentaciones dinámicas o dashboards modernos

### 3. **✨ HEATMAP MINIMALISTA** (`heatmap_minimalista.png`)
- **Colores**: Paleta personalizada suave
- **Características**:
  - Diseño limpio y elegante
  - Sin bordes, estilo moderno
  - Tipografía monospace
- **Uso recomendado**: Interfaces de usuario, aplicaciones web

### 4. **💼 HEATMAP PROFESIONAL** (`heatmap_profesional.png`)
- **Colores**: Escala de azules corporativos
- **Características**:
  - Estilo ejecutivo y empresarial
  - Estadísticas adicionales en el gráfico
  - Formato de reporte
- **Uso recomendado**: Presentaciones a ejecutivos, reportes de negocio

### 5. **🎨 HEATMAP ARTÍSTICO** (`heatmap_artistico.png`)
- **Colores**: Gradiente personalizado (azules a amarillos)
- **Características**:
  - Diseño creativo con efectos visuales
  - Gradientes suaves y atractivos
  - Estilo más creativo y moderno
- **Uso recomendado**: Presentaciones creativas, material de marketing

### 6. **📋 PANEL COMPARATIVO** (`heatmap_comparativo.png`)
- **Múltiples estilos en uno**: 6 variaciones lado a lado
- **Características**:
  - Comparación visual de diferentes paletas
  - Ideal para elegir el estilo preferido
  - Vista general de todas las opciones
- **Uso recomendado**: Selección de estilo, documentación completa

---

## 📈 INTERPRETACIÓN ESTRATÉGICA

### 💡 **Insights Clave**:

1. **Estrategia de Precios**: 
   - El precio unitario tiene mayor impacto (67.9%) que la cantidad (60%) en el total de ventas
   - Enfocarse en optimización de precios puede ser más efectivo que aumentar volúmenes

2. **Flexibilidad de Demanda**:
   - La correlación casi nula (-0.074) entre precio y cantidad sugiere que la demanda no es muy sensible al precio
   - Los productos premium pueden mantenerse sin afectar significativamente las cantidades

3. **Oportunidades de Crecimiento**:
   - Ambas variables (precio y cantidad) contribuyen positivamente
   - Estrategias que combinen optimización de precios Y aumento de volumen pueden maximizar resultados

---

## 🔧 **Archivos Disponibles**

Todos los heatmaps están guardados en: `outputs/figures/`

```
📁 figures/
├── 📊 heatmap_clasico.png      (Estilo tradicional)
├── 🔥 heatmap_vibrante.png     (Colores energéticos)  
├── ✨ heatmap_minimalista.png   (Diseño limpio)
├── 💼 heatmap_profesional.png   (Estilo corporativo)
├── 🎨 heatmap_artistico.png     (Gradientes creativos)
└── 📋 heatmap_comparativo.png   (Panel de comparación)
```

---

## 🚀 **Próximos Pasos Sugeridos**

1. **Elegir el estilo** que mejor se adapte a tu audiencia
2. **Profundizar el análisis** por segmentos de productos
3. **Analizar tendencias temporales** si tienes datos de fechas
4. **Crear dashboards interactivos** usando estos insights

---

*Análisis generado el: 6 de Noviembre, 2025*
*Datos fuente: detalle_ventas.xlsx (343 transacciones)*