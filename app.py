"""
Sistema RAG de Optimización de Inventario para Operaciones
============================================================
Herramienta de apoyo para el equipo de operaciones de restaurantes,
cafeterías y servicios de alimentos. Optimiza el uso del inventario
y reduce mermas sugiriendo recetas basadas en ingredientes disponibles.

Flujo: Retrieval (FAISS + Voyage) → Reranking (Cohere) → LLM (Gemini)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import faiss
import pandas as pd
import pickle
import zlib
import numpy as np
import os
import requests
from dotenv import load_dotenv
import voyageai
import cohere
import google.generativeai as genai

load_dotenv()

# ============================================================
# CONFIGURACIÓN
# ============================================================
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Inicializar clientes
voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
cohere_client = cohere.Client(api_key=COHERE_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-pro")

# ============================================================
# CARGAR DATOS
# ============================================================
print("Cargando datos...")

# Cargar índice FAISS
index = faiss.read_index("FAISS_recetas/spoonacular_medium.index")
print(f"✓ FAISS: {index.ntotal} vectores, dim={index.d}")

# Cargar recetas
df_recetas = pd.read_parquet("FAISS_recetas/spoonacular_medium.parquet")
print(f"✓ Recetas: {len(df_recetas)} registros")

# Cargar metadata (comprimido con zlib)
try:
    with open("FAISS_recetas/spoonacular_medium_metadata.pkl", "rb") as f:
        compressed = f.read()
        decompressed = zlib.decompress(compressed)
        metadata = pickle.loads(decompressed)
    print("✓ Metadata cargado")
except Exception as e:
    print(f"⚠ Metadata no disponible: {e}")
    metadata = None

print("=" * 50)

# ============================================================
# CIUDADES PROFECO QQP
# ============================================================
CIUDADES_QQP = {
    "Acapulco": "1201",
    "Aguascalientes": "0101",
    "Apizaco": "2902",
    "Campeche": "0401",
    "Cancún": "2301",
    "Chihuahua": "0801",
    "Ciudad Juárez": "0802",
    "Ciudad de México": "0901",
    "Cuernavaca": "1701",
    "Culiacán": "2501",
    "Durango": "1001",
    "Estado de México": "1502",
    "Guadalajara": "1401",
    "Hermosillo": "2601",
    "La Paz": "0301",
    "León": "1101",
    "Monterrey": "1901",
    "Morelia": "1601",
    "Mérida": "3101",
    "Oaxaca": "2001",
    "Orizaba": "3004",
    "Pachuca": "1301",
    "Playa del Carmen": "2303",
    "Puebla": "2101",
    "Querétaro": "2201",
    "Saltillo": "0501",
    "San Luis Potosí": "2401",
    "Tampico": "2804",
    "Tijuana": "0201",
    "Tlaxcala": "2901",
    "Tuxtla Gutiérrez": "0701",
    "Veracruz": "3001",
    "Villahermosa": "2701",
    "Zacatecas": "3201",
}

# Mapeo de ingredientes inglés → español para búsqueda en PROFECO
INGREDIENTES_EN_ES = {
    "chicken": "pollo",
    "rice": "arroz",
    "tomato": "tomate",
    "onion": "cebolla",
    "garlic": "ajo",
    "potato": "papa",
    "carrot": "zanahoria",
    "beef": "carne de res",
    "pork": "cerdo",
    "fish": "pescado",
    "egg": "huevo",
    "eggs": "huevo",
    "milk": "leche",
    "cheese": "queso",
    "butter": "mantequilla",
    "oil": "aceite",
    "olive oil": "aceite de oliva",
    "salt": "sal",
    "pepper": "pimienta",
    "sugar": "azúcar",
    "flour": "harina",
    "bread": "pan",
    "lettuce": "lechuga",
    "cucumber": "pepino",
    "avocado": "aguacate",
    "lemon": "limón",
    "lime": "limón",
    "orange": "naranja",
    "apple": "manzana",
    "banana": "plátano",
    "beans": "frijol",
    "corn": "maíz",
    "cream": "crema",
    "yogurt": "yogurt",
    "pasta": "pasta",
    "noodles": "fideos",
    "sausage": "salchicha",
    "ham": "jamón",
    "bacon": "tocino",
    "shrimp": "camarón",
    "tuna": "atún",
    "salmon": "salmón",
    "spinach": "espinaca",
    "broccoli": "brócoli",
    "bell pepper": "pimiento",
    "mushroom": "champiñón",
    "mushrooms": "champiñón",
    "celery": "apio",
    "parsley": "perejil",
    "cilantro": "cilantro",
    "ginger": "jengibre",
    "honey": "miel",
    "vinegar": "vinagre",
    "soy sauce": "salsa de soya",
    "mayonnaise": "mayonesa",
    "ketchup": "catsup",
    "mustard": "mostaza",
}

# ============================================================
# MODELOS PYDANTIC
# ============================================================
class InventarioRequest(BaseModel):
    """Solicitud de consulta de recetas basada en inventario disponible"""
    ingredientes: list[str] = Field(
        ..., 
        description="Lista de ingredientes disponibles en inventario (en inglés para mejor precisión del sistema)",
        example=["chicken", "rice", "tomato", "onion", "garlic"]
    )
    max_resultados: int = Field(
        default=5, 
        ge=1, 
        le=10,
        description="Número máximo de recetas a retornar"
    )
    tiempo_max_minutos: Optional[int] = Field(
        default=None,
        description="Filtrar recetas que se preparen en menos de X minutos"
    )
    presupuesto_max: Optional[float] = Field(
        default=None,
        description="Filtrar por precio máximo por porción"
    )
    generar_sugerencias: bool = Field(
        default=True,
        description="Si True, genera sugerencias con IA"
    )

class RecetaResponse(BaseModel):
    """Una receta individual con scores de optimización"""
    id: str
    titulo: str
    ingredientes: str
    instrucciones: str
    link: str
    tags: str
    precio_por_porcion: float
    tiempo_minutos: int
    porciones: int
    health_score: float
    score_relevancia: float = Field(description="Score de relevancia del reranker")
    ingredientes_coincidentes: list[str] = Field(description="Ingredientes del usuario usados")
    # Nuevos scores para optimización dual
    score_popularidad: float = Field(default=0.0, description="Score de popularidad/satisfacción cliente (0-100)")
    score_costo_efectividad: float = Field(default=0.0, description="Score de costo-efectividad (0-100)")
    score_combinado: float = Field(default=0.0, description="Score balanceado popularidad + costo")
    recomendacion: str = Field(default="", description="Tipo de recomendación: 'MEJOR_COSTO', 'MAS_POPULAR', 'EQUILIBRADA'")

class SugerenciasResponse(BaseModel):
    """Respuesta del sistema de optimización de inventario"""
    query_ingredientes: list[str]
    recetas: list[RecetaResponse]
    receta_mejor_costo: Optional[str] = Field(default=None, description="ID de la receta más costo-efectiva")
    receta_mas_popular: Optional[str] = Field(default=None, description="ID de la receta más popular")
    receta_equilibrada: Optional[str] = Field(default=None, description="ID de la receta con mejor balance")
    analisis_operativo: Optional[str] = Field(default=None, description="Análisis y recomendaciones para el equipo de operaciones")
    total_encontradas: int
    mensaje: str


class PrecioIngrediente(BaseModel):
    """Precio de un ingrediente encontrado en PROFECO"""
    ingrediente_original: str = Field(description="Ingrediente en inglés")
    ingrediente_busqueda: str = Field(description="Término de búsqueda en español")
    producto: str = Field(description="Nombre del producto encontrado")
    precio: float = Field(description="Precio en MXN")
    cadena_comercial: str
    establecimiento: str
    direccion: str
    colonia: str
    municipio: str
    fecha_observacion: str


class InventarioConPreciosRequest(BaseModel):
    """Request para buscar recetas con precios de PROFECO"""
    ingredientes: list[str] = Field(
        ...,
        description="Lista de ingredientes (en inglés)",
        example=["chicken", "rice", "tomato", "onion"]
    )
    ciudad: str = Field(
        default="Ciudad de México",
        description="Ciudad para buscar precios en PROFECO"
    )
    max_resultados: int = Field(default=3, ge=1, le=5)
    generar_sugerencias: bool = Field(default=True)


class RecetaConPreciosResponse(BaseModel):
    """Respuesta con receta y precios de ingredientes"""
    receta: RecetaResponse
    precios_ingredientes: list[PrecioIngrediente]
    costo_estimado_total: float
    ingredientes_sin_precio: list[str]


class BusquedaConPreciosResponse(BaseModel):
    """Respuesta completa con recetas y precios"""
    ciudad: str
    query_ingredientes: list[str]
    recetas_con_precios: list[RecetaConPreciosResponse]
    analisis_operativo: Optional[str] = None
    total_encontradas: int
    mensaje: str


# ============================================================
# FUNCIONES CORE
# ============================================================

def obtener_precio_profeco(ciudad: str, busqueda: str) -> pd.DataFrame:
    """
    Consulta la API de PROFECO QQP y devuelve un DataFrame
    con los precios encontrados para un producto en una ciudad.
    """
    if ciudad not in CIUDADES_QQP:
        return pd.DataFrame()
    
    clave_ciudad = CIUDADES_QQP[ciudad]
    BASE_URL = "https://qqp.profeco.gob.mx/api/precios"
    
    params = {
        "clave_ciudad": clave_ciudad,
        "busqueda": busqueda
    }
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        if response.status_code != 200:
            return pd.DataFrame()
        
        data = response.json()
        if not data.get("success", False):
            return pd.DataFrame()
        
        productos = data["data"].get("productos", [])
        if len(productos) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(productos)
        if "fecha_observacion" in df.columns:
            df["fecha_observacion"] = pd.to_datetime(df["fecha_observacion"], errors="coerce")
        
        # Ordenar por precio (más barato primero)
        df = df.sort_values(["precio", "fecha_observacion"], ascending=[True, False])
        return df
    
    except Exception as e:
        print(f"Error consultando PROFECO para '{busqueda}': {e}")
        return pd.DataFrame()


def traducir_ingrediente(ingrediente_en: str) -> str:
    """Traduce ingrediente de inglés a español para buscar en PROFECO"""
    ingrediente_lower = ingrediente_en.lower().strip()
    return INGREDIENTES_EN_ES.get(ingrediente_lower, ingrediente_lower)


def obtener_mejor_precio(ciudad: str, ingrediente_en: str) -> Optional[dict]:
    """
    Busca el mejor precio de un ingrediente en PROFECO.
    Retorna dict con info del mejor precio o None si no encuentra.
    """
    ingrediente_es = traducir_ingrediente(ingrediente_en)
    df = obtener_precio_profeco(ciudad, ingrediente_es)
    
    if df.empty:
        return None
    
    # Tomar el más barato
    mejor = df.iloc[0]
    
    return {
        "ingrediente_original": ingrediente_en,
        "ingrediente_busqueda": ingrediente_es,
        "producto": mejor.get("producto", ""),
        "precio": float(mejor.get("precio", 0)),
        "cadena_comercial": mejor.get("cadena_comercial", ""),
        "establecimiento": mejor.get("establecimiento", ""),
        "direccion": mejor.get("direccion", ""),
        "colonia": mejor.get("colonia", ""),
        "municipio": mejor.get("municipio", ""),
        "fecha_observacion": str(mejor.get("fecha_observacion", ""))
    }


def crear_embedding(texto: str) -> np.ndarray:
    """Genera embedding usando Voyage AI (voyage-large-3)"""
    result = voyage_client.embed(
        texts=[texto],
        model="voyage-3-large",
        input_type="query"
    )
    return np.array(result.embeddings[0], dtype=np.float32)


def buscar_faiss(query_embedding: np.ndarray, top_k: int = 20) -> list[int]:
    """Busca los top_k vectores más similares en FAISS"""
    query_embedding = query_embedding.reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    return indices[0].tolist()


def rerank_con_cohere(query: str, documentos: list[str], top_n: int = 5) -> list[tuple[int, float]]:
    """
    Reranquea documentos usando Cohere Rerank.
    Retorna lista de (índice_original, score)
    """
    if not documentos:
        return []
    
    response = cohere_client.rerank(
        model="rerank-v3.5",
        query=query,
        documents=documentos,
        top_n=top_n,
        return_documents=False
    )
    
    return [(r.index, r.relevance_score) for r in response.results]


def encontrar_ingredientes_coincidentes(ingredientes_usuario: list[str], ingredientes_receta: str) -> list[str]:
    """Encuentra qué ingredientes del usuario están en la receta"""
    ingredientes_receta_lower = ingredientes_receta.lower()
    coincidentes = []
    for ing in ingredientes_usuario:
        if ing.lower() in ingredientes_receta_lower:
            coincidentes.append(ing)
    return coincidentes


# ============================================================
# TAGS POPULARES - Indicadores de satisfacción del cliente
# ============================================================
TAGS_POPULARIDAD = {
    # Tags muy populares (alto peso)
    "comfort food": 15, "family friendly": 12, "kid friendly": 12,
    "quick & easy": 10, "one pot": 10, "30 minutes or less": 10,
    "crowd pleaser": 15, "date night": 10, "romantic": 8,
    # Tags de tendencia
    "healthy": 8, "low carb": 7, "high protein": 7, "keto": 6,
    "vegetarian": 6, "vegan": 6, "gluten free": 5,
    # Tags de sabor
    "savory": 5, "spicy": 5, "sweet": 5, "umami": 6,
    # Tags regionales populares
    "mexican": 8, "italian": 8, "asian": 7, "mediterranean": 7,
    "american": 6, "indian": 7, "chinese": 7, "japanese": 7,
    # Tags de ocasión
    "dinner": 5, "lunch": 5, "main course": 5, "side dish": 4,
}


def calcular_score_popularidad(health_score: float, tags: str, tiempo_minutos: int) -> float:
    """
    Calcula un score de popularidad basado en:
    - Health score (clientes buscan opciones saludables)
    - Tags populares (indicadores de preferencias del cliente)
    - Tiempo de preparación (recetas rápidas son más populares)
    
    Retorna un score de 0-100
    """
    score = 0.0
    
    # 1. Health Score contribuye hasta 30 puntos
    # Health score va de 0-100, lo escalamos a 0-30
    score += (health_score / 100) * 30
    
    # 2. Tags populares contribuyen hasta 40 puntos
    tags_lower = tags.lower()
    puntos_tags = 0
    for tag, peso in TAGS_POPULARIDAD.items():
        if tag in tags_lower:
            puntos_tags += peso
    # Capear a 40 puntos máximo
    score += min(puntos_tags, 40)
    
    # 3. Tiempo de preparación contribuye hasta 30 puntos
    # Menos tiempo = más popular
    if tiempo_minutos <= 15:
        score += 30
    elif tiempo_minutos <= 30:
        score += 25
    elif tiempo_minutos <= 45:
        score += 20
    elif tiempo_minutos <= 60:
        score += 15
    elif tiempo_minutos <= 90:
        score += 10
    else:
        score += 5
    
    return min(round(score, 2), 100)


def calcular_score_costo_efectividad(
    precio_por_porcion: float, 
    porciones: int, 
    ingredientes_coincidentes: int,
    total_ingredientes_usuario: int,
    tiempo_minutos: int
) -> float:
    """
    Calcula un score de costo-efectividad basado en:
    - Precio por porción (menor = mejor)
    - Aprovechamiento de ingredientes del inventario
    - Eficiencia del tiempo de preparación
    
    Retorna un score de 0-100
    """
    score = 0.0
    
    # 1. Precio por porción contribuye hasta 35 puntos
    # Asumimos que $5 USD o menos es excelente, $15+ es caro
    if precio_por_porcion <= 2:
        score += 35
    elif precio_por_porcion <= 5:
        score += 30
    elif precio_por_porcion <= 8:
        score += 25
    elif precio_por_porcion <= 12:
        score += 20
    elif precio_por_porcion <= 15:
        score += 15
    else:
        score += 10
    
    # 2. Aprovechamiento de ingredientes contribuye hasta 40 puntos
    # Cuantos más ingredientes del usuario use, mejor
    if total_ingredientes_usuario > 0:
        ratio_aprovechamiento = ingredientes_coincidentes / total_ingredientes_usuario
        score += ratio_aprovechamiento * 40
    
    # 3. Porciones por receta contribuye hasta 15 puntos
    # Más porciones = mejor rendimiento
    if porciones >= 8:
        score += 15
    elif porciones >= 6:
        score += 12
    elif porciones >= 4:
        score += 10
    elif porciones >= 2:
        score += 7
    else:
        score += 5
    
    # 4. Tiempo eficiente contribuye hasta 10 puntos
    if tiempo_minutos <= 30:
        score += 10
    elif tiempo_minutos <= 60:
        score += 7
    else:
        score += 4
    
    return min(round(score, 2), 100)


def calcular_score_combinado(score_popularidad: float, score_costo: float, peso_costo: float = 0.5) -> float:
    """
    Calcula un score combinado balanceando popularidad y costo-efectividad.
    peso_costo: 0.5 = balance igual, 0.7 = priorizar costo, 0.3 = priorizar popularidad
    """
    peso_popularidad = 1 - peso_costo
    return round(score_popularidad * peso_popularidad + score_costo * peso_costo, 2)


def asignar_recomendacion(recetas: list) -> list:
    """
    Asigna etiquetas de recomendación a las recetas:
    - MEJOR_COSTO: La más costo-efectiva
    - MAS_POPULAR: La más popular entre clientes
    - EQUILIBRADA: Mejor balance de ambos
    """
    if not recetas:
        return recetas
    
    # Encontrar los mejores en cada categoría
    mejor_costo_idx = max(range(len(recetas)), key=lambda i: recetas[i].score_costo_efectividad)
    mas_popular_idx = max(range(len(recetas)), key=lambda i: recetas[i].score_popularidad)
    equilibrada_idx = max(range(len(recetas)), key=lambda i: recetas[i].score_combinado)
    
    # Asignar etiquetas
    for i, receta in enumerate(recetas):
        etiquetas = []
        if i == mejor_costo_idx:
            etiquetas.append("💰 MEJOR_COSTO")
        if i == mas_popular_idx:
            etiquetas.append("⭐ MAS_POPULAR")
        if i == equilibrada_idx and equilibrada_idx not in [mejor_costo_idx, mas_popular_idx]:
            etiquetas.append("⚖️ EQUILIBRADA")
        
        receta.recomendacion = " | ".join(etiquetas) if etiquetas else ""
    
    return recetas


def generar_sugerencias_llm(ingredientes: list[str], recetas: list[dict]) -> str:
    """Genera análisis operativo con Gemini considerando costo-efectividad Y satisfacción del cliente"""
    
    recetas_texto = "\n\n".join([
        f"""Receta #{i+1}: {r['titulo']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Ingredientes: {r['ingredientes'][:300]}...
- Tiempo: {r['tiempo_minutos']} min | Porciones: {r['porciones']}
- Costo/porción: ${r['precio_por_porcion']:.2f} USD
- Health Score: {r['health_score']}
- Del inventario usa: {', '.join(r['ingredientes_coincidentes']) if r['ingredientes_coincidentes'] else 'Ninguno'}
- 📊 SCORES DE OPTIMIZACIÓN:
  • Score Popularidad (satisfacción cliente): {r.get('score_popularidad', 0):.1f}/100
  • Score Costo-Efectividad (negocio): {r.get('score_costo_efectividad', 0):.1f}/100
  • Score Combinado: {r.get('score_combinado', 0):.1f}/100
  • Etiqueta: {r.get('recomendacion', 'N/A')}"""
        for i, r in enumerate(recetas[:5])
    ])
    
    # Identificar las mejores
    mejor_costo = max(recetas, key=lambda x: x.get('score_costo_efectividad', 0))
    mas_popular = max(recetas, key=lambda x: x.get('score_popularidad', 0))
    equilibrada = max(recetas, key=lambda x: x.get('score_combinado', 0))
    
    prompt = f"""Eres un asistente de gestión operativa para restaurantes y servicios de alimentos. 
Generas reportes que BALANCEAN la eficiencia del negocio CON la satisfacción del cliente.

═══════════════════════════════════════════════════════════════
                    DATOS DE ENTRADA
═══════════════════════════════════════════════════════════════

INVENTARIO ACTUAL: {', '.join(ingredientes)}

RECETAS IDENTIFICADAS CON SCORES:
{recetas_texto}

═══════════════════════════════════════════════════════════════
                    ANÁLISIS RÁPIDO
═══════════════════════════════════════════════════════════════

🏆 MEJOR COSTO-EFECTIVIDAD: {mejor_costo['titulo']} (Score: {mejor_costo.get('score_costo_efectividad', 0):.1f})
⭐ MÁS POPULAR/SATISFACCIÓN: {mas_popular['titulo']} (Score: {mas_popular.get('score_popularidad', 0):.1f})
⚖️ MEJOR EQUILIBRIO: {equilibrada['titulo']} (Score: {equilibrada.get('score_combinado', 0):.1f})

═══════════════════════════════════════════════════════════════

GENERA UN REPORTE OPERATIVO CON LA SIGUIENTE ESTRUCTURA:

## 📋 RESUMEN EJECUTIVO
[Análisis de 2-3 líneas sobre el balance entre costos y satisfacción del cliente]

## 💰 OPCIÓN COSTO-EFECTIVA (Para maximizar margen)
- **Receta:** [Nombre en español]
- **Por qué es económica:** [Explicación]
- **Score Costo-Efectividad:** X/100
- **Ingredientes del inventario:** [Lista en español]
- **Ahorro estimado:** [Comparación]

## ⭐ OPCIÓN POPULAR (Para satisfacer clientes)
- **Receta:** [Nombre en español]
- **Por qué gusta a los clientes:** [Explicación basada en tags, health score, etc.]
- **Score Popularidad:** X/100
- **Público objetivo:** [Tipo de cliente]

## ⚖️ RECOMENDACIÓN EQUILIBRADA (Mejor de ambos mundos)
- **Receta:** [Nombre en español]
- **Balance logrado:** [Explicación de por qué es el mejor compromiso]
- **Score Combinado:** X/100

## 🎯 DECISIÓN SUGERIDA
[Indica cuál receta elegir según el contexto: 
- Si es día de alta demanda → opción popular
- Si el margen está apretado → opción económica
- Para operación normal → opción equilibrada]

## 📊 ANÁLISIS DE APROVECHAMIENTO DEL INVENTARIO
- Porcentaje de inventario utilizado: X%
- Ingredientes prioritarios a usar (por caducidad): [Lista]

## ⚠️ RECOMENDACIONES OPERATIVAS
[2-3 puntos que balanceen costos con calidad de servicio]

REGLAS OBLIGATORIAS:
1. TODO en ESPAÑOL (traduce nombres de recetas e ingredientes)
2. Tono formal y profesional
3. Equilibra SIEMPRE negocio + cliente
4. Sé conciso pero completo"""

    response = gemini_model.generate_content(prompt)
    return response.text


def aplicar_filtros(df: pd.DataFrame, tiempo_max: int = None, presupuesto_max: float = None) -> pd.DataFrame:
    """Aplica filtros opcionales al DataFrame de recetas"""
    resultado = df.copy()
    
    if tiempo_max is not None:
        resultado = resultado[resultado['readyInMinutes'] <= tiempo_max]
    
    if presupuesto_max is not None:
        resultado = resultado[resultado['pricePerServing'] <= presupuesto_max]
    
    return resultado


# ============================================================
# API FASTAPI
# ============================================================
app = FastAPI(
    title="📊 Sistema de Optimización de Inventario - Operaciones",
    description="API para el equipo de operaciones. Optimiza el uso del inventario mediante sugerencias de recetas basadas en ingredientes disponibles. Reduce mermas y mejora la eficiencia operativa.",
    version="1.0.0"
)

# CORS para frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """Health check y bienvenida"""
    return {
        "mensaje": "📊 Sistema de Optimización de Inventario - Módulo de Operaciones",
        "version": "1.0.0",
        "descripcion": "Herramienta para optimizar el uso del inventario y reducir mermas en operaciones de alimentos",
        "endpoints": {
            "POST /buscar": "Consultar recetas óptimas según inventario disponible",
            "POST /buscar-con-precios": "🆕 Recetas + precios en tiempo real de PROFECO",
            "POST /buscar/html": "Igual que /buscar pero con HTML formateado",
            "GET /recetas": "Catálogo completo de recetas",
            "GET /recetas/{id}": "Detalle de receta específica",
            "GET /ciudades": "Ciudades disponibles para consulta de precios",
            "GET /precio/{ciudad}/{ingrediente}": "Consultar precio de un ingrediente",
            "GET /stats": "Estadísticas del sistema"
        },
        "total_recetas_disponibles": len(df_recetas),
        "ciudades_profeco": len(CIUDADES_QQP)
    }


@app.post("/buscar", response_model=SugerenciasResponse)
def buscar_recetas(request: InventarioRequest):
    """
    🔍 Endpoint principal: Busca recetas óptimas para los ingredientes disponibles.
    
    Flujo:
    1. Crea embedding del query con Voyage
    2. Busca en FAISS (retrieval)
    3. Reranquea con Cohere
    4. Genera sugerencias con Gemini (opcional)
    """
    try:
        # 1. Crear query text
        query_text = f"Recetas con: {', '.join(request.ingredientes)}"
        
        # 2. Generar embedding con Voyage
        query_embedding = crear_embedding(query_text)
        
        # 3. Buscar en FAISS (retrieval) - traemos más para luego filtrar
        top_k_inicial = min(30, index.ntotal)
        indices_faiss = buscar_faiss(query_embedding, top_k_inicial)
        
        # 4. Obtener recetas candidatas
        candidatas = df_recetas.iloc[indices_faiss].copy()
        
        # 5. Aplicar filtros opcionales
        candidatas = aplicar_filtros(
            candidatas, 
            tiempo_max=request.tiempo_max_minutos,
            presupuesto_max=request.presupuesto_max
        )
        
        if len(candidatas) == 0:
            return SugerenciasResponse(
                query_ingredientes=request.ingredientes,
                recetas=[],
                analisis_operativo=None,
                total_encontradas=0,
                mensaje="No se encontraron recetas con los filtros aplicados"
            )
        
        # 6. Preparar documentos para reranking
        documentos_rerank = []
        for _, row in candidatas.iterrows():
            doc = f"{row['title']}. Ingredientes: {row['ingredients']}. Tags: {row['tags']}"
            documentos_rerank.append(doc)
        
        # 7. Reranking con Cohere
        rerank_results = rerank_con_cohere(
            query=query_text,
            documentos=documentos_rerank,
            top_n=request.max_resultados
        )
        
        # 8. Construir respuesta con scores de optimización
        recetas_response = []
        total_ingredientes = len(request.ingredientes)
        
        for idx_local, score in rerank_results:
            row = candidatas.iloc[idx_local]
            coincidentes = encontrar_ingredientes_coincidentes(
                request.ingredientes, 
                row['ingredients']
            )
            
            # Calcular scores de optimización
            score_pop = calcular_score_popularidad(
                health_score=row['healthScore'],
                tags=row['tags'],
                tiempo_minutos=row['readyInMinutes']
            )
            
            score_costo = calcular_score_costo_efectividad(
                precio_por_porcion=row['pricePerServing'],
                porciones=row['servings'],
                ingredientes_coincidentes=len(coincidentes),
                total_ingredientes_usuario=total_ingredientes,
                tiempo_minutos=row['readyInMinutes']
            )
            
            score_comb = calcular_score_combinado(score_pop, score_costo)
            
            receta = RecetaResponse(
                id=row['id'],
                titulo=row['title'],
                ingredientes=row['ingredients'],
                instrucciones=row['instructions'][:500] + "..." if len(row['instructions']) > 500 else row['instructions'],
                link=row['link'],
                tags=row['tags'],
                precio_por_porcion=row['pricePerServing'],
                tiempo_minutos=row['readyInMinutes'],
                porciones=row['servings'],
                health_score=row['healthScore'],
                score_relevancia=round(score, 4),
                ingredientes_coincidentes=coincidentes,
                score_popularidad=score_pop,
                score_costo_efectividad=score_costo,
                score_combinado=score_comb
            )
            recetas_response.append(receta)
        
        # 9. Asignar etiquetas de recomendación
        recetas_response = asignar_recomendacion(recetas_response)
        
        # Identificar las mejores en cada categoría
        receta_mejor_costo = max(recetas_response, key=lambda r: r.score_costo_efectividad).id if recetas_response else None
        receta_mas_popular = max(recetas_response, key=lambda r: r.score_popularidad).id if recetas_response else None
        receta_equilibrada = max(recetas_response, key=lambda r: r.score_combinado).id if recetas_response else None
        
        # 10. Generar sugerencias con LLM (opcional)
        sugerencias = None
        if request.generar_sugerencias and recetas_response:
            try:
                recetas_dict = [r.model_dump() for r in recetas_response]
                sugerencias = generar_sugerencias_llm(request.ingredientes, recetas_dict)
            except Exception as e:
                sugerencias = f"(Error generando sugerencias: {str(e)})"
        
        return SugerenciasResponse(
            query_ingredientes=request.ingredientes,
            recetas=recetas_response,
            receta_mejor_costo=receta_mejor_costo,
            receta_mas_popular=receta_mas_popular,
            receta_equilibrada=receta_equilibrada,
            analisis_operativo=sugerencias,
            total_encontradas=len(recetas_response),
            mensaje="Consulta procesada exitosamente"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recetas")
def listar_recetas(limit: int = 10, offset: int = 0):
    """Lista todas las recetas con paginación"""
    total = len(df_recetas)
    recetas = df_recetas.iloc[offset:offset+limit].to_dict(orient='records')
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "recetas": recetas
    }


@app.get("/recetas/{receta_id}")
def obtener_receta(receta_id: str):
    """Obtiene una receta específica por ID"""
    receta = df_recetas[df_recetas['id'] == receta_id]
    if len(receta) == 0:
        raise HTTPException(status_code=404, detail="Receta no encontrada")
    return receta.iloc[0].to_dict()


@app.get("/stats")
def estadisticas():
    """Estadísticas del sistema"""
    return {
        "total_recetas": len(df_recetas),
        "dimensiones_embedding": index.d,
        "modelo_embedding": "voyage-3-large",
        "modelo_rerank": "cohere-rerank-v3.5",
        "modelo_llm": "gemini-2.5-pro",
        "precio_promedio": round(df_recetas['pricePerServing'].mean(), 2),
        "tiempo_promedio_min": round(df_recetas['readyInMinutes'].mean(), 1),
        "health_score_promedio": round(df_recetas['healthScore'].mean(), 1),
        "ciudades_profeco": list(CIUDADES_QQP.keys())
    }


# ============================================================
# ENDPOINT CON PRECIOS PROFECO
# ============================================================

@app.post("/buscar-con-precios", response_model=BusquedaConPreciosResponse)
def buscar_recetas_con_precios(request: InventarioConPreciosRequest):
    """
    🛒 Busca recetas óptimas Y obtiene precios en tiempo real de PROFECO.
    
    Flujo:
    1. Busca recetas con RAG + Reranking
    2. Para cada ingrediente de la receta ganadora, consulta PROFECO
    3. Devuelve receta + mejores precios + ubicación de tiendas
    """
    try:
        # Validar ciudad
        if request.ciudad not in CIUDADES_QQP:
            raise HTTPException(
                status_code=400, 
                detail=f"Ciudad no válida. Ciudades disponibles: {list(CIUDADES_QQP.keys())}"
            )
        
        # 1. Buscar recetas (reutilizar lógica existente)
        inv_request = InventarioRequest(
            ingredientes=request.ingredientes,
            max_resultados=request.max_resultados,
            generar_sugerencias=False  # Lo haremos con más contexto después
        )
        resultado_recetas = buscar_recetas(inv_request)
        
        if resultado_recetas.total_encontradas == 0:
            return BusquedaConPreciosResponse(
                ciudad=request.ciudad,
                query_ingredientes=request.ingredientes,
                recetas_con_precios=[],
                analisis_operativo=None,
                total_encontradas=0,
                mensaje="No se encontraron recetas"
            )
        
        # 2. Para cada receta, buscar precios de ingredientes coincidentes
        recetas_con_precios = []
        
        for receta in resultado_recetas.recetas[:request.max_resultados]:
            precios_encontrados = []
            ingredientes_sin_precio = []
            costo_total = 0.0
            
            # Buscar precio para cada ingrediente coincidente
            for ing in receta.ingredientes_coincidentes:
                precio_info = obtener_mejor_precio(request.ciudad, ing)
                if precio_info:
                    precios_encontrados.append(PrecioIngrediente(**precio_info))
                    costo_total += precio_info["precio"]
                else:
                    ingredientes_sin_precio.append(ing)
            
            recetas_con_precios.append(RecetaConPreciosResponse(
                receta=receta,
                precios_ingredientes=precios_encontrados,
                costo_estimado_total=round(costo_total, 2),
                ingredientes_sin_precio=ingredientes_sin_precio
            ))
        
        # 3. Generar análisis operativo con contexto de precios
        analisis = None
        if request.generar_sugerencias and recetas_con_precios:
            try:
                analisis = generar_analisis_con_precios(
                    request.ingredientes,
                    request.ciudad,
                    recetas_con_precios
                )
            except Exception as e:
                analisis = f"(Error generando análisis: {str(e)})"
        
        return BusquedaConPreciosResponse(
            ciudad=request.ciudad,
            query_ingredientes=request.ingredientes,
            recetas_con_precios=recetas_con_precios,
            analisis_operativo=analisis,
            total_encontradas=len(recetas_con_precios),
            mensaje="Consulta con precios procesada exitosamente"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def generar_analisis_con_precios(ingredientes: list[str], ciudad: str, recetas_con_precios: list) -> str:
    """Genera análisis operativo completo incluyendo recetas, precios y ubicaciones"""
    
    # Construir información detallada de cada receta con precios
    recetas_detalle = ""
    for i, rcp in enumerate(recetas_con_precios):
        receta = rcp.receta
        recetas_detalle += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### 🍽️ RECETA {i+1}: {receta.titulo}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- **Score de relevancia:** {receta.score_relevancia:.2f}
- **Tiempo de preparación:** {receta.tiempo_minutos} minutos
- **Porciones:** {receta.porciones}
- **Precio por porción (referencia):** ${receta.precio_por_porcion:.2f} USD
- **Health Score:** {receta.health_score}
- **Ingredientes del inventario utilizados:** {', '.join(receta.ingredientes_coincidentes) if receta.ingredientes_coincidentes else 'Ninguno'}

**INGREDIENTES COMPLETOS:**
{receta.ingredientes}

**INSTRUCCIONES:**
{receta.instrucciones}

**LINK ORIGINAL:** {receta.link}

"""
        # Agregar información de precios PROFECO
        if rcp.precios_ingredientes:
            recetas_detalle += f"**💰 PRECIOS EN TIEMPO REAL ({ciudad}) - PROFECO:**\n"
            for p in rcp.precios_ingredientes:
                recetas_detalle += f"""
- **{p.ingrediente_busqueda.upper()}:** ${p.precio:.2f} MXN
  - Producto: {p.producto}
  - Tienda: {p.cadena_comercial} - {p.establecimiento}
  - Dirección: {p.direccion}, Col. {p.colonia}, {p.municipio}
  - Fecha: {p.fecha_observacion}
"""
            recetas_detalle += f"\n**COSTO TOTAL ESTIMADO:** ${rcp.costo_estimado_total:.2f} MXN\n"
        
        if rcp.ingredientes_sin_precio:
            recetas_detalle += f"\n⚠️ **Sin precio disponible:** {', '.join(rcp.ingredientes_sin_precio)}\n"
    
    prompt = f"""Eres un asistente de gestión operativa para restaurantes y servicios de alimentos en México.
Tu trabajo es generar reportes ejecutivos COMPLETOS y DETALLADOS para el equipo de operaciones.

═══════════════════════════════════════════════════════════════
                    DATOS DE LA CONSULTA
═══════════════════════════════════════════════════════════════

**CIUDAD:** {ciudad}
**INGREDIENTES DISPONIBLES EN INVENTARIO:** {', '.join(ingredientes)}
**NÚMERO DE RECETAS ENCONTRADAS:** {len(recetas_con_precios)}

{recetas_detalle}

═══════════════════════════════════════════════════════════════
                    INSTRUCCIONES
═══════════════════════════════════════════════════════════════

Genera un REPORTE EJECUTIVO COMPLETO con la siguiente estructura:

## 📋 RESUMEN EJECUTIVO
[Análisis de 3-4 líneas sobre las opciones encontradas y la mejor recomendación]

## 🥇 RECETA RECOMENDADA: [Nombre traducido al español]

### Información General
- **Nombre en español:** [Traduce el nombre]
- **Tiempo de preparación:** X minutos
- **Porciones:** X
- **Dificultad estimada:** [Fácil/Media/Difícil]

### Ingredientes Necesarios (traducidos al español)
[Lista COMPLETA de ingredientes traducidos al español con cantidades si están disponibles]

### Instrucciones de Preparación (en español)
[Traduce las instrucciones paso a paso al español, numeradas]

### 💰 Costos de Ingredientes en {ciudad}
[Para CADA ingrediente con precio disponible, incluir:]
| Ingrediente | Precio | Tienda | Dirección |
|-------------|--------|--------|-----------|
[Tabla con todos los precios]

**COSTO TOTAL ESTIMADO:** $X.XX MXN

### 🛒 Dónde Comprar - Mejores Opciones
[Lista las tiendas recomendadas con direcciones completas]

## 📊 ALTERNATIVAS
[Menciona brevemente las otras recetas encontradas como alternativas]

## ⚠️ RECOMENDACIONES OPERATIVAS
1. [Recomendación sobre aprovechamiento del inventario]
2. [Recomendación sobre compras]
3. [Recomendación sobre preparación]

## 📍 MAPA DE COMPRAS
[Agrupa los ingredientes por tienda para optimizar la ruta de compras]

---
*Precios obtenidos de PROFECO - Quién es Quién en los Precios*
*Datos actualizados en tiempo real*

═══════════════════════════════════════════════════════════════

REGLAS OBLIGATORIAS:
1. TODO debe estar en ESPAÑOL (traduce nombres, ingredientes, instrucciones)
2. Incluye TODAS las direcciones de las tiendas
3. Sé muy detallado y específico
4. Usa formato Markdown con tablas y listas
5. Tono formal y profesional
6. Enfócate en eficiencia y reducción de costos
7. Si hay ingredientes sin precio, sugiere alternativas o tiendas donde buscar"""

    response = gemini_model.generate_content(prompt)
    return response.text


@app.get("/ciudades")
def listar_ciudades():
    """Lista las ciudades disponibles para consulta de precios PROFECO"""
    return {
        "ciudades": list(CIUDADES_QQP.keys()),
        "total": len(CIUDADES_QQP)
    }


@app.get("/precio/{ciudad}/{ingrediente}")
def consultar_precio(ciudad: str, ingrediente: str):
    """
    Consulta el precio de un ingrediente específico en una ciudad.
    El ingrediente puede estar en inglés o español.
    """
    if ciudad not in CIUDADES_QQP:
        raise HTTPException(status_code=400, detail=f"Ciudad no válida: {ciudad}")
    
    # Intentar traducir si está en inglés
    ingrediente_es = traducir_ingrediente(ingrediente)
    
    df = obtener_precio_profeco(ciudad, ingrediente_es)
    
    if df.empty:
        return {
            "ingrediente": ingrediente,
            "ingrediente_busqueda": ingrediente_es,
            "ciudad": ciudad,
            "resultados": [],
            "mensaje": "No se encontraron precios para este ingrediente"
        }
    
    # Retornar top 5 mejores precios
    resultados = df.head(5).to_dict(orient='records')
    
    return {
        "ingrediente": ingrediente,
        "ingrediente_busqueda": ingrediente_es,
        "ciudad": ciudad,
        "resultados": resultados,
        "mejor_precio": resultados[0] if resultados else None,
        "mensaje": f"Se encontraron {len(df)} opciones"
    }


# ============================================================
# ENDPOINT HTML FORMATEADO
# ============================================================
import markdown

@app.post("/buscar/html")
def buscar_recetas_html(request: InventarioRequest):
    """
    Igual que /buscar pero devuelve el análisis operativo en HTML formateado.
    Útil para renderizar directamente en el frontend sin librerías de markdown.
    """
    # Reutilizar la lógica de búsqueda
    resultado = buscar_recetas(request)
    
    # Convertir markdown a HTML si hay análisis
    analisis_html = None
    if resultado.analisis_operativo:
        analisis_html = markdown.markdown(
            resultado.analisis_operativo,
            extensions=['extra', 'nl2br']
        )
    
    return {
        "query_ingredientes": resultado.query_ingredientes,
        "recetas": [r.model_dump() for r in resultado.recetas],
        "analisis_operativo_md": resultado.analisis_operativo,
        "analisis_operativo_html": analisis_html,
        "total_encontradas": resultado.total_encontradas,
        "mensaje": resultado.mensaje
    }


# ============================================================
# ENDPOINT /recommend - OPTIMIZADO PARA FRONTEND (TARJETAS)
# ============================================================

class RecommendRequest(BaseModel):
    """Request para tarjetas de recetas"""
    ingredients: list[str] = Field(..., description="Ingredientes disponibles", example=["chicken", "rice", "tomato"])
    city: str = Field(default="Ciudad de México", description="Ciudad para precios PROFECO")
    max_results: int = Field(default=3, ge=1, le=5, description="Número de tarjetas (1-5)")


class IngredientePrecio(BaseModel):
    """Precio de un ingrediente para la tarjeta"""
    nombre: str = Field(description="Nombre en español")
    precio: float = Field(description="Precio en MXN")
    tienda: str = Field(description="Nombre de la tienda")
    direccion: str = Field(description="Dirección de la tienda")


class TarjetaReceta(BaseModel):
    """Una tarjeta de receta para el frontend"""
    id: str
    nombre_receta: str = Field(description="Nombre de la receta en español")
    imagen_url: Optional[str] = Field(default=None, description="URL de imagen")
    tiempo_preparacion: int = Field(description="Minutos de preparación")
    porciones: int
    
    # Scores para badges
    score_popularidad: float = Field(description="Score satisfacción cliente 0-100")
    score_ahorro: float = Field(description="Score costo-efectividad 0-100")
    etiqueta: str = Field(description="'💰 Mejor Precio', '❤️ Más Popular', '⚖️ Equilibrada'")
    
    # Ingredientes con precios
    ingredientes_con_precio: list[IngredientePrecio]
    ingredientes_sin_precio: list[str]
    
    # Análisis de costos
    costo_total: float = Field(description="Costo total en MXN")
    costo_por_porcion: float = Field(description="Costo por porción en MXN")
    precio_venta_sugerido: float = Field(description="Precio de venta sugerido MXN")
    ganancia_por_porcion: float = Field(description="Ganancia por porción MXN")
    
    # Análisis LLM corto
    analisis: str = Field(description="Análisis breve de ganancias y ahorro")


class RecommendResponse(BaseModel):
    """Respuesta con tarjetas para frontend"""
    ciudad: str
    ingredientes: list[str]
    tarjetas: list[TarjetaReceta]
    resumen: str = Field(description="Resumen para el header del UI")


# Traducciones de nombres de recetas
TRADUCCION_RECETAS = {
    "chicken": "Pollo", "rice": "Arroz", "soup": "Sopa", "salad": "Ensalada",
    "pasta": "Pasta", "stew": "Guiso", "roasted": "Asado", "grilled": "A la Parrilla",
    "fried": "Frito", "baked": "Horneado", "with": "con", "and": "y", "red": "Rojo",
    "lentil": "Lentejas", "turnips": "Nabos", "creamy": "Cremoso", "spicy": "Picante",
    "garlic": "Ajo", "tomato": "Tomate", "onion": "Cebolla", "beef": "Res",
    "pork": "Cerdo", "fish": "Pescado", "shrimp": "Camarones", "vegetable": "Verduras",
    "mushroom": "Champiñones", "cheese": "Queso", "cream": "Crema", "butter": "Mantequilla",
}


def traducir_nombre_receta(nombre_en: str) -> str:
    """Traduce nombre de receta al español"""
    nombre = nombre_en
    for en, es in TRADUCCION_RECETAS.items():
        nombre = nombre.lower().replace(en.lower(), es)
    return nombre.title()


def generar_analisis_tarjeta(nombre: str, costo_total: float, costo_porcion: float, 
                              precio_venta: float, ganancia: float, porciones: int,
                              ingredientes_precio: list) -> str:
    """Genera análisis breve con LLM enfocado en ganancias"""
    
    ingredientes_texto = ", ".join([f"{p['nombre']}: ${p['precio']:.2f}" for p in ingredientes_precio[:4]]) if ingredientes_precio else "Sin datos"
    
    prompt = f"""Genera un análisis MUY BREVE (máximo 3 oraciones) para una tarjeta de receta.
TODO EN ESPAÑOL. Enfócate SOLO en: ganancias, ahorro y rentabilidad.

DATOS:
- Receta: {nombre}
- Costo total ingredientes: ${costo_total:.2f} MXN
- Costo por porción: ${costo_porcion:.2f} MXN
- Precio venta sugerido: ${precio_venta:.2f} MXN
- Ganancia por porción: ${ganancia:.2f} MXN
- Porciones: {porciones}
- Ganancia total potencial: ${ganancia * porciones:.2f} MXN
- Precios ingredientes: {ingredientes_texto}

GENERA exactamente 3 oraciones cortas:
1. Costo y margen de ganancia
2. Ganancia total si vendes todas las porciones
3. Una recomendación de venta

NO uses markdown, solo texto plano. Sé DIRECTO y CONCISO."""

    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except:
        return f"💰 Costo: ${costo_total:.2f} MXN (${costo_porcion:.2f}/porción). Precio venta: ${precio_venta:.2f}. Ganancia potencial: ${ganancia * porciones:.2f} MXN vendiendo {porciones} porciones."


def asignar_etiquetas_tarjetas(tarjetas: list[TarjetaReceta]) -> list[TarjetaReceta]:
    """Asigna etiquetas a las tarjetas"""
    if not tarjetas:
        return tarjetas
    
    mejor_ahorro = max(range(len(tarjetas)), key=lambda i: tarjetas[i].score_ahorro)
    mas_popular = max(range(len(tarjetas)), key=lambda i: tarjetas[i].score_popularidad)
    
    for i, t in enumerate(tarjetas):
        if i == mejor_ahorro and i == mas_popular:
            t.etiqueta = "⭐ Mejor Opción"
        elif i == mejor_ahorro:
            t.etiqueta = "💰 Mejor Precio"
        elif i == mas_popular:
            t.etiqueta = "❤️ Más Popular"
        else:
            t.etiqueta = "⚖️ Equilibrada"
    
    return tarjetas


@app.post("/recommend", response_model=RecommendResponse)
def recommend_recipes(request: RecommendRequest):
    """
    🎯 Endpoint para TARJETAS del frontend
    
    Devuelve tarjetas con:
    - Nombre en español
    - Ingredientes con precios y tiendas
    - Análisis de ganancias (LLM)
    - Costo total, por porción, precio venta sugerido
    """
    try:
        if request.city not in CIUDADES_QQP:
            raise HTTPException(status_code=400, detail=f"Ciudad no válida. Opciones: {list(CIUDADES_QQP.keys())}")
        
        # 1. Buscar recetas con RAG + Reranking
        query_text = f"Recipes with: {', '.join(request.ingredients)}"
        query_embedding = crear_embedding(query_text)
        indices = buscar_faiss(query_embedding, top_k=20)
        candidatas = df_recetas.iloc[indices].copy()
        
        if len(candidatas) == 0:
            return RecommendResponse(
                ciudad=request.city,
                ingredientes=request.ingredients,
                tarjetas=[],
                resumen="No se encontraron recetas con estos ingredientes."
            )
        
        # Reranking
        docs = [f"{row['title']}. {row['ingredients']}. {row['tags']}" for _, row in candidatas.iterrows()]
        rerank_results = rerank_con_cohere(query_text, docs, top_n=request.max_results)
        
        # 2. Construir tarjetas
        tarjetas = []
        total_ing = len(request.ingredients)
        
        for idx_local, relevance in rerank_results:
            row = candidatas.iloc[idx_local]
            matching = encontrar_ingredientes_coincidentes(request.ingredients, row['ingredients'])
            
            # Obtener precios de PROFECO
            ingredientes_precio = []
            ingredientes_sin_precio = []
            costo_total = 0.0
            
            for ing in matching:
                precio_info = obtener_mejor_precio(request.city, ing)
                if precio_info:
                    ingredientes_precio.append(IngredientePrecio(
                        nombre=precio_info["ingrediente_busqueda"],
                        precio=precio_info["precio"],
                        tienda=precio_info["cadena_comercial"],
                        direccion=f"{precio_info['direccion']}, {precio_info['colonia']}"
                    ))
                    costo_total += precio_info["precio"]
                else:
                    ingredientes_sin_precio.append(traducir_ingrediente(ing))
            
            # Calcular scores
            score_pop = calcular_score_popularidad(row['healthScore'], row['tags'], row['readyInMinutes'])
            score_ahorro = calcular_score_costo_efectividad(
                row['pricePerServing'], row['servings'], len(matching), total_ing, row['readyInMinutes']
            )
            
            # Calcular costos y ganancias
            # Si no hay costo de PROFECO, estimar con precio referencia * tipo de cambio
            if costo_total == 0:
                costo_total = row['pricePerServing'] * row['servings'] * 17  # USD a MXN aprox
            
            costo_porcion = costo_total / row['servings']
            precio_venta = round(costo_porcion * 2.5, 2)  # Margen 60%
            ganancia = round(precio_venta - costo_porcion, 2)
            
            # Traducir nombre
            nombre_es = traducir_nombre_receta(row['title'])
            
            # Generar imagen URL
            recipe_id = row['id'].replace('spoon_', '') if 'spoon_' in str(row['id']) else str(row['id'])
            imagen_url = f"https://spoonacular.com/recipeImages/{recipe_id}-480x360.jpg"
            
            # Generar análisis con LLM
            analisis = generar_analisis_tarjeta(
                nombre=nombre_es,
                costo_total=costo_total,
                costo_porcion=costo_porcion,
                precio_venta=precio_venta,
                ganancia=ganancia,
                porciones=row['servings'],
                ingredientes_precio=[p.model_dump() for p in ingredientes_precio]
            )
            
            tarjeta = TarjetaReceta(
                id=row['id'],
                nombre_receta=nombre_es,
                imagen_url=imagen_url,
                tiempo_preparacion=row['readyInMinutes'],
                porciones=row['servings'],
                score_popularidad=score_pop,
                score_ahorro=score_ahorro,
                etiqueta="",
                ingredientes_con_precio=ingredientes_precio,
                ingredientes_sin_precio=ingredientes_sin_precio,
                costo_total=round(costo_total, 2),
                costo_por_porcion=round(costo_porcion, 2),
                precio_venta_sugerido=precio_venta,
                ganancia_por_porcion=ganancia,
                analisis=analisis
            )
            tarjetas.append(tarjeta)
        
        # 3. Asignar etiquetas
        tarjetas = asignar_etiquetas_tarjetas(tarjetas)
        
        # 4. Generar resumen
        mejor = max(tarjetas, key=lambda t: t.score_ahorro + t.score_popularidad) if tarjetas else None
        resumen = f"🎯 {len(tarjetas)} recetas encontradas en {request.city}. Recomendación: {mejor.nombre_receta} ({mejor.etiqueta})" if mejor else "Sin resultados"
        
        return RecommendResponse(
            ciudad=request.city,
            ingredientes=request.ingredients,
            tarjetas=tarjetas,
            resumen=resumen
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
