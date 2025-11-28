# 🍳 SmartPantry API

**AI-powered inventory optimization for food service operations.**

Sistema RAG (Retrieval-Augmented Generation) que recomienda recetas basadas en ingredientes disponibles, con precios en tiempo real de PROFECO y análisis de rentabilidad.

## 🚀 Tecnologías

| Componente | Tecnología |
|------------|------------|
| Embeddings | Voyage AI (`voyage-3-large`) |
| Vector DB | FAISS (1024 dimensiones) |
| Reranking | Cohere (`rerank-v3.5`) |
| LLM | Google Gemini |
| Precios | API PROFECO (34 ciudades de México) |
| Framework | FastAPI |

## 📦 Instalación

```bash
# Clonar repositorio
git clone https://github.com/Vania-Janet/food_api.git
cd food_api

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus API keys
```

## 🔑 Variables de Entorno

Crea un archivo `.env` con:

```env
VOYAGE_API_KEY=tu_api_key_voyage
COHERE_API_KEY=tu_api_key_cohere
GOOGLE_API_KEY=tu_api_key_google
```

## ▶️ Ejecutar

```bash
python app.py
```

El servidor estará disponible en `http://localhost:8000`

---

## 📡 API Endpoints

### `POST /recommend`

Endpoint principal para obtener tarjetas de recetas con precios y análisis de rentabilidad.

#### Request

```bash
curl -X POST https://food-api-vgt8.onrender.com/recommend \
  -H "Content-Type: application/json" \
  -d '{"ingredients": ["chicken", "rice", "tomato"], "city": "Ciudad de México"}'
```

#### Response

```json
{
    "ciudad": "Ciudad de México",
    "ingredientes": [
        "chicken",
        "rice",
        "tomato",
        "onion"
    ],
    "tarjetas": [
        {
            "id": "spoon_716361",
            "nombre_receta": "Stir Frito Quinoa, Brown Arroz Y Pollo Breast",
            "imagen_url": "https://spoonacular.com/recipeImages/716361-480x360.jpg",
            "tiempo_preparacion": 45,
            "porciones": 1,
            "score_popularidad": 47.8,
            "score_ahorro": 62.0,
            "etiqueta": "💰 Mejor Precio",
            "ingredientes_con_precio": [
                {
                    "nombre": "pollo",
                    "precio": 99.0,
                    "tienda": "CHEDRAUI",
                    "direccion": "ANFORA 71, ESQ. EJE 1 NORTE, MADERO"
                },
                {
                    "nombre": "arroz",
                    "precio": 14.0,
                    "tienda": "CENTRAL DE ABASTOS",
                    "direccion": "EJE 6 SUR 560, LOCAL 176  PASILLO 2, SAN JOSE ACULCO"
                },
                {
                    "nombre": "tomate",
                    "precio": 21.68,
                    "tienda": "U.N.A.M.",
                    "direccion": "DALIAS S/N CERCA METRO UNIVERSIDAD, COPILCO EL ALTO"
                },
                {
                    "nombre": "cebolla",
                    "precio": 10.0,
                    "tienda": "CENTRAL DE ABASTOS",
                    "direccion": "PROL. EJE 6 SUR 520, ESQ. CANAL DE APATLACO, EJIDOS DEL MORAL"
                }
            ],
            "ingredientes_sin_precio": [],
            "costo_total": 144.68,
            "costo_por_porcion": 144.68,
            "precio_venta_sugerido": 361.7,
            "ganancia_por_porcion": 217.02,
            "analisis": "El costo por porción es de $144.68, ofreciendo una ganancia neta de $217.02. La ganancia total potencial por esta receta es de $217.02. Dado su alto margen, promociónalo como un plato premium para maximizar la rentabilidad."
        },
        {
            "id": "spoon_638257",
            "nombre_receta": "Pollo Porridge",
            "imagen_url": "https://spoonacular.com/recipeImages/638257-480x360.jpg",
            "tiempo_preparacion": 45,
            "porciones": 4,
            "score_popularidad": 27.8,
            "score_ahorro": 57.0,
            "etiqueta": "⚖️ Equilibrada",
            "ingredientes_con_precio": [
                {
                    "nombre": "pollo",
                    "precio": 99.0,
                    "tienda": "CHEDRAUI",
                    "direccion": "ANFORA 71, ESQ. EJE 1 NORTE, MADERO"
                },
                {
                    "nombre": "arroz",
                    "precio": 14.0,
                    "tienda": "CENTRAL DE ABASTOS",
                    "direccion": "EJE 6 SUR 560, LOCAL 176  PASILLO 2, SAN JOSE ACULCO"
                },
                {
                    "nombre": "cebolla",
                    "precio": 10.0,
                    "tienda": "CENTRAL DE ABASTOS",
                    "direccion": "PROL. EJE 6 SUR 520, ESQ. CANAL DE APATLACO, EJIDOS DEL MORAL"
                }
            ],
            "ingredientes_sin_precio": [],
            "costo_total": 123.0,
            "costo_por_porcion": 30.75,
            "precio_venta_sugerido": 76.88,
            "ganancia_por_porcion": 46.13,
            "analisis": "Cada porción cuesta $30.75 y genera una ganancia de $46.13. Vender las cuatro porciones te dará una ganancia total de $184.52. Este platillo es altamente rentable, ya que la ganancia es superior al costo."
        },
        {
            "id": "spoon_982382",
            "nombre_receta": "Instant Pot Pollo Taco Sopa",
            "imagen_url": "https://spoonacular.com/recipeImages/982382-480x360.jpg",
            "tiempo_preparacion": 25,
            "porciones": 4,
            "score_popularidad": 57.9,
            "score_ahorro": 60.0,
            "etiqueta": "❤️ Más Popular",
            "ingredientes_con_precio": [
                {
                    "nombre": "pollo",
                    "precio": 99.0,
                    "tienda": "CHEDRAUI",
                    "direccion": "ANFORA 71, ESQ. EJE 1 NORTE, MADERO"
                },
                {
                    "nombre": "tomate",
                    "precio": 21.68,
                    "tienda": "U.N.A.M.",
                    "direccion": "DALIAS S/N CERCA METRO UNIVERSIDAD, COPILCO EL ALTO"
                },
                {
                    "nombre": "cebolla",
                    "precio": 10.0,
                    "tienda": "CENTRAL DE ABASTOS",
                    "direccion": "PROL. EJE 6 SUR 520, ESQ. CANAL DE APATLACO, EJIDOS DEL MORAL"
                }
            ],
            "ingredientes_sin_precio": [],
            "costo_total": 130.68,
            "costo_por_porcion": 32.67,
            "precio_venta_sugerido": 81.68,
            "ganancia_por_porcion": 49.01,
            "analisis": "Cada porción cuesta $32.67 y deja una ganancia de $49.01. Al vender las 4 porciones, obtienes una ganancia total de $196.04. Ofrece combos con bebida o guarniciones extra para aumentar la rentabilidad."
        }
    ],
    "resumen": "🎯 3 recetas encontradas en Ciudad de México. Recomendación: Instant Pot Pollo Taco Sopa (❤️ Más Popular)"
}
```

---

### `GET /cities`

Lista las ciudades disponibles para consulta de precios PROFECO.

```bash
curl http://localhost:8000/cities
```

---

### `GET /stats`

Estadísticas del sistema.

```bash
curl http://localhost:8000/stats
```

---

## 📊 Estructura de Respuesta

Cada tarjeta de receta incluye:

| Campo | Descripción |
|-------|-------------|
| `nombre_receta` | Nombre de la receta en español |
| `imagen_url` | URL de la imagen de la receta |
| `tiempo_preparacion` | Minutos de preparación |
| `porciones` | Número de porciones |
| `score_popularidad` | Puntuación de satisfacción del cliente (0-100) |
| `score_ahorro` | Puntuación de costo-efectividad (0-100) |
| `etiqueta` | Badge: 💰 Mejor Precio, ❤️ Más Popular, ⚖️ Equilibrada |
| `ingredientes_con_precio` | Lista de ingredientes con precios de PROFECO |
| `costo_total` | Costo total estimado en MXN |
| `costo_por_porcion` | Costo por porción en MXN |
| `precio_venta_sugerido` | Precio de venta sugerido (margen 60%) |
| `ganancia_por_porcion` | Ganancia estimada por porción |
| `analisis` | Análisis de rentabilidad generado por IA |

---

## 🏙️ Ciudades Disponibles

El sistema soporta 34 ciudades de México:

Acapulco, Aguascalientes, Campeche, Cancún, Chihuahua, Ciudad de México, Ciudad Juárez, Cuernavaca, Culiacán, Durango, Estado de México, Guadalajara, Hermosillo, La Paz, León, Monterrey, Morelia, Mérida, Oaxaca, Orizaba, Pachuca, Playa del Carmen, Puebla, Querétaro, Saltillo, San Luis Potosí, Tampico, Tijuana, Tlaxcala, Tuxtla Gutiérrez, Veracruz, Villahermosa, Zacatecas.

---

## 👥 Equipo

Proyecto desarrollado para el curso de Machine Learning.

---

## 📄 Licencia

MIT
