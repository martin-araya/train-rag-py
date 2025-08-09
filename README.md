# API RAG con Robyn, ChromaDB y LangChain

Una API de alto rendimiento de Generación Aumentada por Recuperación (RAG) construida con **Robyn**, **ChromaDB** y **LangChain** para procesamiento de documentos y consultas inteligentes. Este sistema te permite subir documentos PDF, procesarlos en una base de datos vectorial y realizar consultas en lenguaje natural contra el contenido.

## 🏗️ Arquitectura

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Subida de PDF  │───▶│ Divisor de Texto │───▶│    ChromaDB     │
│                 │    │    (Chunks)      │    │ (Vector Store)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐             │
│ Consulta Usuario│───▶│  Pipeline RAG    │◀────────────┘
│                 │    │   (LangChain)    │
└─────────────────┘    └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Ollama LLM     │
                       │   Respuesta      │
                       └──────────────────┘
```

## 🚀 Características

- **Procesamiento Rápido de PDF**: Sube y procesa documentos PDF con PyMuPDF
- **Búsqueda Vectorial**: Búsqueda semántica usando ChromaDB y embeddings de Ollama
- **Respuestas Inteligentes**: Respuestas contextualmente conscientes usando LLM de Ollama
- **Alto Rendimiento**: Construido con Robyn para rendimiento asíncrono
- **Soporte CORS**: Listo para integración con frontend
- **Monitoreo de Salud**: Verificaciones de salud y endpoints de debug integrados
- **Preguntas Rápidas**: Consultas pre-optimizadas para respuestas más veloces

## 📋 Requisitos Previos

Antes de ejecutar este proyecto, asegúrate de tener los siguientes servicios ejecutándose:

### 1. Servidor ChromaDB
```bash
# Usando Docker (recomendado)
docker run -p 8000:8000 chromadb/chroma

# O instalar localmente
pip install chromadb
chroma run --host 0.0.0.0 --port 8000
```

### 2. Servidor Ollama
```bash
# Instalar Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Descargar los modelos necesarios
ollama pull granite-embedding:278m  # Para embeddings
ollama pull llama3.2:3b-instruct-q6_K  # Para generación de texto

# Iniciar el servidor Ollama (normalmente corre en el puerto 11434)
ollama serve
```

## 🛠️ Instalación

1. **Clonar el repositorio**
```bash
git clone <url-del-repositorio>
cd rag-api
```

2. **Crear entorno virtual**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

### Requirements.txt
```txt
robyn
chromadb-client
langchain
langchain-community
pymupdf
sentence-transformers
langchain_core
```

## ⚙️ Configuración

La API usa la siguiente configuración por defecto:

```python
# Configuración ChromaDB
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "pdf_documents_collection"

# Configuración Ollama
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_EMBEDDING_MODEL = "granite-embedding:278m"
OLLAMA_GENERATION_MODEL = "llama3.2:3b-instruct-q6_K"

# Configuración del Servidor
HOST = "0.0.0.0"
PORT = 8080
```

Para modificar estas configuraciones, edita las constantes al principio del archivo Python principal.

## 🏃‍♂️ Ejecutar la Aplicación

1. **Iniciar los servicios requeridos** (ChromaDB y Ollama)

2. **Ejecutar el servidor de la API**
```bash
python main.py
```

El servidor iniciará en `http://localhost:8080`

## 📚 Endpoints de la API

### Endpoints Principales

#### `POST /upload-pdf`
Subir y procesar un documento PDF.

**Petición:**
```bash
curl -X POST \
  -F "upload_file=@documento.pdf" \
  http://localhost:8080/upload-pdf
```

**Respuesta:**
```json
{
  "message": "Archivo 'documento.pdf' procesado exitosamente.",
  "chunks_added": 45,
  "status": "success"
}
```

#### `GET /query?q={pregunta}`
Consultar los documentos procesados.

**Petición:**
```bash
curl "http://localhost:8080/query?q=¿Cuál%20es%20el%20tema%20principal%20de%20este%20documento?"
```

**Respuesta:**
```json
{
  "query": "¿Cuál es el tema principal de este documento?",
  "answer": "El documento discute técnicas de aprendizaje automático...",
  "status": "success",
  "docs_found": 3,
  "processing_time": "1.23s",
  "context_length": 1850
}
```

### Endpoints de Información

#### `GET /doc-info`
Obtener información básica sobre los documentos cargados.

**Respuesta:**
```json
{
  "total_chunks": 45,
  "sample_content": "Este paper de investigación presenta...",
  "metadata": {
    "filename": "investigacion.pdf"
  },
  "content_preview": [
    {
      "chunk": 1,
      "preview": "Contenido de la sección de introducción..."
    }
  ]
}
```

#### `GET /quick-questions`
Obtener preguntas sugeridas y consejos para consultar documentos.

**Respuesta:**
```json
{
  "suggested_questions": [
    "¿De qué trata este documento?",
    "¿Cuál es el tema principal?",
    "¿Qué métodos se describen?",
    "¿Cuáles son las conclusiones principales?"
  ],
  "document_type": "Research Paper/Academic Document",
  "content_sample": "...",
  "tips": [
    "Pregunta sobre 'métodos' o 'methodology'",
    "Pregunta sobre 'resultados' o 'results'"
  ]
}
```

### Endpoints del Sistema

#### `GET /health`
Verificar el estado de salud de todos los servicios.

**Respuesta:**
```json
{
  "status": "healthy",
  "chromadb": true,
  "ollama": true,
  "timestamp": "2024-01-15T10:30:00"
}
```

#### `GET /debug`
Obtener información detallada de depuración.

**Respuesta:**
```json
{
  "vector_store": true,
  "llm": true,
  "chroma_client": true,
  "collection_name": "pdf_documents_collection",
  "ollama_base_url": "http://localhost:11434"
}
```

#### `GET /`
Obtener información de la API y endpoints disponibles.

## 🔧 Integración con Frontend

La API incluye soporte CORS para aplicaciones frontend. Ejemplo de uso con JavaScript:

```javascript
// Subir PDF
const subirPDF = async (archivo) => {
  const formData = new FormData();
  formData.append('upload_file', archivo);
  
  const respuesta = await fetch('http://localhost:8080/upload-pdf', {
    method: 'POST',
    body: formData
  });
  
  return await respuesta.json();
};

// Consultar documentos
const consultarDocumentos = async (pregunta) => {
  const respuesta = await fetch(
    `http://localhost:8080/query?q=${encodeURIComponent(pregunta)}`
  );
  
  return await respuesta.json();
};
```

## 🐛 Solución de Problemas

### Problemas Comunes

1. **Error de Conexión con ChromaDB**
   ```
   ❌ No se pudo conectar a ChromaDB
   ```
   - Asegúrate de que ChromaDB esté ejecutándose en el puerto 8000
   - Verifica si el puerto está disponible: `netstat -an | grep 8000`

2. **Modelo de Ollama No Encontrado**
   ```
   ❌ Error configurando modelo de generación
   ```
   - Descarga los modelos requeridos: `ollama pull granite-embedding:278m`
   - Verifica que los modelos estén disponibles: `ollama list`

3. **Error de Procesamiento de PDF**
   ```
   ❌ No se pudo procesar el contenido del PDF
   ```
   - Asegúrate de que el archivo subido sea un PDF válido
   - Verifica permisos de archivo y límites de tamaño

4. **Respuestas Lentas en Consultas**
   - Considera usar modelos más pequeños para respuestas más rápidas
   - Reduce el tamaño de chunk en la configuración
   - Verifica recursos del sistema (CPU, memoria)

### Optimización de Rendimiento

1. **Reducir Tamaño del Modelo**: Usa modelos de Ollama más pequeños para respuestas más rápidas
2. **Optimizar Tamaño de Chunk**: Ajusta el parámetro `chunk_size` (predeterminado: 1000)
3. **Limitar Contexto**: Reduce `max_context_length` para procesamiento LLM más rápido
4. **Usar GPU**: Configura Ollama para usar aceleración GPU si está disponible

## 📊 Monitoreo

La API proporciona varias capacidades de monitoreo:

- **Verificaciones de Salud**: Endpoint `/health` para estado de servicios
- **Info de Debug**: Endpoint `/debug` para detalles de configuración
- **Métricas de Procesamiento**: Tiempos de respuesta de consultas y conteos de documentos
- **Logging**: Logging detallado en consola con emojis para fácil lectura

## 🔒 Consideraciones de Seguridad

- La API actualmente permite todos los orígenes CORS para desarrollo
- La validación de subida de archivos es básica (solo extensión de archivo)
- Considera agregar autenticación para uso en producción
- Implementa limitación de velocidad para despliegues públicos

## 🚀 Despliegue en Producción

Para despliegue en producción:

1. **Usar Variables de Entorno** para configuración
2. **Configurar Proxy Inverso** (nginx/Apache)
3. **Configurar SSL/TLS**
4. **Configurar Logging Apropiado**
5. **Implementar Monitoreo de Salud**
6. **Usar Gestor de Procesos** (PM2, systemd)

Ejemplo de servicio systemd:
```ini
[Unit]
Description=Servicio API RAG
After=network.target

[Service]
Type=simple
User=tu-usuario
WorkingDirectory=/ruta/a/rag-api
ExecStart=/ruta/a/venv/bin/python main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

## 🔧 Variables de Entorno para Producción

Crea un archivo `.env` para configuración:

```env
# ChromaDB
CHROMA_HOST=localhost
CHROMA_PORT=8000
COLLECTION_NAME=pdf_documents_collection

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=granite-embedding:278m
OLLAMA_GENERATION_MODEL=llama3.2:3b-instruct-q6_K

# Servidor
HOST=0.0.0.0
PORT=8080

# Límites
MAX_CHUNK_SIZE=1000
MAX_CONTEXT_LENGTH=2000
MAX_FILE_SIZE=50MB
```

## 🤝 Contribuir

1. Hacer fork del repositorio
2. Crear una rama de característica
3. Hacer tus cambios
4. Agregar pruebas si es aplicable
5. Enviar un pull request

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo LICENSE para detalles.

## 🙏 Agradecimientos

- **Robyn**: Framework web Python de alto rendimiento
- **ChromaDB**: Base de datos vectorial de código abierto
- **LangChain**: Framework para desarrollar aplicaciones LLM
- **Ollama**: Plataforma de despliegue local de LLM
- **PyMuPDF**: Biblioteca de procesamiento de PDF

## 📞 Soporte

Si encuentras problemas o tienes preguntas:

1. Revisa la sección de solución de problemas arriba
2. Revisa los logs para mensajes de error
3. Asegúrate de que todos los requisitos previos estén instalados correctamente
4. Abre un issue en el repositorio

## 🔥 Casos de Uso Comunes

### Análisis de Documentos Académicos
```bash
# Subir paper de investigación
curl -X POST -F "upload_file=@paper_investigacion.pdf" http://localhost:8080/upload-pdf

# Preguntas típicas
curl "http://localhost:8080/query?q=¿Cuál es la metodología utilizada?"
curl "http://localhost:8080/query?q=¿Cuáles son las conclusiones principales?"
```

### Procesamiento de Manuales Técnicos
```bash
# Subir manual
curl -X POST -F "upload_file=@manual_tecnico.pdf" http://localhost:8080/upload-pdf

# Consultas específicas
curl "http://localhost:8080/query?q=¿Cómo configurar el sistema?"
curl "http://localhost:8080/query?q=¿Cuáles son los requisitos de instalación?"
```

### Análisis de Reportes Empresariales
```bash
# Subir reporte
curl -X POST -F "upload_file=@reporte_anual.pdf" http://localhost:8080/upload-pdf

# Análisis financiero
curl "http://localhost:8080/query?q=¿Cuáles fueron los ingresos del último trimestre?"
curl "http://localhost:8080/query?q=¿Qué estrategias se mencionan para el crecimiento?"
```

---

**Construido con ❤️ usando tecnologías Python modernas para procesamiento rápido e inteligente de documentos.**