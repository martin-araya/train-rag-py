from robyn import Robyn, Request, Response, ALLOW_CORS
from urllib.parse import unquote
import chromadb
import os
import tempfile
import json
import traceback
import time

# --- Importaciones de LangChain ---
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

app = Robyn(__file__)

# --- Configuración ---
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "pdf_documents_collection"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_EMBEDDING_MODEL = "granite-embedding:278m"  # El "Archivista" que encuentra documentos
OLLAMA_GENERATION_MODEL = "llama3.2:3b-instruct-q6_K"  # El "Analista" que genera respuestas

print(f"🔧 Configuración:")
print(f"   ChromaDB: {CHROMA_HOST}:{CHROMA_PORT}")
print(f"   Ollama: {OLLAMA_BASE_URL}")
print(f"   Modelo Embeddings: {OLLAMA_EMBEDDING_MODEL}")
print(f"   Modelo Generación: {OLLAMA_GENERATION_MODEL}")

# --- Configuración de Embeddings con Ollama (El Archivista) ---
print(f"🤖 Configurando el modelo de embeddings (Archivista): '{OLLAMA_EMBEDDING_MODEL}'...")
embedding_function = OllamaEmbeddings(
    base_url=OLLAMA_BASE_URL,
    model=OLLAMA_EMBEDDING_MODEL
)
print("✅ Modelo de embeddings configurado.")

# --- Configuración del LLM de Generación con Ollama (El Analista) ---
print(f"🧠 Configurando el modelo de generación (Analista): '{OLLAMA_GENERATION_MODEL}'...")
try:
    llm = Ollama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_GENERATION_MODEL
    )
    # Probar que el modelo funciona
    test_response = llm.invoke("Hello")
    if test_response:
        print("✅ Modelo de generación configurado y probado.")
        print(f"🔍 LLM después de inicialización: {llm is not None} (type: {type(llm)})")
    else:
        print("❌ Modelo de generación responde vacío.")
        llm = None
except Exception as llm_error:
    print(f"❌ Error configurando modelo de generación: {llm_error}")
    llm = None

# --- Conexión a ChromaDB y VectorStore de LangChain ---
chroma_client = None
vector_store = None

try:
    # 1. Conexión directa al cliente de ChromaDB
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    chroma_client.heartbeat()
    print("✅ Conexión exitosa con ChromaDB.")

    # 2. Envolver la conexión con el adaptador de LangChain.
    #    Esto nos permite usar métodos como `add_documents` y `similarity_search`.
    vector_store = Chroma(
        client=chroma_client,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )
    print(f"📚 VectorStore de LangChain conectado a la colección '{COLLECTION_NAME}'.")
    print(f"🔍 Estado después de ChromaDB - vector_store: {vector_store is not None}, llm: {llm is not None}")

except Exception as connection_error:
    print(f"❌ No se pudo conectar a ChromaDB: {connection_error}")
    chroma_client = None
    vector_store = None
    # ✅ NO tocar llm aquí - mantener su estado original
    print(f"🔍 Estado después de error ChromaDB - vector_store: {vector_store is not None}, llm: {llm is not None}")

# --- Configuración CORS global ---
ALLOW_CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5173"])


# === FUNCIONES DE UTILIDAD ===

def get_query_param_safe(request: Request, param_name: str):
    """Método seguro para obtener parámetros de query en Robyn con decodificación URL"""
    try:
        if hasattr(request, 'query_params') and request.query_params:
            # Obtener el parámetro
            raw_value = request.query_params.get(param_name, None)
            if raw_value:
                # Decodificar URL encoding (%20 -> espacio, etc.)
                decoded_value = unquote(raw_value)
                return decoded_value
            return None
        return None
    except Exception as e:
        print(f"Error obteniendo parámetro '{param_name}': {e}")
        return None


def get_header_safe(request: Request, header_name: str):
    """Método seguro para obtener headers en Robyn"""
    try:
        if hasattr(request, 'headers') and request.headers and header_name in request.headers:
            return request.headers[header_name]
        return None
    except Exception as e:
        print(f"Error obteniendo header '{header_name}': {e}")
        return None


# === ENDPOINTS ===

@app.options("/*")
def handle_options(request: Request):
    """Maneja las peticiones OPTIONS para CORS"""
    print(f"📡 [OPTIONS] Petición OPTIONS recibida")

    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, Accept",
            "Access-Control-Max-Age": "86400"
        },
        description=""
    )


@app.get("/")
def index():
    return {"message": "API de RAG con Robyn, LangChain y ChromaDB", "status": "running",
            "endpoints": ["/upload-pdf", "/query", "/health", "/debug", "/doc-info", "/quick-questions"]}


@app.post("/upload-pdf")
async def upload_pdf(request: Request):
    """
    Endpoint para subir un archivo PDF, procesarlo y añadirlo a la base de datos vectorial.
    """
    print(f"📄 Recibida solicitud de upload de PDF")

    # Verificar estado de los servicios con más detalle
    vector_store_available = vector_store is not None
    llm_available = llm is not None

    print(f"🔍 Verificación detallada de servicios:")
    print(f"   vector_store: {vector_store_available} (type: {type(vector_store)})")
    print(f"   llm: {llm_available} (type: {type(llm)})")

    if not vector_store_available or not llm_available:
        print(f"❌ Servicios no disponibles - vector_store: {vector_store_available}, llm: {llm_available}")
        return Response(
            status_code=503,
            headers={"Content-Type": "application/json"},
            description=json.dumps({"error": "La conexión con los servicios (ChromaDB/Ollama) no está disponible."})
        )

    try:
        # Try different ways to access the uploaded file
        uploaded_file = None
        filename = ""
        file_content = None

        # Check body first (most likely scenario)
        if hasattr(request, 'body') and request.body:
            file_content = request.body
            filename = "body_upload.pdf"
            print(f"✅ Using request body as file content (size: {len(file_content)})")

        # Check files attribute
        elif hasattr(request, 'files') and request.files and "upload_file" in request.files:
            uploaded_file = request.files["upload_file"]
            print(f"✅ Found file in request.files['upload_file']")

            # Handle different file object types in Robyn
            if hasattr(uploaded_file, 'filename') and hasattr(uploaded_file, 'file'):
                filename = uploaded_file.filename
                file_content = uploaded_file.file
            elif isinstance(uploaded_file, bytes):
                filename = "uploaded_file.pdf"
                file_content = uploaded_file
            else:
                # Try to extract content anyway
                filename = str(getattr(uploaded_file, 'filename', 'unknown.pdf'))
                file_content = getattr(uploaded_file, 'file', uploaded_file)

        # Check form data
        elif hasattr(request, 'form_data') and request.form_data and "upload_file" in request.form_data:
            uploaded_file = request.form_data["upload_file"]
            filename = "form_upload.pdf"
            file_content = uploaded_file
            print(f"✅ Found file in form_data")

        if not file_content:
            error_msg = "Se requiere un archivo PDF en el campo 'upload_file'."
            print(f"❌ No file content found. {error_msg}")
            return Response(
                status_code=400,
                headers={"Content-Type": "application/json"},
                description=json.dumps({"error": error_msg})
            )

        # Verify it's a PDF file by filename
        if not filename.lower().endswith('.pdf'):
            print(f"❌ File is not a PDF: {filename}")
            return Response(
                status_code=400,
                headers={"Content-Type": "application/json"},
                description=json.dumps({"error": "El archivo debe ser un PDF."})
            )

        print(
            f"📄 Procesando archivo: {filename} (size: {len(file_content) if isinstance(file_content, bytes) else 'unknown'})")

        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            if isinstance(file_content, bytes):
                tmp.write(file_content)
            elif hasattr(file_content, 'read'):
                content = file_content.read()
                if isinstance(content, str):
                    content = content.encode('utf-8')
                tmp.write(content)
            elif isinstance(file_content, str):
                tmp.write(file_content.encode('utf-8'))
            else:
                tmp.write(bytes(file_content))

            tmp_path = tmp.name

        print(f"📁 Archivo temporal creado: {tmp_path}")

        # Load and process the PDF
        loader = PyMuPDFLoader(tmp_path)
        documents = loader.load()

        if not documents:
            print(f"❌ No se pudieron extraer documentos del PDF")
            return Response(
                status_code=400,
                headers={"Content-Type": "application/json"},
                description=json.dumps({"error": "No se pudo procesar el contenido del PDF."})
            )

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        # Add metadata
        for chunk in chunks:
            chunk.metadata["filename"] = filename

        # Add to vector store
        vector_store.add_documents(documents=chunks, ids=None)

        print(f"✅ Archivo '{filename}' procesado. {len(chunks)} chunks añadidos a la colección.")

        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            print(f"🗑️ Archivo temporal eliminado: {tmp_path}")

        return Response(
            status_code=200,
            headers={"Content-Type": "application/json"},
            description=json.dumps({
                "message": f"Archivo '{filename}' procesado exitosamente.",
                "chunks_added": len(chunks),
                "status": "success"
            })
        )

    except Exception as upload_error:
        print(f"❌ Error procesando PDF: {str(upload_error)}")
        print(f"Error traceback: {traceback.format_exc()}")

        # Clean up temp file if it exists
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                print(f"🗑️ Archivo temporal eliminado después del error: {tmp_path}")
            except:
                pass

        return Response(
            status_code=500,
            headers={"Content-Type": "application/json"},
            description=json.dumps({
                "error": f"Ocurrió un error al procesar el PDF: {str(upload_error)}",
                "status": "error"
            })
        )


@app.get("/query")
def query_collection(request: Request):
    """
    Endpoint optimizado para hacer preguntas sin timeout.
    """
    print(f"🔎 Recibida petición query")

    # Verificar estado de los servicios
    if not vector_store or not llm:
        return Response(
            status_code=503,
            headers={"Content-Type": "application/json"},
            description=json.dumps({"error": "Servicios no disponibles."})
        )

    # Obtener y decodificar pregunta
    query_text = get_query_param_safe(request, "q")
    if not query_text:
        return Response(
            status_code=400,
            headers={"Content-Type": "application/json"},
            description=json.dumps({"error": "Parámetro 'q' requerido."})
        )

    print(f"🔎 Pregunta: '{query_text}'")

    try:
        start_time = time.time()

        # 1. BÚSQUEDA RÁPIDA de documentos relevantes
        print(f"🔍 Buscando documentos...")
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Reducido a 3 para speed
        relevant_docs = retriever.invoke(query_text)

        print(f"📄 Encontrados {len(relevant_docs)} documentos relevantes")

        if len(relevant_docs) == 0:
            return Response(
                status_code=200,
                headers={"Content-Type": "application/json"},
                description=json.dumps({
                    "query": query_text,
                    "answer": "No se encontraron documentos relevantes para tu pregunta.",
                    "status": "no_docs"
                })
            )

        # 2. CREAR CONTEXTO CONCISO (limitar tamaño)
        context_parts = []
        max_context_length = 2000  # Limitar contexto para acelerar LLM
        current_length = 0

        for doc in relevant_docs:
            content = doc.page_content.strip()
            if current_length + len(content) > max_context_length:
                # Truncar el contenido si es necesario
                remaining_space = max_context_length - current_length
                if remaining_space > 100:  # Solo agregar si queda espacio suficiente
                    content = content[:remaining_space] + "..."
                    context_parts.append(content)
                break
            context_parts.append(content)
            current_length += len(content)

        context = "\n\n".join(context_parts)
        print(f"📝 Contexto creado ({len(context)} caracteres)")

        # 3. PROMPT OPTIMIZADO para respuestas rápidas
        optimized_prompt = f"""Documento: {context}

Pregunta: {query_text}

Respuesta breve (máximo 2 oraciones):"""

        # 4. INVOCAR LLM sin timeout
        print(f"🧠 Invocando LLM...")

        try:
            # Llamada directa al LLM (más rápida que la cadena completa)
            llm_response = llm.invoke(optimized_prompt)

        except Exception as llm_error:
            print(f"❌ Error en LLM: {llm_error}")
            llm_response = "Error procesando la consulta."

        end_time = time.time()
        processing_time = end_time - start_time

        print(f"✅ Respuesta en {processing_time:.2f}s: '{llm_response[:100]}...'")

        # 5. RESPUESTA RÁPIDA con información útil
        if not llm_response or llm_response.strip() in ["No lo sé.", "No sé.", ""]:
            # Fallback: información básica del documento
            llm_response = f"Encontré información en el documento. El contenido incluye: {context[:200]}..."

        response_data = {
            "query": query_text,
            "answer": llm_response.strip(),
            "status": "success",
            "docs_found": len(relevant_docs),
            "processing_time": f"{processing_time:.2f}s",
            "context_length": len(context)
        }

        return Response(
            status_code=200,
            headers={"Content-Type": "application/json"},
            description=json.dumps(response_data)
        )

    except Exception as query_error:
        print(f"❌ Error en consulta: {str(query_error)}")

        return Response(
            status_code=500,
            headers={"Content-Type": "application/json"},
            description=json.dumps({
                "error": f"Error: {str(query_error)}",
                "status": "error"
            })
        )


# ENDPOINT RÁPIDO PARA INFORMACIÓN DEL DOCUMENTO
@app.get("/doc-info")
def document_info(request: Request):
    """Endpoint rápido para obtener información básica del documento"""
    try:
        if not vector_store:
            return Response(
                status_code=503,
                headers={"Content-Type": "application/json"},
                description=json.dumps({"error": "Vector store no disponible"})
            )

        # Búsqueda simple para obtener metadatos
        sample_docs = vector_store.similarity_search("", k=5)

        if not sample_docs:
            return Response(
                status_code=200,
                headers={"Content-Type": "application/json"},
                description=json.dumps({
                    "total_docs": 0,
                    "message": "No hay documentos cargados"
                })
            )

        # Analizar metadatos y contenido
        doc_info = {
            "total_chunks": len(sample_docs),
            "sample_content": sample_docs[0].page_content[:300] + "..." if sample_docs[0].page_content else "",
            "metadata": sample_docs[0].metadata if hasattr(sample_docs[0], 'metadata') else {},
            "content_preview": []
        }

        # Agregar preview de varios chunks
        for i, doc in enumerate(sample_docs[:3]):
            preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
            doc_info["content_preview"].append({
                "chunk": i + 1,
                "preview": preview
            })

        return Response(
            status_code=200,
            headers={"Content-Type": "application/json"},
            description=json.dumps(doc_info, indent=2)
        )

    except Exception as e:
        return Response(
            status_code=500,
            headers={"Content-Type": "application/json"},
            description=json.dumps({"error": str(e)})
        )


# ENDPOINT PARA PREGUNTAS PREDEFINIDAS RÁPIDAS
@app.get("/quick-questions")
def quick_questions(request: Request):
    """Endpoint con preguntas pre-optimizadas para el documento"""

    quick_queries = [
        "¿De qué trata este documento?",
        "¿Cuál es el tema principal?",
        "¿Qué métodos se describen?",
        "¿Cuáles son las conclusiones principales?"
    ]

    try:
        if not vector_store:
            return Response(
                status_code=503,
                headers={"Content-Type": "application/json"},
                description=json.dumps({"error": "Vector store no disponible"})
            )

        # Obtener muestra del contenido para generar respuestas rápidas
        sample_docs = vector_store.similarity_search("research paper", k=2)

        if not sample_docs:
            return Response(
                status_code=200,
                headers={"Content-Type": "application/json"},
                description=json.dumps({
                    "suggested_questions": quick_queries,
                    "message": "No hay documentos cargados"
                })
            )

        # Crear respuesta con información básica
        content_sample = sample_docs[0].page_content[:500] if sample_docs else ""

        quick_answers = {
            "suggested_questions": quick_queries,
            "document_type": "Research Paper/Academic Document",
            "content_sample": content_sample,
            "tips": [
                "Pregunta sobre 'métodos' o 'methodology'",
                "Pregunta sobre 'resultados' o 'results'",
                "Pregunta sobre 'conclusiones' o 'conclusions'",
                "Usa términos académicos específicos"
            ]
        }

        return Response(
            status_code=200,
            headers={"Content-Type": "application/json"},
            description=json.dumps(quick_answers, indent=2)
        )

    except Exception as e:
        return Response(
            status_code=500,
            headers={"Content-Type": "application/json"},
            description=json.dumps({"error": str(e)})
        )


@app.get("/health")
def health_check():
    """Endpoint para verificar el estado de la API"""
    status = {
        "status": "healthy",
        "chromadb": False,
        "ollama": False,
        "timestamp": None
    }

    try:
        # Verificar ChromaDB
        if vector_store and chroma_client:
            chroma_client.heartbeat()
            status["chromadb"] = True
            print("✅ ChromaDB conectado")
        else:
            print("❌ ChromaDB no disponible")
    except Exception as chroma_error:
        print(f"❌ Error verificando ChromaDB: {chroma_error}")
        status["chromadb_error"] = str(chroma_error)

    try:
        # Verificar Ollama (hacer una consulta simple)
        if llm:
            test_response = llm.invoke("Hello")
            if test_response:
                status["ollama"] = True
                print("✅ Ollama conectado")
            else:
                print("❌ Ollama no responde")
        else:
            print("❌ Ollama no disponible")
    except Exception as ollama_error:
        print(f"❌ Error verificando Ollama: {ollama_error}")
        status["ollama_error"] = str(ollama_error)

    # Agregar timestamp
    import datetime
    status["timestamp"] = datetime.datetime.now().isoformat()

    return Response(
        status_code=200,
        headers={"Content-Type": "application/json"},
        description=json.dumps(status)
    )


@app.get("/debug")
def debug_info():
    """Endpoint para información de debug"""
    debug_data = {
        "vector_store": vector_store is not None,
        "llm": llm is not None,
        "chroma_client": chroma_client is not None if 'chroma_client' in globals() else False,
        "collection_name": COLLECTION_NAME,
        "ollama_base_url": OLLAMA_BASE_URL,
        "ollama_embedding_model": OLLAMA_EMBEDDING_MODEL,
        "ollama_generation_model": OLLAMA_GENERATION_MODEL,
        "chroma_host": CHROMA_HOST,
        "chroma_port": CHROMA_PORT,
    }

    return Response(
        status_code=200,
        headers={"Content-Type": "application/json"},
        description=json.dumps(debug_data, indent=2)
    )


if __name__ == "__main__":
    print("🚀 Iniciando servidor Robyn...")
    print(f"🔧 ChromaDB: {CHROMA_HOST}:{CHROMA_PORT}")
    print(f"🤖 Ollama: {OLLAMA_BASE_URL}")

    # Verificación final del estado DESPUÉS de todas las inicializaciones
    vector_store_status = vector_store is not None
    llm_status = llm is not None

    print(f"📊 Estado final - vector_store: {vector_store_status}, llm: {llm_status}")

    # Debug adicional
    if not llm_status:
        print("⚠️  ADVERTENCIA: LLM no inicializado. Verificar configuración de Ollama.")
        print(f"   Intentando conectar a: {OLLAMA_BASE_URL}")
        print(f"   Modelo esperado: {OLLAMA_GENERATION_MODEL}")

    if not vector_store_status:
        print("⚠️  ADVERTENCIA: Vector store no inicializado. Verificar configuración de ChromaDB.")

    if vector_store_status and llm_status:
        print("🎉 Todos los servicios inicializados correctamente!")

    app.start(host="0.0.0.0", port=8080)