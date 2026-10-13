# Claim Triage Assistant — explicación técnica

## 1. Qué problema resuelve

Una aseguradora quiere que sus peritos tramiten los siniestros de daños materiales más rápido y de forma más
consistente. Cada siniestro llega con una descripción, fotografías y documentos. El sistema analiza la
información, busca siniestros históricos parecidos y genera un **informe de triaje estructurado**. Bain exige
un **sistema multiagente con bucle ReAct**, una **base de conocimiento vectorial** de siniestros históricos y una
**interfaz web ligera**.

**Material ausente.** La carpeta descargada del assessment se extrajo sin archivos: todas sus subcarpetas
(`api-contract`, `auto-grader/tests`, `data/knowledge-base`, `ground-truth`, `test-images`, `test-documents`,
`tools`) están vacías. Por eso he usado `docs/correo.md` como especificación y he creado yo el resto: contrato
de API interno, base de conocimiento sintética, entradas de demo sintéticas, pruebas y conjunto de evaluación
propios. **No afirmo compatibilidad** con el contrato ni con el auto-grader de Bain.

## 2. Arquitectura elegida

- Backend en **Python + FastAPI**, interfaz de una página en **JavaScript sin frameworks**, todos los modelos en
  **Ollama** (local, gratuito, sin claves).
- **Cinco agentes** con responsabilidades claras; el **Coordinador** ejecuta el bucle ReAct.
- **Base vectorial**: 40 siniestros históricos sintéticos con embeddings de `nomic-embed-text` y búsqueda
  exacta por similitud coseno.
- Esquemas **Pydantic** para cada resultado intermedio y para el informe final.

```
UI ─► POST /api/claims ─► validación de entrada ─► tarea en segundo plano ─► Coordinador (ReAct)
                                                                         ├─ IntakeAnalyst (descripción / foto / documento)
                                                                         ├─ KnowledgeResearcher (búsqueda vectorial)
                                                                         ├─ EvidenceAuditor (comprobaciones de consistencia)
                                                                         └─ TriageAssessor (prioridad, confianza, narrativa)
UI ◄─ GET /api/claims/{job} (traza en vivo + TriageReport validado)
```

### Por qué este lenguaje y estas tecnologías

| Decisión | Motivo |
|---|---|
| Python | Mejor ecosistema para LLM, PDF e imágenes; lo más rápido de construir y probar. |
| FastAPI | Gestión tipada de formularios con archivos, OpenAPI automático (`/docs`), tareas en segundo plano sencillas. |
| Modelos locales en Ollama | Ya disponibles en el equipo, sin coste ni secretos, reproducible. Comprobé disponibilidad y velocidad antes de depender de ellos. |
| `qwen2.5:3b` para texto | Es el único que cabe entero en la GPU de 4 GB junto con las otras aplicaciones: ~8 s por llamada. `gemma3:4b` no cabe y era entre 10 y 50 veces más lento con texto. |
| `gemma3:4b` para fotos | Es el único modelo local que realmente interpreta imágenes. En este portátil se ejecuta en CPU (`VISION_NUM_GPU=0`): ~50 s por foto, frente a más de 10 min repartido entre GPU y CPU. |
| `nomic-embed-text` | Modelo de embeddings pequeño y bueno para recuperación; usa los prefijos `search_document:` / `search_query:`. |
| Búsqueda exacta con NumPy en lugar de FAISS/Chroma | Con decenas o miles de registros, un producto matriz-vector es instantáneo, determinista y sin dependencias (además, las ruedas de FAISS/Chroma para Python 3.14 eran un riesgo). Solo habría que cambiar la clase `VectorIndex` para usar FAISS o pgvector. |
| UI en JS sin frameworks | Sin compilación, nada que se rompa en la demo; solo muestra lo que devuelve el backend. |

## 3. Cómo funciona el sistema multiagente

| Agente | Herramientas | Qué hace | ¿LLM? |
|---|---|---|---|
| **IntakeAnalystAgent** | `analyze_description`, `analyze_image`, `extract_document` | Convierte las entradas en hechos estructurados: tipo de daño, causa, riesgos, habitabilidad, daño en curso, importe; en las fotos, daño visible, gravedad, riesgos y si la imagen parece sintética; en los documentos, texto (pypdf), importes y fechas (regex), tipo y hechos clave (LLM). | sí |
| **KnowledgeResearcherAgent** | `search_similar_claims` | Construye la consulta con *toda* la evidencia (declarada y visual), busca en el índice vectorial, explica cada resultado y resume los desenlaces de los precedentes. | solo embeddings |
| **EvidenceAuditorAgent** | `cross_check_evidence` | Comprobaciones deterministas: daño declarado frente a visible, si las fotos muestran daño, autenticidad de la foto, importe reclamado frente a documentos, fechas, documentos anteriores al siniestro, alertas en precedentes cercanos, entradas sin procesar. | no (a propósito) |
| **TriageAssessorAgent** | `assess_claim` | Rúbrica de prioridad e indicador de confianza deterministas; el LLM redacta la narrativa, que se valida y se filtra. | solo la narrativa |
| **CoordinatorAgent** | `finish` | Ejecuta el bucle ReAct y decide qué herramienta usar. | sí |

Las comprobaciones y la prioridad son deterministas porque deben ser reproducibles y explicables. El LLM se usa
donde hace falta comprensión del lenguaje o de imágenes.

## 4. Cómo se implementa el bucle ReAct (`claim_triage/agents/coordinator.py`)

En cada iteración (máximo 12):
1. **Observar**: se calculan las acciones válidas a partir de las precondiciones de cada herramienta.
2. **Razonar**: el LLM recibe un resumen compacto del estado, la última observación y el menú de acciones
   permitidas, y responde en JSON `{reason, action, target}`.
3. **Actuar**: se ejecuta la herramienta, que modifica el estado compartido (`ClaimState`).
4. **Registrar**: se guarda un `TraceStep` con agente, acción, objetivo, razón, quién decidió, estado,
   observación y duración. La UI lo muestra en directo.

Detalles clave:
- **Selección real**: el modelo elige órdenes distintos (p. ej., foto antes que descripción). Hay pruebas que
  verifican que se ejecuta la elección del planificador.
- **Precondiciones**: la búsqueda necesita la descripción analizada; la evaluación necesita al menos una búsqueda
  y una comprobación actualizada; `finish` necesita una evaluación actualizada.
- **Evidencia obsoleta**: cada nueva evidencia incrementa `evidence_version`; una búsqueda, comprobación o
  evaluación hecha con una versión anterior vuelve a estar disponible. Así el agente decide **repetir la
  búsqueda** si una foto revela información nueva.
- **Límites y salvaguardas**: el `enum` del esquema JSON restringe la acción al menú y luego se vuelve a validar.
  Si la acción es válida pero el objetivo no (visto en la práctica: `analyze_description` con el nombre de una
  imagen), se corrige y se marca `[guardrail …]`. En cualquier otro caso, o si falla el LLM, decide la política
  determinista (`fallback_rules`). Si solo hay una acción válida no se llama al LLM. Máximo 2 búsquedas y 2
  evaluaciones; una búsqueda fallida se reintenta una vez.
- **Finalización controlada**: si se alcanza el límite de pasos, se genera igualmente un informe con la evidencia
  disponible, `termination = max_steps_reached` y se exige revisión humana.
- `COORDINATOR_MODE=rules` ejecuta el mismo bucle solo con la política determinista (línea base).

La traza guarda razones **operativas** (por qué esta acción ahora), no razonamientos privados del modelo.

## 5. Cómo funciona la base de conocimiento vectorial

- **Datos**: `data/knowledge_base/historical_claims.json`, 40 siniestros **sintéticos** (`"synthetic": true`) de
  agua, inundación, viento, granizo, cubierta, incendio, humo, estructura, moho y vandalismo. Incluyen ID,
  descripción, causa, tipo de inmueble, evidencias, gravedad, prioridad, importes, desenlace (aprobado, parcial,
  denegado, derivado a investigación), notas, días de cierre y alertas (`gradual_damage`, `fraud_indicators`,
  `electrical_hazard`…). Algunos son casos "difíciles" a propósito para que los precedentes aporten avisos útiles.
- **Representación**: cada siniestro se convierte en un texto canónico y se vectoriza con el prefijo
  `search_document:`.
- **Índice**: matriz float32 normalizada en `data/index/vectors.npy` y `manifest.json` (modelo, dimensiones, hash
  del dataset). Se reconstruye sola si cambian los datos o el modelo.
- **Búsqueda**: similitud coseno, top 5, umbral **0,63**, elegido midiendo: las consultas relevantes dan entre
  0,64 y 0,92 y las ajenas al tema, ≤ 0,61.
- **Evidencias mostradas**: ID, similitud, tipo, gravedad, desenlace, importe, días de cierre, alertas y
  "por qué es similar" (mismo tipo, mismo inmueble, conceptos compartidos). Esa explicación es una capa aparte:
  el ranking es solo por similitud de embeddings.

## 6. Cómo se analizan las entradas

- **Validación**: descripción de al menos 20 caracteres; hasta 5 imágenes (jpg/png/webp) y 5 documentos
  (pdf/txt/md) de hasta 10 MB. Los tipos no admitidos o los archivos vacíos se rechazan (HTTP 422) y la UI
  lista todos los errores. Las imágenes corruptas se aceptan con aviso y aparecen como limitación.
- **Descripción**: LLM con esquema JSON y definiciones explícitas. **Anclaje**: si el modelo extrae un importe o
  una fecha que no aparece en el texto, se descarta y se indica en las incertidumbres.
- **Fotos**: el modelo de visión describe solo lo visible y señala si la imagen parece dibujada (lo hace con
  nuestras imágenes de demo).
- **Documentos**: pypdf, expresiones regulares para importes y fechas, LLM para tipo y hechos clave. Los PDF sin
  capa de texto se notifican como "sin texto extraíble (no hay OCR)".

## 7. Cómo se genera el informe de triaje

`TriageReport` (Pydantic, versión 1.0) se muestra en la UI y se descarga en JSON. Contiene: resumen, tipo de
daño, **hechos con categoría y fuente** (declarado, observado, documento, histórico, inferido), análisis de fotos
y documentos, siniestros similares con similitud y motivo, conclusiones históricas, comprobaciones, evaluación
preliminar (marcada como *inferencia*), prioridad, recomendaciones (marcadas como *recomendación*),
información a solicitar, incertidumbres, confianza, necesidad de revisión humana con sus motivos, origen de la
narrativa, estado de finalización, traza completa y aviso legal.

**Prioridad** (rúbrica determinista, cada factor con su fuente): riesgo para personas +3 · inhabitable +3
(parcialmente +1) · daño en curso +2 · gravedad visible crítica +3 / grave +2 / moderada +1 (sin foto: gravedad del
precedente más cercano −1, marcada como inferida) · pérdida ≥ 25.000 € +2 / ≥ 5.000 € +1 · ocupantes vulnerables +1.
**P1 ≥ 6** (respuesta en 24 h) · **P2 4–5** (contacto en 48 h) · **P3 2–3** (cola estándar) · **P4 0–1** (vía rápida).

**Confianza**: media de cinco componentes (cobertura de entradas, acuerdo entre evidencias, apoyo de precedentes,
completitud de hechos y evidencia visual). Es un **indicador heurístico, no una probabilidad calibrada**. Las
contradicciones lo limitan: *low* si se superan menos de la mitad de las comprobaciones aplicables y como máximo
*medium* si alguna falla.

**Revisión humana**: se recomienda en los casos P1, si falla alguna comprobación, ante avisos clave (daño declarado
frente a visible, importe, fechas de documentos, alertas en precedentes, autenticidad de la foto, entradas sin
procesar), con confianza baja o si la ejecución termina antes de tiempo.

**Limpieza de la salida**: se eliminan los valores de relleno como `"unknown"` de las listas extraídas (si no,
`safety_hazards: ["unknown"]` sumaría +3) y las frases repetidas de la narrativa.

**Narrativa segura**: debe cumplir el esquema, no puede citar siniestros no recuperados ni contener decisiones
de cobertura o pago. Si falla, se reintenta una vez con el error y, si vuelve a fallar, se usa una plantilla
determinista. El sistema **nunca decide** cobertura, responsabilidad ni indemnización.

## 8. Cómo ejecutar el proyecto

```bash
ollama pull qwen2.5:3b && ollama pull gemma3:4b && ollama pull nomic-embed-text
python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt
python run.py                    # http://localhost:8000
```

Por línea de comandos: `python scripts/run_case.py data/demo/01_ceiling_leak`. Configuración en
`claim_triage/config.py` y `.env.example`. Demo paso a paso: `DOC/Demo_Instructions.md`.

## 9. Qué pruebas he realizado

| Prueba | Comando | Resultado |
|---|---|---|
| Unitarias y de API (modelos simulados) | `pytest` | **52 superadas** |
| Integración con Ollama real | `pytest -m ollama` | **6 superadas** |
| Lint y formato | `ruff check .`, `ruff format --check .` | sin errores |
| Evaluación de recuperación (12 consultas parafraseadas, datos propios) | `python scripts/run_eval.py` | hit@3 = 1,00; MRR@5 = 1,00; precisión de tipo@3 = 0,72; consultas ajenas rechazadas 3/3 |
| Evaluación de extremo a extremo (4 casos demo, coordinador LLM) | `python scripts/run_eval.py --e2e` | **4/4 casos** superan todas sus comprobaciones; entre 6 y 31 s por caso en la ejecución final; 0 decisiones de respaldo (`output/eval_results_e2e.json`) |
| Prueba manual de la UI en el navegador | — | error de descripción vacía, carga del caso demo, traza en vivo e informe completo |

Los conjuntos de evaluación son pequeños y propios: demuestran que el sistema se comporta como se ha diseñado,
**no** son una medición sobre siniestros reales.

## 10. Limitaciones

- Modelos locales pequeños: cometen errores de extracción (observado: `ongoing_damage = true` aunque se cortó el
  agua; la visión a veces "ve" grietas inexistentes). Quedan a la vista porque cada punto de prioridad indica su
  fuente y se marcan revisiones.
- La latencia depende del hardware: ~30–70 s por siniestro multimodal en el portátil de desarrollo, sobre todo
  por la foto.
- Solo datos sintéticos; las imágenes de demo son dibujos, no fotografías.
- Sin OCR, sin autenticación, tareas en memoria, un solo trabajador, prompts en inglés.
- No se ha podido verificar la compatibilidad con el contrato, la ground truth ni el auto-grader de Bain.

## 11. Decisiones más importantes

1. **El LLM decide el flujo y las reglas lo protegen**: ReAct real con precondiciones, elección restringida por
   esquema, política de respaldo y límite de pasos.
2. **Prioridad y comprobaciones deterministas; LLM solo para lenguaje y visión**: misma evidencia, misma
   prioridad, y cada punto es trazable.
3. **Cada hecho indica su origen** y los números no respaldados se eliminan.
4. **Búsqueda vectorial exacta con índice persistente que se invalida solo**: lo más simple y correcto a esta escala.
5. **Todo local**, elegido tras medir el hardware, con la parte lenta (visión) aislada y configurable.
6. **Honestidad por diseño**: datos sintéticos señalados, confianza no presentada como probabilidad y el informe
   no decide coberturas.

**Para producción**: modelos más grandes o en la nube detrás de `LLMClient`, OCR, base vectorial con filtros,
cola y almacenamiento persistentes, autenticación y protección de datos personales, calibración de la rúbrica
con decisiones reales de peritos y un conjunto de evaluación con siniestros reales etiquetados.
