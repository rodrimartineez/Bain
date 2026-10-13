# Checklist final

Basada en los requisitos del correo de Bain (`docs/correo.md`). Estados: **Verificado** (lo he comprobado
ejecutándolo), **Parcial** (implementado con alguna limitación o verificación incompleta), **Pendiente** (falta
hacerlo).

Fecha de la verificación: 2026-10-13. Equipo: Windows 11, Python 3.14.3, Ollama 0.40.3, RTX 3050 Ti de 4 GB
compartida con otras aplicaciones.

## A. Material del assessment

| Requisito | Implementación | Evidencia | Estado |
|---|---|---|---|
| Leer el correo original | `docs/correo.md` (sin modificar) | Leído completo; requisitos extraídos al principio | Verificado |
| Usar el material descargado | `software-engineering-take-home/` | Todas sus carpetas están **vacías** (lo comprobé, también los archivos ocultos); no había ningún zip en el equipo | Verificado (material ausente) |
| No inventar datos de Bain | Datos propios marcados como sintéticos | `"synthetic": true` en cada registro; avisos en la UI, el README y los informes | Verificado |
| Compatibilidad con el contrato de API y el auto-grader de Bain | — | Imposible de comprobar: no existen. Contrato propio en `/docs` (OpenAPI) | Pendiente (no se puede resolver sin el material) |

## B. Requisitos técnicos del encargo

| Requisito | Implementación | Prueba o evidencia | Estado |
|---|---|---|---|
| Sistema multiagente | 5 agentes en `claim_triage/agents/` (intake, research, auditor, assessor, coordinator) | `tests/test_coordinator.py`; la traza muestra el agente en cada paso | Verificado |
| Bucle ReAct real | `CoordinatorAgent.run` + `LLMPolicy` | Pruebas: se ejecuta la elección del LLM, acción inválida → respaldo, caída del planificador, corrección de objetivo, repetición de búsqueda, límite de pasos. Ejecuciones reales: 4 decisiones del LLM por caso y 0 de respaldo (`output/eval_results_e2e.json`) | Verificado |
| Selección de herramientas según el estado | Precondiciones y `evidence_version` | `test_research_reruns_when_new_evidence_arrives_after_search`, `test_llm_choice_is_respected` | Verificado |
| Límites contra bucles | 12 pasos máximo, 2 búsquedas, 2 evaluaciones, sin repetir acciones | `test_step_limit_terminates_in_a_controlled_way`, `test_no_action_is_repeated_unnecessarily`, `test_vector_search_outage_is_bounded` | Verificado |
| Base de conocimiento vectorial | `knowledge_base.py`, 40 siniestros sintéticos, `nomic-embed-text` (768 dimensiones), similitud coseno exacta | `tests/test_knowledge_base.py`; `pytest -m ollama` (6 superadas); evaluación de recuperación: hit@3 = 1,00, MRR@5 = 1,00 | Verificado |
| Recuperación conectada al flujo y visible en el informe | `search_similar_claims` → `similar_claims` e `historical_insights` | Visto en la UI: HC-2024-0040 (0,91) con su explicación | Verificado |
| Análisis de descripciones | `analyze_description` (qwen2.5:3b, esquema JSON y anclaje de importes y fechas) | Ejecuciones reales; `test_hallucinated_amount_is_dropped` | Verificado (con errores ocasionales del modelo, ver G) |
| Análisis de fotografías | `analyze_image` (gemma3:4b con visión) | Ejecuciones reales: mancha → water_leak, tejado → roof, suelo intacto → sin daño; las tres se marcan como sintéticas | Verificado (con errores ocasionales del modelo) |
| Análisis de documentos | `extract_document` (pypdf, regex y LLM) | Importes `[1150, 1620, 780, 800, 4350]` extraídos del PDF demo; PDF sin texto detectado | Verificado (sin OCR) |
| Informe estructurado de triaje | `TriageReport` (Pydantic) | `test_report_has_disclaimer_and_is_json_serialisable`; descarga JSON desde la UI | Verificado |
| Interfaz web ligera | `web/` (FastAPI y JS sin frameworks) | Probada en el navegador: error de validación, caso demo, traza en directo, informe, ejecuciones anteriores y vista de móvil sin scroll horizontal | Verificado |

## C. Calidad del informe

| Requisito | Implementación | Evidencia | Estado |
|---|---|---|---|
| Distinguir hechos, declaraciones, inferencias y recomendaciones | `Fact.category`; etiquetas *inference* y *recommendation* en la UI | Informes de los casos 01 y 03 | Verificado |
| Prioridad con una escala definida | Rúbrica determinista P1–P4 con factores y fuentes | `test_priority_factors_are_traceable`, `test_high_risk_claim_is_p1`, `test_minor_text_only_claim_is_low_priority` | Verificado |
| Confianza no presentada como probabilidad | `Confidence.note`, limitada si hay contradicciones | Caso 03: *low*; caso 01: *high* | Verificado |
| Incertidumbres y limitaciones | `uncertainties`, `human_review_reasons` | Caso con foto corrupta y PDF escaneado probado contra el servidor real | Verificado |
| Sin decisiones de cobertura o indemnización | Filtro `FORBIDDEN` y aviso legal | `test_narrative_with_coverage_decision_is_rejected` | Verificado |
| No depender de que el LLM genere JSON correcto | Esquema en la decodificación, Pydantic, reintento y plantilla | `test_invalid_narrative_json_retries_then_uses_template`, `test_narrative_llm_outage_uses_template` | Verificado |

## D. Robustez y casos negativos

| Caso | Comportamiento | Evidencia | Estado |
|---|---|---|---|
| Descripción vacía | 422 con un mensaje claro | UI y `test_empty_or_short_description_rejected` | Verificado |
| Archivo no compatible o vacío | 422 con todos los errores | `curl` contra el servidor real y `test_unsupported_and_empty_files_rejected_with_all_errors` | Verificado |
| Imagen corrupta | Se acepta con aviso; la herramienta falla de forma controlada | Servidor real: paso `error` y limitación en el informe | Verificado |
| Documento sin texto extraíble | Limitación "no extractable text (no OCR)" | Servidor real y `test_corrupt_image_and_textless_pdf_become_limitations` | Verificado |
| Fallo de una herramienta | Se registra y la ejecución continúa | `test_failing_vision_tool_is_recorded_and_run_continues` | Verificado |
| Sin casos históricos relevantes | Umbral 0,63 → lista vacía y limitación | `test_threshold_returns_no_results_for_unrelated_query`; evaluación: 3/3 consultas ajenas rechazadas | Verificado |
| Respuesta del modelo con formato inválido | Respaldo (planificador) o plantilla (narrativa) | `test_invalid_llm_action_falls_back_to_rules` y pruebas de narrativa | Verificado |
| Límite de iteraciones | `max_steps_reached` con informe parcial y revisión humana | `test_step_limit_terminates_in_a_controlled_way` | Verificado |
| Ollama caído | Cabecera en rojo y 503 al enviar | Implementado en `web/main.py`; no probado apagando Ollama | Parcial |

## E. Comandos ejecutados y resultados

| Comando | Resultado |
|---|---|
| `pytest` | 52 superadas, 6 de integración excluidas (~1 s) |
| `pytest -m ollama` | 6 superadas con los modelos reales |
| `ruff check .` / `ruff format --check .` | Sin errores |
| `python scripts/run_eval.py --e2e --mode llm` | Recuperación: hit@3 = 1,00, MRR@5 = 1,00, precisión de tipo@3 = 0,72, consultas ajenas rechazadas 3/3. Extremo a extremo: **4/4 casos** superan sus comprobaciones (repetido con el código final: 31, 31, 25 y 6 s) |
| `python scripts/run_case.py data/demo/01_ceiling_leak` | P1, water_leak, narrativa del LLM, 54–69 s |
| Envío por la API real (casos 01 y 03) | Completados en 35 s y 30 s |

**Errores encontrados y corregidos durante la verificación:**
- gemma3 era demasiado lento con la GPU compartida → texto con qwen2.5:3b y visión en CPU.
- La narrativa se cortaba por el límite de tokens → límite ampliado e instrucciones más concisas.
- El LLM devolvía un objetivo erróneo para una acción válida → corrección del objetivo por la salvaguarda.
- Un `"unknown"` en la lista de riesgos sumaba 3 puntos → se filtran los valores de relleno.
- Frases repetidas en el resumen → limpieza determinista.
- Confianza *high* con evidencias contradictorias → límite según el acuerdo entre evidencias.
- Las entradas fallidas no pedían revisión humana → añadido como motivo de revisión.
- README en UTF-16 → convertido a UTF-8.

## F. Documentación y entrega

| Requisito | Implementación | Estado |
|---|---|---|
| README breve en inglés (enfoque y trade-offs) | `README.md` | Verificado |
| Código en inglés | Todo el código, los comentarios y la UI en inglés | Verificado |
| `DOC/Explicacion.md` (español) | Existe | Verificado |
| `DOC/Explanation.md` (inglés) | Existe | Verificado |
| `DOC/Guion.md` (inglés, ≤ 5 min) | 517 palabras ≈ 4:00 de narración (~4:15 con las pausas); las instrucciones de pantalla corresponden a funciones probadas | Verificado |
| `DOC/Demo_Instructions.md` (español) | Existe | Verificado |
| Sin secretos en el repositorio | No se necesita ninguna clave; `.env` ignorado por git; búsqueda de claves sin resultados | Verificado |
| `.gitignore`, `.env.example`, `requirements.txt` y versiones fijadas | Existen (`requirements.lock.txt` con las versiones probadas) | Verificado |
| **Vídeo de presentación en inglés (≤ 5 min)** | Guion e instrucciones listos. **No está grabado**: necesita tu voz. No he creado ningún archivo de vídeo | **Pendiente (lo tienes que grabar tú)** |
| Nombre del zip: `FirstName_LastName_Country_Solution` | Instrucciones en `Demo_Instructions.md` §6 | Pendiente (lo creas tú al entregar) |
| Nombre del vídeo: `FirstName_LastName_Country_Video_Presentation` | Ídem | Pendiente |
| Subir ambos archivos al enlace de Bain | Acción externa | Pendiente (lo haces tú) |

## G. Limitaciones conocidas (no ocultas)

- Errores de extracción de los modelos pequeños: por ejemplo, `ongoing_damage = true` aunque se cortó el agua, o
  "grietas" que la visión ve en una mancha. Son visibles en la traza y en los factores de prioridad.
- Latencia de 30–70 s por siniestro multimodal en este equipo.
- Las imágenes de demo son dibujos sintéticos (el propio sistema las marca así).
- Sin OCR, sin autenticación, tareas en memoria y un solo trabajador.
- Las evaluaciones usan datos propios y pequeños: no son una medición sobre siniestros reales.
