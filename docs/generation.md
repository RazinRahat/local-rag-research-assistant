# Local Language Generation

## Purpose

The generation layer provides local language-model inference behind a provider-independent application interface.

Before this phase, the project had a complete retrieval system:

```text
Question
   ↓
Query Embedding
   ↓
Semantic Retrieval
   ↓
RetrievalResult[]
```

but no language-generation capability.

Phase 6 introduces a separate generation pipeline:

```text
ChatMessage[]
     ↓
 LLMProvider
     ↓
Ollama Runtime
     ↓
Local Model
     ↓
GenerationResult
```

Retrieval and generation intentionally remain independent at this stage.

They will be connected through context construction and grounding during the end-to-end RAG phase.

---

## Why Generation Is a Separate Layer

The language model is not responsible for document retrieval.

Likewise, the retriever should not know how an LLM is hosted or invoked.

The project therefore maintains two distinct subsystems:

```text
Retrieval
---------
query
 ↓
embedding
 ↓
vector search
 ↓
evidence
```

and:

```text
Generation
----------
messages
 ↓
LLM runtime
 ↓
response
```

This separation makes failures easier to diagnose.

For example:

```text
wrong evidence
    → retrieval problem

correct evidence + wrong answer
    → generation / prompting problem

good answer but unsupported claim
    → grounding problem
```

A monolithic RAG abstraction would make these boundaries harder to inspect.

---

## Local Runtime

The baseline local inference runtime is Ollama.

Conceptually:

```text
Python Application
       │
       │ HTTP
       ▼
Ollama API
       │
       ▼
Local Language Model
```

The application communicates with the Ollama HTTP API rather than importing model-runtime behaviour directly into the RAG code.

This creates a clear process boundary between:

```text
application logic
```

and:

```text
model execution
```

---

## Baseline Model

The current development baseline is:

```text
qwen3.5:9b
```

The model is intended to provide a practical balance between:

* local execution
* technical instruction following
* research-oriented responses
* resource requirements
* response quality

The project architecture does not depend on this specific model.

The model can be replaced through generation configuration without rewriting downstream RAG logic.

---

## LLM Provider Abstraction

Application code depends on an `LLMProvider` interface.

Conceptually:

```text
LLMProvider
     │
     └── OllamaProvider
```

The provider currently exposes:

```text
model_name
chat()
close()
```

Future providers could include:

```text
LLMProvider
├── OllamaProvider
├── OpenAIProvider
├── AnthropicProvider
└── other local runtimes
```

The later RAG pipeline therefore does not need to know whether the model is local or remote.

---

## Chat Representation

Generation uses typed chat messages instead of concatenating arbitrary strings.

A message contains:

```text
role
content
```

Supported roles currently include:

```text
system
user
assistant
```

Example:

```text
system
"You are a concise technical research assistant."

user
"Explain semantic retrieval."
```

This structure becomes important when the RAG layer later adds:

* grounding instructions
* evidence delimiters
* answer constraints
* source-handling instructions

---

## Generation Result

The provider returns a normalised `GenerationResult` rather than exposing raw Ollama response objects.

The result currently preserves:

```text
content
model_name
thinking
done_reason
prompt_tokens
completion_tokens
total_duration_ns
load_duration_ns
prompt_eval_duration_ns
generation_duration_ns
```

This provides both:

```text
generated answer
```

and:

```text
inference metadata
```

without leaking the runtime-specific response schema into higher application layers.

---

## Generation Configuration

The initial generation baseline uses conservative settings appropriate for a technical research assistant.

Conceptually:

```text
temperature        0.2
top_p              0.9
context_size       8192
max_output_tokens   512
seed                42
thinking           disabled
keep_alive          5m
```

These values are baselines rather than final optimisation choices.

---

## Runtime Context vs Model Capability

A model may support a very large theoretical context window.

That does not mean the application should allocate the maximum possible context locally.

The project therefore explicitly configures a smaller runtime context:

```text
8192 tokens
```

This provides a manageable initial budget for:

```text
system instructions
+
user question
+
retrieved evidence
+
generated answer
```

The later context-construction layer will treat this as a finite resource.

Conceptually:

```text
Context Budget
│
├── system instructions
├── user query
├── retrieved evidence
└── reserved generation space
```

The system should not blindly concatenate arbitrary numbers of retrieved chunks.

---

## Output Budget

Generation is currently limited to approximately:

```text
512 output tokens
```

This prevents uncontrolled responses from consuming the full available context or producing unnecessarily long research answers.

The output allowance will later become part of the context-budget calculation.

---

## Temperature

The current temperature is deliberately low:

```text
0.2
```

The project is oriented toward:

```text
evidence synthesis
technical explanation
research assistance
```

rather than creative text generation.

A lower temperature generally reduces variation and provides a more stable baseline for later evaluation.

This is still an experimental configuration rather than a claim of optimality.

---

## Seed

A fixed seed is currently supplied as a reproducibility aid.

This may reduce response variation across similar runs.

It should not be interpreted as a guarantee of identical output across:

* runtime versions
* hardware
* model revisions
* quantisation changes
* inference backends

The seed supports reproducibility but does not define it absolutely.

---

## Thinking Output

The initial generation baseline disables explicit thinking output.

The immediate objective is:

```text
input
 ↓
model
 ↓
answer
```

rather than introducing an additional reasoning-output channel.

If reasoning modes later prove useful for complex research synthesis, they can be evaluated experimentally.

They are not required for the baseline local generation architecture.

---

## Non-Streaming Baseline

The Ollama adapter currently requests non-streaming responses.

```text
request
   ↓
model inference
   ↓
complete JSON response
```

This intentionally avoids introducing:

* partial message assembly
* asynchronous streaming
* cancellation handling
* client disconnect behaviour
* streaming FastAPI responses

before the basic generation interface is stable.

Streaming may be added later as a user-interface optimisation.

---

## HTTP Boundary

The Ollama provider uses HTTPX to communicate with the local runtime.

Conceptually:

```text
OllamaProvider
     ↓
HTTP POST
     ↓
/api/chat
     ↓
Ollama
```

The request includes:

```text
model
messages
stream configuration
thinking configuration
keep-alive behaviour
generation options
```

The response is validated before being converted into the application's generation-domain model.

---

## Response Validation

Responses from Ollama are treated as external data.

The application does not pass raw response dictionaries further into the system.

Instead:

```text
Ollama response
      ↓
Pydantic validation
      ↓
GenerationResult
```

The adapter checks for conditions such as:

* malformed responses
* incomplete non-streaming responses
* empty generated content
* HTTP failures
* connection failures

These cases are converted into explicit generation-domain exceptions.

---

## Error Model

The generation layer currently distinguishes between:

```text
LLMConnectionError
```

and:

```text
LLMResponseError
```

A connection error may indicate:

* Ollama is not running
* the configured host is unavailable
* a network-level request failed

A response error may indicate:

* HTTP failure
* malformed runtime response
* incomplete generation
* empty output

Future API layers can translate these into cleaner user-facing errors rather than exposing raw HTTPX exceptions.

---

## Resource Management

The provider owns a reusable HTTP client.

This allows connection reuse across multiple local generation requests.

The provider therefore exposes:

```text
close()
```

so application resources can be released explicitly.

Future application lifecycle management may initialise the provider once and close it during server shutdown.

---

## Keep-Alive Behaviour

The model runtime is currently configured to keep the model loaded temporarily after inference.

Conceptually:

```text
first request
    ↓
load model
    ↓
generate
    ↓
keep model resident

second request
    ↓
reuse loaded model
```

This can significantly affect perceived latency.

A first request may include model-loading cost while subsequent requests may primarily reflect prompt processing and token generation.

This distinction is why generation timing metadata is preserved.

---

## Token Accounting

The provider records:

```text
prompt_tokens
completion_tokens
```

These measurements will become increasingly important once RAG context is introduced.

A future request might look like:

```text
System instructions      250 tokens
Question                  40 tokens
Retrieved evidence      3100 tokens
Generated answer         380 tokens
```

Without token accounting, context-window behaviour becomes difficult to inspect or optimise.

---

## Timing Metrics

The generation result also preserves timing data.

Examples include:

```text
total duration
model load duration
prompt evaluation duration
generation duration
```

Eventually a RAG request can be decomposed into:

```text
retrieval latency
+
context construction
+
model loading
+
prompt prefill
+
token generation
```

rather than reporting only one opaque end-to-end latency value.

This forms an early foundation for later observability work.

---

## Testing Strategy

The generation layer is tested without requiring a local model during normal unit tests.

### Model Tests

Pydantic models are tested for:

* valid chat messages
* whitespace normalisation
* empty-message rejection
* configuration defaults

### Provider Tests

HTTPX mock transports simulate the Ollama API.

This verifies:

```text
application messages
      ↓
HTTP request
      ↓
mock response
      ↓
response validation
      ↓
GenerationResult
```

without requiring:

* Ollama to be running
* a downloaded model
* GPU/accelerator resources
* network access

---

## Real-Model Smoke Testing

The local model is validated separately through a manual smoke test.

The test verifies that:

* Ollama can be reached
* the configured model loads
* system and user messages are accepted
* text is generated successfully
* response metadata is parsed
* prompt-token counts are reported
* completion-token counts are reported
* latency values are available

The real-model smoke test validates integration.

It is not intended to replace unit tests.

---

## Why Retrieval Is Not Connected Yet

Phase 6 intentionally does not inject retrieved document chunks into the LLM.

Connecting retrieval introduces a separate set of design questions:

```text
How many chunks should be included?

How should sources be delimited?

How should overlapping chunks be handled?

How should the evidence be ordered?

How much context can be used?

What happens when evidence is weak?

What instructions enforce grounding?

How should unsupported questions be answered?
```

These are not model-runtime questions.

They belong to the RAG orchestration and context-construction layer.

Keeping them separate prevents the Ollama adapter from becoming responsible for application-specific RAG logic.

---

## Current Architecture

At the end of Phase 6, two independent systems exist.

### Retrieval

```text
Question
   ↓
BGE Query Embedding
   ↓
Qdrant
   ↓
RetrievalResult[]
```

### Generation

```text
ChatMessage[]
   ↓
OllamaProvider
   ↓
Qwen
   ↓
GenerationResult
```

Phase 7 will connect them.

---

## Current Limitations

The generation layer currently does not implement:

* retrieved-evidence prompting
* context budgeting
* grounding verification
* citation generation
* response streaming
* structured-output schemas
* tool calling
* multiple LLM backends
* automatic model selection
* generation evaluation
* conversation persistence
* prompt versioning

These are intentionally outside the baseline local-generation layer.

---

## Next Step

The next phase will implement the first complete RAG pipeline:

```text
Question
   ↓
Semantic Retrieval
   ↓
Retrieved Chunks
   ↓
Context Construction
   ↓
Grounded Prompt
   ↓
Local LLM
   ↓
RAG Answer
```

The context builder will need to manage evidence selection, token budgets, source labels, grounding instructions, and insufficient-evidence behaviour before citations are introduced as a first-class feature.
