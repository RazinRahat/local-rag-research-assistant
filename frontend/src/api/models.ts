export interface HealthResponse {
  status: string;
  service: string;
}

export interface StoredDocument {
  document_id: string;
  file_name: string;
  chunk_count: number;
}

export interface DocumentChunk {
  chunk_id: string;
  document_id: string;
  file_name: string;
  page_number: number;
  chunk_index: number;
  page_chunk_index: number;
  text: string;
  token_count: number;
  char_count: number;
  token_start: number;
  token_end: number;
}

export interface RetrievalResult {
  rank: number;
  score: number;
  chunk: DocumentChunk;
}

export interface RetrievalOptions {
  top_k?: number;
  score_threshold?: number | null;
  document_id?: string | null;
}

export interface SearchRequest extends RetrievalOptions {
  query: string;
}

export interface QueryRequest extends RetrievalOptions {
  question: string;
}

export interface SearchResponse {
  query: string;
  results: RetrievalResult[];
}

export interface EvidenceBlock {
  source_id: string;
  retrieval_rank: number;
  score: number;
  chunk: DocumentChunk;
}

export interface GenerationResult {
  content: string;
  model_name: string;
  thinking: string | null;
  done_reason: string | null;
  prompt_tokens: number;
  completion_tokens: number;
  total_duration_ns: number;
  load_duration_ns: number;
  prompt_eval_duration_ns: number;
  generation_duration_ns: number;
}

export interface RAGResponse {
  question: string;
  answer: string;
  evidence: EvidenceBlock[];
  retrieved_count: number;
  used_evidence_count: number;
  estimated_prompt_tokens: number;
  actual_prompt_tokens: number;
  context_truncated: boolean;
  insufficient_evidence: boolean;
  generation: GenerationResult | null;
}

export interface CitationReference {
  source_id: string;
  start_index: number;
  end_index: number;
}

export interface CitationSource {
  source_id: string;
  evidence: EvidenceBlock;
}

export interface CitationValidationResult {
  references: CitationReference[];
  sources: CitationSource[];
}

export interface CitedRAGResponse {
  rag: RAGResponse;
  citations: CitationValidationResult;
}