import {
  useState,
} from "react";

import type {
  CitedRAGResponse,
  CitationSource,
  GenerationResult,
} from "../api";

import {
  AnswerWithCitations,
} from "./AnswerWithCitations";

import "./CitedAnswer.css";

import {
  EvidenceInspector,
} from "./EvidenceInspector";

interface CitedAnswerProps {
  response: CitedRAGResponse | null;
  isQuerying: boolean;
  error: string | null;
  onDismissError: () => void;
}

export function CitedAnswer({
  response,
  isQuerying,
  error,
  onDismissError,
}: CitedAnswerProps) {
  const [
    selectedSourceId,
    setSelectedSourceId,
  ] = useState<string | null>(
    null,
  );

  if (isQuerying) {
    return (
      <section
        className="answer-status"
        aria-live="polite"
      >
        <div className="answer-loader" />

        <h3>
          Researching your question
        </h3>

        <p>
          Retrieving evidence,
          constructing a bounded
          context, and generating a
          grounded answer locally.
        </p>
      </section>
    );
  }

  if (error) {
    return (
      <section
        className="answer-error"
        role="alert"
      >
        <div>
          <strong>
            Research query failed
          </strong>

          <p>{error}</p>
        </div>

        <button
          type="button"
          onClick={
            onDismissError
          }
        >
          Dismiss
        </button>
      </section>
    );
  }

  if (!response) {
    return (
      <section className="answer-placeholder">
        <div className="answer-placeholder-icon">
          ⌕
        </div>

        <h3>
          Your answer will appear here
        </h3>

        <p>
          Supporting evidence,
          source pages, retrieval
          metadata, and validated
          citations will remain
          inspectable.
        </p>
      </section>
    );
  }

  if (
    response.rag.insufficient_evidence
  ) {
    return (
      <section
        className="insufficient-answer"
        aria-label="Insufficient evidence response"
      >
        <p className="answer-eyebrow">
          INSUFFICIENT EVIDENCE
        </p>

        <h3>
          The indexed evidence was
          not sufficient
        </h3>

        <p className="answer-text">
          {response.rag.answer}
        </p>

        <AnswerMetrics
          response={response}
        />
      </section>
    );
  }

  const selectedSource =
    response.citations.sources.find(
      (source) =>
        source.source_id ===
        selectedSourceId,
    ) ??
    response.citations.sources[0] ??
    null;

  const activeSourceId =
    selectedSource?.source_id ??
    null;

  return (
    <section
      className="cited-answer"
      aria-label="Cited research answer"
    >
      <header className="answer-header">
        <div>
          <p className="answer-eyebrow">
            GROUNDED ANSWER
          </p>

          <h3>
            Research response
          </h3>

          <p className="answer-question">
            {response.rag.question}
          </p>
        </div>

        <div className="answer-source-count">
          {
            response.citations
              .sources.length
          }{" "}
          {response.citations
            .sources.length === 1
            ? "source"
            : "sources"}
        </div>
      </header>

      <div className="answer-body">
        <AnswerWithCitations
          answer={
            response.rag.answer
          }
          references={
            response.citations
              .references
          }
          sources={
            response.citations
              .sources
          }
          activeSourceId={
            activeSourceId
          }
          onSelectSource={
            setSelectedSourceId
          }
        />
      </div>

      <AnswerMetrics
        response={response}
      />

      {response.citations.sources
        .length > 0 && (
        <section className="answer-sources">
          <div className="source-heading">
            <div>
              <p className="answer-eyebrow">
                VALIDATED SOURCES
              </p>

              <span className="source-heading-copy">
                Select a source to
                inspect the exact
                retrieved evidence.
              </span>
            </div>

            <span>
              {
                response.citations
                  .references.length
              }{" "}
              inline{" "}
              {response.citations
                .references.length ===
              1
                ? "reference"
                : "references"}
            </span>
          </div>

          <div className="source-list">
            {response.citations.sources.map(
              (source) => (
                <CitationSourceCard
                  key={
                    source.source_id
                  }
                  source={source}
                  selected={
                    activeSourceId ===
                    source.source_id
                  }
                  onSelect={() =>
                    setSelectedSourceId(
                      source.source_id,
                    )
                  }
                />
              ),
            )}
          </div>

          {selectedSource && (
            <EvidenceInspector
              source={
                selectedSource
              }
            />
          )}
        </section>
      )}

      {response.rag.generation && (
        <GenerationDiagnostics
          generation={
            response.rag.generation
          }
        />
      )}
    </section>
  );
}

interface AnswerMetricsProps {
  response: CitedRAGResponse;
}

function AnswerMetrics({
  response,
}: AnswerMetricsProps) {
  return (
    <div className="answer-metrics">
      <Metric
        label="Retrieved"
        value={String(
          response.rag
            .retrieved_count,
        )}
      />

      <Metric
        label="Used"
        value={String(
          response.rag
            .used_evidence_count,
        )}
      />

      <Metric
        label="Citations"
        value={String(
          response.citations
            .references.length,
        )}
      />

      <Metric
        label="Prompt"
        value={`${response.rag.actual_prompt_tokens} tokens`}
      />

      {response.rag
        .context_truncated && (
        <Metric
          label="Context"
          value="Truncated"
        />
      )}
    </div>
  );
}

interface MetricProps {
  label: string;
  value: string;
}

function Metric({
  label,
  value,
}: MetricProps) {
  return (
    <div className="answer-metric">
      <span>{label}</span>

      <strong>{value}</strong>
    </div>
  );
}

interface CitationSourceCardProps {
  source: CitationSource;
  selected: boolean;
  onSelect: () => void;
}

function CitationSourceCard({
  source,
  selected,
  onSelect,
}: CitationSourceCardProps) {
  const {
    evidence,
  } = source;

  return (
    <button
      className={`citation-source-card ${
        selected
          ? "selected"
          : ""
      }`}
      type="button"
      aria-label={`Inspect ${source.source_id} from ${evidence.chunk.file_name} page ${evidence.chunk.page_number}`}
      aria-pressed={selected}
      onClick={onSelect}
    >
      <span className="citation-source-id">
        {source.source_id}
      </span>

      <span className="citation-source-info">
        <strong
          title={
            evidence.chunk.file_name
          }
        >
          {
            evidence.chunk.file_name
          }
        </strong>

        <span>
          Page{" "}
          {
            evidence.chunk.page_number
          }{" "}
          · Retrieval rank{" "}
          {
            evidence.retrieval_rank
          }
        </span>
      </span>

      <span className="citation-source-score">
        <span>
          Score
        </span>

        <strong>
          {evidence.score.toFixed(
            3,
          )}
        </strong>
      </span>
    </button>
  );
}

interface GenerationDiagnosticsProps {
  generation: GenerationResult;
}

function GenerationDiagnostics({
  generation,
}: GenerationDiagnosticsProps) {
  const durationSeconds =
    generation.total_duration_ns /
    1_000_000_000;

  return (
    <details className="generation-diagnostics">
      <summary>
        Generation diagnostics
      </summary>

      <dl>
        <div>
          <dt>Model</dt>

          <dd>
            {generation.model_name}
          </dd>
        </div>

        <div>
          <dt>Prompt tokens</dt>

          <dd>
            {
              generation.prompt_tokens
            }
          </dd>
        </div>

        <div>
          <dt>Completion tokens</dt>

          <dd>
            {
              generation.completion_tokens
            }
          </dd>
        </div>

        <div>
          <dt>Total duration</dt>

          <dd>
            {durationSeconds.toFixed(
              2,
            )}{" "}
            s
          </dd>
        </div>

        {generation.done_reason && (
          <div>
            <dt>
              Completion reason
            </dt>

            <dd>
              {
                generation.done_reason
              }
            </dd>
          </div>
        )}
      </dl>
    </details>
  );
}