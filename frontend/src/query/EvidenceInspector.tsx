import type {
  CitationSource,
} from "../api";

interface EvidenceInspectorProps {
  source: CitationSource;
}

export function EvidenceInspector({
  source,
}: EvidenceInspectorProps) {
  const {
    evidence,
  } = source;

  const {
    chunk,
  } = evidence;

  return (
    <section
      className="evidence-inspector"
      aria-label={`Evidence for ${source.source_id}`}
    >
      <header className="evidence-inspector-header">
        <div>
          <p className="answer-eyebrow">
            EVIDENCE INSPECTOR
          </p>

          <h4>
            {source.source_id}
          </h4>
        </div>

        <div className="evidence-score">
          <span>
            Similarity
          </span>

          <strong>
            {evidence.score.toFixed(
              3,
            )}
          </strong>
        </div>
      </header>

      <div className="evidence-origin">
        <div>
          <span>
            Document
          </span>

          <strong
            title={
              chunk.file_name
            }
          >
            {chunk.file_name}
          </strong>
        </div>

        <div>
          <span>
            Page
          </span>

          <strong>
            {chunk.page_number}
          </strong>
        </div>

        <div>
          <span>
            Retrieval rank
          </span>

          <strong>
            {
              evidence.retrieval_rank
            }
          </strong>
        </div>
      </div>

      <div className="evidence-text-section">
        <div className="evidence-section-heading">
          <span>
            Retrieved evidence
          </span>

          <span>
            {chunk.token_count}{" "}
            {chunk.token_count === 1
              ? "token"
              : "tokens"}
          </span>
        </div>

        <blockquote className="evidence-text">
          {chunk.text}
        </blockquote>
      </div>

      <details className="evidence-metadata">
        <summary>
          Technical provenance
        </summary>

        <dl>
          <MetadataRow
            label="Prompt source"
            value={
              source.source_id
            }
          />

          <MetadataRow
            label="Document ID"
            value={
              chunk.document_id
            }
            mono
          />

          <MetadataRow
            label="Chunk ID"
            value={
              chunk.chunk_id
            }
            mono
          />

          <MetadataRow
            label="Global chunk"
            value={String(
              chunk.chunk_index +
                1,
            )}
          />

          <MetadataRow
            label="Page chunk"
            value={String(
              chunk.page_chunk_index +
                1,
            )}
          />

          <MetadataRow
            label="Token range"
            value={`${chunk.token_start}–${chunk.token_end}`}
          />

          <MetadataRow
            label="Characters"
            value={String(
              chunk.char_count,
            )}
          />
        </dl>
      </details>

      <p className="evidence-integrity-note">
        Source identity and
        provenance are resolved from
        validated application
        evidence. Citation validation
        does not by itself establish
        semantic entailment for every
        claim in the answer.
      </p>
    </section>
  );
}

interface MetadataRowProps {
  label: string;
  value: string;
  mono?: boolean;
}

function MetadataRow({
  label,
  value,
  mono = false,
}: MetadataRowProps) {
  return (
    <div>
      <dt>{label}</dt>

      <dd
        className={
          mono
            ? "metadata-mono"
            : undefined
        }
      >
        {value}
      </dd>
    </div>
  );
}