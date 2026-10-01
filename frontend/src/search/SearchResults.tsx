import type {
  RetrievalResult,
} from "../api";

interface SearchResultsProps {
  query: string | null;
  results: RetrievalResult[];
  isSearching: boolean;
  error: string | null;
  onDismissError: () => void;
}

export function SearchResults({
  query,
  results,
  isSearching,
  error,
  onDismissError,
}: SearchResultsProps) {
  if (isSearching) {
    return (
      <section
        className="result-status"
        aria-live="polite"
      >
        <div className="result-loader" />

        <h3>
          Searching the index
        </h3>

        <p>
          Embedding your query and
          retrieving the most relevant
          chunks.
        </p>
      </section>
    );
  }

  if (error) {
    return (
      <section
        className="result-error"
        role="alert"
      >
        <div>
          <strong>
            Search failed
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

  if (query === null) {
    return (
      <section className="result-placeholder">
        <div className="placeholder-icon">
          ⌕
        </div>

        <h3>
          Search results will appear here
        </h3>

        <p>
          Ranked chunks, similarity
          scores, source documents,
          and page provenance will
          remain inspectable.
        </p>
      </section>
    );
  }

  if (results.length === 0) {
    return (
      <section className="result-placeholder">
        <div className="placeholder-icon">
          ∅
        </div>

        <h3>
          No matching evidence
        </h3>

        <p>
          No indexed chunks were
          returned for “{query}”.
        </p>
      </section>
    );
  }

  return (
    <section
      className="search-results"
      aria-label="Semantic search results"
    >
      <div className="results-heading">
        <div>
          <p className="eyebrow">
            RETRIEVAL
          </p>

          <h3>
            {results.length}{" "}
            {results.length === 1
              ? "result"
              : "results"}
          </h3>

          <p className="results-query">
            Results for “{query}”
          </p>
        </div>
        <span>
          Ranked by semantic
          similarity
        </span>
      </div>

      <div className="result-list">
        {results.map(
          (result) => (
            <SearchResultCard
              key={
                result.chunk.chunk_id
              }
              result={result}
            />
          ),
        )}
      </div>
    </section>
  );
}

interface SearchResultCardProps {
  result: RetrievalResult;
}

function SearchResultCard({
  result,
}: SearchResultCardProps) {
  const score =
    result.score.toFixed(3);

  return (
    <article className="search-result-card">
      <header className="result-card-header">
        <div className="result-rank">
          #{result.rank}
        </div>

        <div className="result-source">
          <strong
            title={
              result.chunk.file_name
            }
          >
            {
              result.chunk.file_name
            }
          </strong>

          <span>
            Page{" "}
            {
              result.chunk.page_number
            }
          </span>
        </div>

        <div
          className="result-score"
          title="Cosine similarity score"
        >
          <span>
            Score
          </span>

          <strong>
            {score}
          </strong>
        </div>
      </header>

      <p className="result-text">
        {result.chunk.text}
      </p>

      <footer className="result-card-footer">
        <span>
          {
            result.chunk.token_count
          }{" "}
          tokens
        </span>

        <span>
          Chunk{" "}
          {
            result.chunk.chunk_index +
            1
          }
        </span>
      </footer>
    </article>
  );
}