import type {
  RetrievalResult,
} from "../api";

import type {
  RetrievalMode,
} from "../api/models";

interface SearchResultsProps {
  query: string | null;
  results: RetrievalResult[];
  isSearching: boolean;
  error: string | null;
  onDismissError: () => void;

  /*
   * Optional so existing component
   * tests remain backward-compatible.
   */
  retrievalMode?: RetrievalMode;
}

function getModeLabel(
  mode: RetrievalMode,
): string {
  switch (mode) {
    case "dense":
      return "Dense";

    case "lexical":
      return "Lexical";

    case "hybrid":
      return "Hybrid";

    case "hybrid_reranked":
      return "Reranked";
  }
}

function getRankingDescription(
  mode: RetrievalMode,
): string {
  switch (mode) {
    case "dense":
      return "Ranked by semantic similarity";

    case "lexical":
      return "Ranked by BM25 relevance";

    case "hybrid":
      return "Ranked by reciprocal rank fusion";

    case "hybrid_reranked":
      return "Ranked by cross-encoder relevance";
  }
}

function getScoreTitle(
  mode: RetrievalMode,
): string {
  switch (mode) {
    case "dense":
      return "Cosine similarity score";

    case "lexical":
      return "BM25 retrieval score";

    case "hybrid":
      return "Reciprocal rank fusion score";

    case "hybrid_reranked":
      return "Cross-encoder relevance score";
  }
}

export function SearchResults({
  query,
  results,
  isSearching,
  error,
  onDismissError,
  retrievalMode = "dense",
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
          Retrieving the most relevant
          chunks using{" "}
          {getModeLabel(
            retrievalMode,
          ).toLowerCase()}{" "}
          retrieval.
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
          Ranked chunks, retrieval
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

  const ariaLabel =
    retrievalMode === "dense"
      ? "Semantic search results"
      : `${getModeLabel(
          retrievalMode,
        )} search results`;

  return (
    <section
      className="search-results"
      aria-label={ariaLabel}
    >
      <div className="results-heading">
        <div>
          <p className="eyebrow">
            {getModeLabel(
              retrievalMode,
            ).toUpperCase()}{" "}
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
          {getRankingDescription(
            retrievalMode,
          )}
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
              retrievalMode={
                retrievalMode
              }
            />
          ),
        )}
      </div>
    </section>
  );
}

interface SearchResultCardProps {
  result: RetrievalResult;
  retrievalMode: RetrievalMode;
}

function SearchResultCard({
  result,
  retrievalMode,
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
          title={
            getScoreTitle(
              retrievalMode,
            )
          }
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