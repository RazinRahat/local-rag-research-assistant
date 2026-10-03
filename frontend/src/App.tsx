import {
  type KeyboardEvent,
  useState,
} from "react";

import "./App.css";

import type {
  RetrievalMode,
} from "./api/models";

import {
  DocumentSidebar,
  type DocumentClient,
  useDocuments,
} from "./documents";

import {
  CitedAnswer,
  type RAGQueryClient,
  useResearchQuery,
} from "./query";

import {
  SearchResults,
  type SearchClient,
  useSearch,
} from "./search";

type ResearchMode =
  | "ask"
  | "search";

interface AppProps {
  documentClient?: DocumentClient;
  searchClient?: SearchClient;
  queryClient?: RAGQueryClient;
}

function App({
  documentClient,
  searchClient,
  queryClient,
}: AppProps) {
  const {
    documents,
    isLoading,
    isUploading,
    deletingDocumentId,
    error: documentError,
    refreshDocuments,
    uploadDocument,
    deleteDocument,
    clearError:
      clearDocumentError,
  } = useDocuments(
    documentClient,
  );

  const {
    results: searchResults,
    lastQuery,
    isSearching,
    error: searchError,
    runSearch,
    clearSearch,
    clearError:
      clearSearchError,
  } = useSearch(
    searchClient,
  );

  const {
    response: queryResponse,
    isQuerying,
    error: queryError,
    runQuery,
    clearQuery,
    clearError:
      clearQueryError,
  } = useResearchQuery(
    queryClient,
  );

  const [
    selectedDocumentId,
    setSelectedDocumentId,
  ] = useState<string | null>(
    null,
  );

  const [
    mode,
    setMode,
  ] = useState<ResearchMode>(
    "ask",
  );

  /*
   * Dense remains the default so the
   * Phase 10 user experience and API
   * behaviour remain unchanged.
   */
  const [
    retrievalMode,
    setRetrievalMode,
  ] = useState<RetrievalMode>(
    "dense",
  );

  const [
    query,
    setQuery,
  ] = useState("");

  const selectedDocument =
    documents.find(
      (document) =>
        document.document_id ===
        selectedDocumentId,
    ) ?? null;

  const activeDocumentId =
    selectedDocument?.document_id ??
    null;

  const scopeDescription =
    selectedDocument
      ? selectedDocument.file_name
      : "All indexed documents";

  const hasDocuments =
    documents.length > 0;

  const researchUnavailable =
    isLoading || !hasDocuments;

  const searchDisabled =
    query.trim().length === 0 ||
    isSearching ||
    researchUnavailable;

  const queryDisabled =
    query.trim().length === 0 ||
    isQuerying ||
    researchUnavailable;

  const activeActionDisabled =
    mode === "ask"
      ? queryDisabled
      : searchDisabled;

  let researchHint: string;

  if (isLoading) {
    researchHint =
      "Loading the indexed document library.";
  } else if (!hasDocuments) {
    researchHint =
      "Index at least one PDF before searching or asking research questions.";
  } else if (mode === "ask") {
    researchHint =
      "Answers are grounded in retrieved evidence. Press Cmd/Ctrl + Enter to submit.";
  } else {
    researchHint =
      "Search returns ranked source chunks without invoking the LLM. Press Cmd/Ctrl + Enter to submit.";
  }

  function clearResearchResults(): void {
    /*
     * Both hooks increment their
     * request sequence here.
     *
     * That means changing scope or
     * retrieval strategy also
     * invalidates in-flight requests.
     */
    clearSearch();
    clearQuery();
  }

  function handleSelectDocument(
    documentId: string | null,
  ): void {
    setSelectedDocumentId(
      documentId,
    );

    clearResearchResults();
  }

  function handleRetrievalModeChange(
    nextMode: RetrievalMode,
  ): void {
    if (
      nextMode === retrievalMode
    ) {
      return;
    }

    setRetrievalMode(
      nextMode,
    );

    /*
     * An old dense result must never
     * remain visible after the UI says
     * "Hybrid", and vice versa.
     */
    clearResearchResults();
  }

  async function handleUpload(
    file: File,
  ): Promise<void> {
    const uploaded =
      await uploadDocument(file);

    if (!uploaded) {
      return;
    }

    setSelectedDocumentId(
      uploaded.document_id,
    );

    clearResearchResults();
  }

  async function handleDelete(
    documentId: string,
  ): Promise<void> {
    const deleted =
      await deleteDocument(
        documentId,
      );

    if (!deleted) {
      return;
    }

    if (
      activeDocumentId ===
      documentId
    ) {
      setSelectedDocumentId(
        null,
      );
    }

    clearResearchResults();
  }

  async function handleSearch(): Promise<void> {
    await runSearch(
      query,
      activeDocumentId,
      retrievalMode,
    );
  }

  async function handleQuery(): Promise<void> {
    await runQuery(
      query,
      activeDocumentId,
      retrievalMode,
    );
  }

  function handleQueryChange(
    value: string,
  ): void {
    setQuery(value);

    if (mode === "ask") {
      clearQueryError();
    } else {
      clearSearchError();
    }
  }

  function handleQueryKeyDown(
    event:
      KeyboardEvent<HTMLTextAreaElement>,
  ): void {
    const submitShortcut =
      (event.metaKey ||
        event.ctrlKey) &&
      event.key === "Enter";

    if (!submitShortcut) {
      return;
    }

    if (activeActionDisabled) {
      return;
    }

    event.preventDefault();

    if (mode === "ask") {
      void handleQuery();
    } else {
      void handleSearch();
    }
  }

  function getActionTitle():
    string | undefined {
    if (isLoading) {
      return "Loading the document library";
    }

    if (!hasDocuments) {
      return "Index at least one PDF first";
    }

    return undefined;
  }

  function getSearchLabel(): string {
    switch (retrievalMode) {
      case "dense":
        return "Semantic search";

      case "lexical":
        return "Lexical search";

      case "hybrid":
        return "Hybrid search";
    }
  }

  function getSearchPlaceholder(): string {
    switch (retrievalMode) {
      case "dense":
        return "Search for concepts, methods, findings, or terminology...";

      case "lexical":
        return "Search for exact terminology, identifiers, acronyms, or phrases...";

      case "hybrid":
        return "Search using semantic and lexical evidence together...";
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <p className="eyebrow">
            LOCAL RESEARCH WORKSPACE
          </p>

          <h1>
            Research Assistant
          </h1>
        </div>

        <div className="runtime-status">
          <span className="status-dot" />
          Local
        </div>
      </header>

      <div className="workspace">
        <DocumentSidebar
          documents={documents}
          selectedDocumentId={
            activeDocumentId
          }
          isLoading={isLoading}
          isUploading={isUploading}
          deletingDocumentId={
            deletingDocumentId
          }
          error={documentError}
          onSelectDocument={
            handleSelectDocument
          }
          onUpload={
            handleUpload
          }
          onDelete={
            handleDelete
          }
          onRetry={
            refreshDocuments
          }
          onDismissError={
            clearDocumentError
          }
        />

        <main className="main">
          <section className="hero">
            <p className="eyebrow">
              RESEARCH WORKSPACE
            </p>

            <h2>
              Ask your research
              corpus.
            </h2>

            <p>
              Retrieve evidence,
              generate grounded
              answers, and inspect
              the exact sources
              supplied to the model.
            </p>
          </section>

          <section className="research-panel">
            <div className="panel-header">
              <div className="panel-controls">
                <div
                  className="mode-switcher"
                  role="group"
                  aria-label="Research action"
                >
                  <button
                    className={`mode-button ${
                      mode === "ask"
                        ? "active"
                        : ""
                    }`}
                    type="button"
                    aria-pressed={
                      mode === "ask"
                    }
                    onClick={() =>
                      setMode("ask")
                    }
                  >
                    Ask
                  </button>

                  <button
                    className={`mode-button ${
                      mode === "search"
                        ? "active"
                        : ""
                    }`}
                    type="button"
                    aria-pressed={
                      mode === "search"
                    }
                    onClick={() =>
                      setMode(
                        "search",
                      )
                    }
                  >
                    Search
                  </button>
                </div>

                <div className="retrieval-strategy">
                  <span className="retrieval-strategy-label">
                    Retrieval
                  </span>

                  <div
                    className="mode-switcher"
                    role="group"
                    aria-label="Retrieval strategy"
                  >
                    <button
                      className={`mode-button ${
                        retrievalMode ===
                        "dense"
                          ? "active"
                          : ""
                      }`}
                      type="button"
                      aria-pressed={
                        retrievalMode ===
                        "dense"
                      }
                      onClick={() =>
                        handleRetrievalModeChange(
                          "dense",
                        )
                      }
                    >
                      Dense
                    </button>

                    <button
                      className={`mode-button ${
                        retrievalMode ===
                        "lexical"
                          ? "active"
                          : ""
                      }`}
                      type="button"
                      aria-pressed={
                        retrievalMode ===
                        "lexical"
                      }
                      onClick={() =>
                        handleRetrievalModeChange(
                          "lexical",
                        )
                      }
                    >
                      Lexical
                    </button>

                    <button
                      className={`mode-button ${
                        retrievalMode ===
                        "hybrid"
                          ? "active"
                          : ""
                      }`}
                      type="button"
                      aria-pressed={
                        retrievalMode ===
                        "hybrid"
                      }
                      onClick={() =>
                        handleRetrievalModeChange(
                          "hybrid",
                        )
                      }
                    >
                      Hybrid
                    </button>
                  </div>
                </div>
              </div>

              <div className="corpus-scope">
                <span>
                  Scope
                </span>

                <strong
                  title={
                    scopeDescription
                  }
                >
                  {
                    scopeDescription
                  }
                </strong>
              </div>
            </div>

            <label
              className="query-label"
              htmlFor="query"
            >
              {mode === "ask"
                ? "Research question"
                : getSearchLabel()}
            </label>

            <textarea
              id="query"
              className="query-input"
              placeholder={
                mode === "ask"
                  ? "Ask a question about your indexed research..."
                  : getSearchPlaceholder()
              }
              rows={5}
              value={query}
              aria-describedby="research-hint"
              onChange={(event) =>
                handleQueryChange(
                  event.target.value,
                )
              }
              onKeyDown={
                handleQueryKeyDown
              }
            />

            <div className="query-actions">
              <span id="research-hint">
                {researchHint}
              </span>

              {mode === "ask" ? (
                <button
                  className="submit-button"
                  type="button"
                  disabled={
                    queryDisabled
                  }
                  title={
                    getActionTitle()
                  }
                  onClick={() => {
                    void handleQuery();
                  }}
                >
                  {isQuerying
                    ? "Researching..."
                    : "Ask research"}
                </button>
              ) : (
                <button
                  className="submit-button"
                  type="button"
                  disabled={
                    searchDisabled
                  }
                  title={
                    getActionTitle()
                  }
                  onClick={() => {
                    void handleSearch();
                  }}
                >
                  {isSearching
                    ? "Searching..."
                    : "Search evidence"}
                </button>
              )}
            </div>
          </section>

          {mode === "search" ? (
            <SearchResults
              query={lastQuery}
              results={
                searchResults
              }
              isSearching={
                isSearching
              }
              error={
                searchError
              }
              retrievalMode={
                retrievalMode
              }
              onDismissError={
                clearSearchError
              }
            />
          ) : (
            <CitedAnswer
              response={
                queryResponse
              }
              isQuerying={
                isQuerying
              }
              error={
                queryError
              }
              onDismissError={
                clearQueryError
              }
            />
          )}
        </main>
      </div>
    </div>
  );
}

export default App;