import {
  useCallback,
  useRef,
  useState,
} from "react";

import {
  ResearchAPIError,
  researchApi,
  type RetrievalResult,
  type SearchRequest,
  type SearchResponse,
} from "../api";

import type {
  RetrievalMode,
} from "../api/models";

export interface SearchClient {
  search(
    request: SearchRequest,
  ): Promise<SearchResponse>;
}

interface UseSearchResult {
  results: RetrievalResult[];
  lastQuery: string | null;
  isSearching: boolean;
  error: string | null;

  runSearch: (
    query: string,
    documentId: string | null,
    retrievalMode?: RetrievalMode,
  ) => Promise<boolean>;

  clearSearch: () => void;
  clearError: () => void;
}

function getErrorMessage(
  error: unknown,
): string {
  if (error instanceof ResearchAPIError) {
    return error.message;
  }

  if (error instanceof Error) {
    return error.message;
  }

  return "An unexpected search error occurred.";
}

export function useSearch(
  client: SearchClient = researchApi,
): UseSearchResult {
  const [
    results,
    setResults,
  ] = useState<RetrievalResult[]>(
    [],
  );

  const [
    lastQuery,
    setLastQuery,
  ] = useState<string | null>(
    null,
  );

  const [
    isSearching,
    setIsSearching,
  ] = useState(false);

  const [
    error,
    setError,
  ] = useState<string | null>(
    null,
  );

  const requestSequence =
    useRef(0);

  const runSearch =
    useCallback(
      async (
        query: string,
        documentId: string | null,
        retrievalMode:
          RetrievalMode = "dense",
      ): Promise<boolean> => {
        const cleanQuery =
          query.trim();

        if (!cleanQuery) {
          setError(
            "Search query cannot be blank.",
          );

          return false;
        }

        const requestId =
          ++requestSequence.current;

        setIsSearching(true);
        setError(null);
        setResults([]);

        try {
          const request:
            SearchRequest = {
            query: cleanQuery,
            top_k: 5,
          };

          if (documentId) {
            request.document_id =
              documentId;
          }

          /*
           * Dense is the backend default.
           *
           * We intentionally omit `mode`
           * for dense requests so the
           * original Phase 10 request
           * shape remains unchanged.
           */
          if (
            retrievalMode !== "dense"
          ) {
            request.mode =
              retrievalMode;
          }

          const response =
            await client.search(
              request,
            );

          if (
            requestId !==
            requestSequence.current
          ) {
            return false;
          }

          setResults(
            response.results,
          );

          setLastQuery(
            response.query,
          );

          return true;
        } catch (caughtError) {
          if (
            requestId !==
            requestSequence.current
          ) {
            return false;
          }

          setError(
            getErrorMessage(
              caughtError,
            ),
          );

          setResults([]);

          setLastQuery(
            cleanQuery,
          );

          return false;
        } finally {
          if (
            requestId ===
            requestSequence.current
          ) {
            setIsSearching(false);
          }
        }
      },
      [client],
    );

  const clearSearch =
    useCallback(() => {
      /*
       * Incrementing this invalidates
       * any request that is still
       * running.
       */
      requestSequence.current += 1;

      setResults([]);
      setLastQuery(null);
      setError(null);
      setIsSearching(false);
    }, []);

  const clearError =
    useCallback(() => {
      setError(null);
    }, []);

  return {
    results,
    lastQuery,
    isSearching,
    error,
    runSearch,
    clearSearch,
    clearError,
  };
}