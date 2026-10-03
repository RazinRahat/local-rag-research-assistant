import {
  useCallback,
  useRef,
  useState,
} from "react";

import {
  ResearchAPIError,
  researchApi,
  type CitedRAGResponse,
  type QueryRequest,
} from "../api";

import type {
  RetrievalMode,
} from "../api/models";

export interface RAGQueryClient {
  query(
    request: QueryRequest,
  ): Promise<CitedRAGResponse>;
}

interface UseResearchQueryResult {
  response: CitedRAGResponse | null;
  lastQuestion: string | null;
  isQuerying: boolean;
  error: string | null;

  runQuery: (
    question: string,
    documentId: string | null,
    retrievalMode?: RetrievalMode,
  ) => Promise<boolean>;

  clearQuery: () => void;
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

  return "An unexpected research query error occurred.";
}

export function useResearchQuery(
  client: RAGQueryClient = researchApi,
): UseResearchQueryResult {
  const [
    response,
    setResponse,
  ] = useState<CitedRAGResponse | null>(
    null,
  );

  const [
    lastQuestion,
    setLastQuestion,
  ] = useState<string | null>(
    null,
  );

  const [
    isQuerying,
    setIsQuerying,
  ] = useState(false);

  const [
    error,
    setError,
  ] = useState<string | null>(
    null,
  );

  const requestSequence =
    useRef(0);

  const runQuery =
    useCallback(
      async (
        question: string,
        documentId: string | null,
        retrievalMode:
          RetrievalMode = "dense",
      ): Promise<boolean> => {
        const cleanQuestion =
          question.trim();

        if (!cleanQuestion) {
          setError(
            "Research question cannot be blank.",
          );

          return false;
        }

        const requestId =
          ++requestSequence.current;

        setIsQuerying(true);
        setError(null);
        setResponse(null);

        try {
          const request:
            QueryRequest = {
            question: cleanQuestion,
            top_k: 5,
          };

          if (documentId) {
            request.document_id =
              documentId;
          }

          /*
           * Dense remains implicit so
           * old Phase 10 requests retain
           * their original shape.
           */
          if (
            retrievalMode !== "dense"
          ) {
            request.mode =
              retrievalMode;
          }

          const result =
            await client.query(
              request,
            );

          if (
            requestId !==
            requestSequence.current
          ) {
            return false;
          }

          setResponse(result);

          setLastQuestion(
            cleanQuestion,
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

          setLastQuestion(
            cleanQuestion,
          );

          return false;
        } finally {
          if (
            requestId ===
            requestSequence.current
          ) {
            setIsQuerying(false);
          }
        }
      },
      [client],
    );

  const clearQuery =
    useCallback(() => {
      requestSequence.current += 1;

      setResponse(null);
      setLastQuestion(null);
      setError(null);
      setIsQuerying(false);
    }, []);

  const clearError =
    useCallback(() => {
      setError(null);
    }, []);

  return {
    response,
    lastQuestion,
    isQuerying,
    error,
    runQuery,
    clearQuery,
    clearError,
  };
}