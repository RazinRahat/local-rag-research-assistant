import {
  act,
  renderHook,
} from "@testing-library/react";

import {
  ResearchAPIError,
  type CitedRAGResponse,
} from "../api";

import {
  type RAGQueryClient,
  useResearchQuery,
} from "./useResearchQuery";
import { describe, expect, it } from "vitest";

const DOCUMENT_ID =
  "a".repeat(64);

const RESPONSE:
  CitedRAGResponse = {
  rag: {
    question:
      "How do Transformers avoid recurrence?",
    answer:
      "Transformers use attention rather than recurrent computation [S1].",
    evidence: [],
    retrieved_count: 1,
    used_evidence_count: 1,
    estimated_prompt_tokens: 120,
    actual_prompt_tokens: 124,
    context_truncated: false,
    insufficient_evidence: false,
    generation: null,
  },
  citations: {
    references: [
      {
        source_id: "S1",
        start_index: 56,
        end_index: 60,
      },
    ],
    sources: [],
  },
};

function makeClient(
  query:
    RAGQueryClient["query"] =
      async () => RESPONSE,
): RAGQueryClient {
  return {
    query,
  };
}

describe("useResearchQuery", () => {
  it("runs a document-scoped RAG query", async () => {
    let receivedDocumentId:
      string | undefined;

    const client =
      makeClient(
        async (request) => {
          receivedDocumentId =
            request.document_id ??
            undefined;

          return RESPONSE;
        },
      );

    const { result } =
      renderHook(() =>
        useResearchQuery(
          client,
        ),
      );

    await act(async () => {
      await result.current.runQuery(
        "  How do Transformers avoid recurrence?  ",
        DOCUMENT_ID,
      );
    });

    expect(
      receivedDocumentId,
    ).toBe(
      DOCUMENT_ID,
    );

    expect(
      result.current
        .lastQuestion,
    ).toBe(
      "How do Transformers avoid recurrence?",
    );

    expect(
      result.current.response,
    ).toEqual(
      RESPONSE,
    );
  });

  it("rejects blank questions locally", async () => {
    let callCount = 0;

    const client =
      makeClient(
        async () => {
          callCount += 1;

          return RESPONSE;
        },
      );

    const { result } =
      renderHook(() =>
        useResearchQuery(
          client,
        ),
      );

    await act(async () => {
      await result.current.runQuery(
        "   ",
        null,
      );
    });

    expect(callCount).toBe(0);

    expect(
      result.current.error,
    ).toBe(
      "Research question cannot be blank.",
    );
  });

  it("exposes API errors", async () => {
    const client =
      makeClient(
        async () => {
          throw new ResearchAPIError(
            503,
            "Local language model is unavailable",
          );
        },
      );

    const { result } =
      renderHook(() =>
        useResearchQuery(
          client,
        ),
      );

    await act(async () => {
      await result.current.runQuery(
        "What is attention?",
        null,
      );
    });

    expect(
      result.current.error,
    ).toBe(
      "Local language model is unavailable",
    );

    expect(
      result.current.response,
    ).toBeNull();
  });

  it("clears previous query state", async () => {
    const { result } =
      renderHook(() =>
        useResearchQuery(
          makeClient(),
        ),
      );

    await act(async () => {
      await result.current.runQuery(
        "What is attention?",
        null,
      );
    });

    act(() => {
      result.current.clearQuery();
    });

    expect(
      result.current.response,
    ).toBeNull();

    expect(
      result.current
        .lastQuestion,
    ).toBeNull();
  });

  it("ignores an older RAG response that finishes later", async () => {
    let resolveFirst:
      (
        response:
          CitedRAGResponse,
      ) => void =
      () => undefined;

    let resolveSecond:
      (
        response:
          CitedRAGResponse,
      ) => void =
      () => undefined;

    const firstPromise =
      new Promise<CitedRAGResponse>(
        (resolve) => {
          resolveFirst = resolve;
        },
      );

    const secondPromise =
      new Promise<CitedRAGResponse>(
        (resolve) => {
          resolveSecond = resolve;
        },
      );

    let callCount = 0;

    const client =
      makeClient(
        async () => {
          callCount += 1;

          return callCount === 1
            ? firstPromise
            : secondPromise;
        },
      );

    const { result } =
      renderHook(() =>
        useResearchQuery(
          client,
        ),
      );

    let firstRun:
      Promise<boolean>;

    let secondRun:
      Promise<boolean>;

    act(() => {
      firstRun =
        result.current.runQuery(
          "first question",
          null,
        );

      secondRun =
        result.current.runQuery(
          "second question",
          null,
        );
    });

    const secondResponse = {
      ...RESPONSE,
      rag: {
        ...RESPONSE.rag,
        question:
          "second question",
      },
    };

    await act(async () => {
      resolveSecond(
        secondResponse,
      );

      await secondRun;
    });

    expect(
      result.current
        .lastQuestion,
    ).toBe(
      "second question",
    );

    await act(async () => {
      resolveFirst({
        ...RESPONSE,
        rag: {
          ...RESPONSE.rag,
          question:
            "first question",
        },
      });

      await firstRun;
    });

    expect(
      result.current
        .lastQuestion,
    ).toBe(
      "second question",
    );
  });

  it("forwards lexical retrieval mode", async () => {
    let receivedMode:
      string | undefined;

    const client =
      makeClient(
        async (request) => {
          receivedMode =
            request.mode;

          return RESPONSE;
        },
      );

    const { result } =
      renderHook(() =>
        useResearchQuery(
          client,
        ),
      );

    await act(async () => {
      await result.current.runQuery(
        "Which optimizer was used?",
        null,
        "lexical",
      );
    });

    expect(
      receivedMode,
    ).toBe(
      "lexical",
    );
  });

  it("forwards hybrid retrieval mode", async () => {
    let receivedMode:
      string | undefined;

    const client =
      makeClient(
        async (request) => {
          receivedMode =
            request.mode;

          return RESPONSE;
        },
      );

    const { result } =
      renderHook(() =>
        useResearchQuery(
          client,
        ),
      );

    await act(async () => {
      await result.current.runQuery(
        "Explain the attention mechanism.",
        null,
        "hybrid",
      );
    });

    expect(
      receivedMode,
    ).toBe(
      "hybrid",
    );
  });

  it("forwards hybrid reranked retrieval mode", async () => {
    let receivedMode:
      string | undefined;

    const client =
      makeClient(
        async (request) => {
          receivedMode =
            request.mode;

          return RESPONSE;
        },
      );

    const { result } =
      renderHook(() =>
        useResearchQuery(
          client,
        ),
      );

    await act(async () => {
      await result.current.runQuery(
        "Explain the attention mechanism.",
        null,
        "hybrid_reranked",
      );
    });

    expect(
      receivedMode,
    ).toBe(
      "hybrid_reranked",
    );
  });
});