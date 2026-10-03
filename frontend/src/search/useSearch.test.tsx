import {
  act,
  renderHook,
} from "@testing-library/react";

import {
  ResearchAPIError,
  type SearchResponse,
} from "../api";

import {
  type SearchClient,
  useSearch,
} from "./useSearch";
import { describe, expect, it } from "vitest";

const DOCUMENT_ID =
  "a".repeat(64);

const RESPONSE: SearchResponse = {
  query: "attention",
  results: [
    {
      rank: 1,
      score: 0.91,
      chunk: {
        chunk_id:
          "chunk-1",
        document_id:
          DOCUMENT_ID,
        file_name:
          "paper.pdf",
        page_number: 2,
        chunk_index: 0,
        page_chunk_index: 0,
        text:
          "Transformers use attention.",
        token_count: 5,
        char_count: 27,
        token_start: 0,
        token_end: 5,
      },
    },
  ],
};

function makeClient(
  search:
    SearchClient["search"] =
      async () => RESPONSE,
): SearchClient {
  return {
    search,
  };
}

describe("useSearch", () => {
  it("runs semantic search", async () => {
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
        useSearch(client),
      );

    await act(async () => {
      await result.current.runSearch(
        "  attention  ",
        DOCUMENT_ID,
      );
    });

    expect(
      receivedDocumentId,
    ).toBe(
      DOCUMENT_ID,
    );

    expect(
      result.current.lastQuery,
    ).toBe(
      "attention",
    );

    expect(
      result.current.results,
    ).toEqual(
      RESPONSE.results,
    );
  });

  it("rejects blank searches locally", async () => {
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
        useSearch(client),
      );

    await act(async () => {
      await result.current.runSearch(
        "   ",
        null,
      );
    });

    expect(callCount).toBe(0);

    expect(
      result.current.error,
    ).toBe(
      "Search query cannot be blank.",
    );
  });

  it("exposes API errors", async () => {
    const client =
      makeClient(
        async () => {
          throw new ResearchAPIError(
            500,
            "Search unavailable",
          );
        },
      );

    const { result } =
      renderHook(() =>
        useSearch(client),
      );

    await act(async () => {
      await result.current.runSearch(
        "attention",
        null,
      );
    });

    expect(
      result.current.error,
    ).toBe(
      "Search unavailable",
    );

    expect(
      result.current.results,
    ).toEqual([]);
  });

  it("clears previous search state", async () => {
    const { result } =
      renderHook(() =>
        useSearch(
          makeClient(),
        ),
      );

    await act(async () => {
      await result.current.runSearch(
        "attention",
        null,
      );
    });

    act(() => {
      result.current.clearSearch();
    });

    expect(
      result.current.lastQuery,
    ).toBeNull();

    expect(
      result.current.results,
    ).toEqual([]);
  });

  it("ignores an older search response that finishes later", async () => {
    let resolveFirst:
      (
        response: SearchResponse,
      ) => void =
      () => undefined;

    let resolveSecond:
      (
        response: SearchResponse,
      ) => void =
      () => undefined;

    const firstPromise =
      new Promise<SearchResponse>(
        (resolve) => {
          resolveFirst = resolve;
        },
      );

    const secondPromise =
      new Promise<SearchResponse>(
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
        useSearch(client),
      );

    let firstRun:
      Promise<boolean>;

    let secondRun:
      Promise<boolean>;

    act(() => {
      firstRun =
        result.current.runSearch(
          "first query",
          null,
        );

      secondRun =
        result.current.runSearch(
          "second query",
          null,
        );
    });

    const secondResponse:
      SearchResponse = {
      ...RESPONSE,
      query: "second query",
    };

    await act(async () => {
      resolveSecond(
        secondResponse,
      );

      await secondRun;
    });

    expect(
      result.current.lastQuery,
    ).toBe(
      "second query",
    );

    await act(async () => {
      resolveFirst({
        ...RESPONSE,
        query: "first query",
      });

      await firstRun;
    });

    expect(
      result.current.lastQuery,
    ).toBe(
      "second query",
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
        useSearch(client),
      );

    await act(async () => {
      await result.current.runSearch(
        "attention",
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

  it("keeps dense retrieval implicit for backward compatibility", async () => {
    let receivedRequest:
      Parameters<
        SearchClient["search"]
      >[0] | null = null;

    const client =
      makeClient(
        async (request) => {
          receivedRequest =
            request;

          return RESPONSE;
        },
      );

    const { result } =
      renderHook(() =>
        useSearch(client),
      );

    await act(async () => {
      await result.current.runSearch(
        "attention",
        null,
        "dense",
      );
    });

    expect(
      receivedRequest,
    ).not.toBeNull();

    expect(
      receivedRequest,
    ).not.toHaveProperty(
      "mode",
    );
  });
});