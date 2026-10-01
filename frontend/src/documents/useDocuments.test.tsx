import {
  act,
  renderHook,
  waitFor,
} from "@testing-library/react";

import {
  ResearchAPIError,
  type StoredDocument,
} from "../api";

import {
  type DocumentClient,
  useDocuments,
} from "./useDocuments";
import { describe, expect, it } from "vitest";

const DOCUMENT_ID =
  "a".repeat(64);

const DOCUMENT: StoredDocument = {
  document_id: DOCUMENT_ID,
  file_name: "paper.pdf",
  chunk_count: 3,
};

function makeClient(
  overrides:
    Partial<DocumentClient> = {},
): DocumentClient {
  return {
    listDocuments: async () => [
      DOCUMENT,
    ],

    uploadDocument:
      async () => DOCUMENT,

    deleteDocument:
      async () => undefined,

    ...overrides,
  };
}

describe("useDocuments", () => {
  it("loads indexed documents", async () => {
    const client =
      makeClient();

    const { result } =
      renderHook(() =>
        useDocuments(client),
      );

    await waitFor(() => {
      expect(
        result.current.isLoading,
      ).toBe(false);
    });

    expect(
      result.current.documents,
    ).toEqual([DOCUMENT]);
  });

  it("upserts an uploaded document", async () => {
    const updated: StoredDocument = {
      ...DOCUMENT,
      chunk_count: 7,
    };

    const client =
      makeClient({
        uploadDocument:
          async () => updated,
      });

    const { result } =
      renderHook(() =>
        useDocuments(client),
      );

    await waitFor(() => {
      expect(
        result.current.documents,
      ).toHaveLength(1);
    });

    const file = new File(
      ["pdf"],
      "paper.pdf",
      {
        type: "application/pdf",
      },
    );

    await act(async () => {
      await result.current.uploadDocument(
        file,
      );
    });

    expect(
      result.current.documents,
    ).toEqual([updated]);
  });

  it("removes a deleted document", async () => {
    const client =
      makeClient();

    const { result } =
      renderHook(() =>
        useDocuments(client),
      );

    await waitFor(() => {
      expect(
        result.current.documents,
      ).toHaveLength(1);
    });

    let deleted = false;

    await act(async () => {
      deleted =
        await result.current.deleteDocument(
          DOCUMENT_ID,
        );
    });

    expect(deleted).toBe(true);

    expect(
      result.current.documents,
    ).toEqual([]);
  });

  it("exposes API failures", async () => {
    const client =
      makeClient({
        listDocuments: async () => {
          throw new ResearchAPIError(
            500,
            "Unable to load documents",
          );
        },
      });

    const { result } =
      renderHook(() =>
        useDocuments(client),
      );

    await waitFor(() => {
      expect(
        result.current.isLoading,
      ).toBe(false);
    });

    expect(
      result.current.error,
    ).toBe(
      "Unable to load documents",
    );
  });
});