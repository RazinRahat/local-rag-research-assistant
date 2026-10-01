import {
  afterEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";

import {
  ResearchAPIClient,
  ResearchAPIError,
} from "./client";

const DOCUMENT_ID = "a".repeat(64);

function jsonResponse(
  body: unknown,
  status = 200,
): Response {
  return new Response(
    JSON.stringify(body),
    {
      status,
      headers: {
        "Content-Type": "application/json",
      },
    },
  );
}

describe("ResearchAPIClient", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("lists indexed documents", async () => {
    const fetchMock = vi.fn<typeof fetch>();

    fetchMock.mockResolvedValue(
      jsonResponse([
        {
          document_id: DOCUMENT_ID,
          file_name: "paper.pdf",
          chunk_count: 4,
        },
      ]),
    );

    vi.stubGlobal(
      "fetch",
      fetchMock,
    );

    const client =
      new ResearchAPIClient("/api");

    const documents =
      await client.listDocuments();

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/documents",
      undefined,
    );

    expect(documents).toEqual([
      {
        document_id: DOCUMENT_ID,
        file_name: "paper.pdf",
        chunk_count: 4,
      },
    ]);
  });

  it("uploads a document using form data", async () => {
    const fetchMock = vi.fn<typeof fetch>();

    fetchMock.mockResolvedValue(
      jsonResponse(
        {
          document_id: DOCUMENT_ID,
          file_name: "paper.pdf",
          chunk_count: 3,
        },
        201,
      ),
    );

    vi.stubGlobal(
      "fetch",
      fetchMock,
    );

    const client =
      new ResearchAPIClient("/api");

    const file = new File(
      ["pdf-content"],
      "paper.pdf",
      {
        type: "application/pdf",
      },
    );

    const result =
      await client.uploadDocument(file);

    expect(result.document_id).toBe(
      DOCUMENT_ID,
    );

    expect(fetchMock).toHaveBeenCalledTimes(1);

    const [
      url,
      init,
    ] = fetchMock.mock.calls[0];

    expect(url).toBe(
      "/api/documents",
    );

    expect(init?.method).toBe(
      "POST",
    );

    expect(init?.body).toBeInstanceOf(
      FormData,
    );

    const body =
      init?.body as FormData;

    expect(
      body.get("file"),
    ).toBeInstanceOf(File);

    expect(init?.headers).toBeUndefined();
  });

  it("deletes an indexed document", async () => {
    const fetchMock = vi.fn<typeof fetch>();

    fetchMock.mockResolvedValue(
      new Response(null, {
        status: 204,
      }),
    );

    vi.stubGlobal(
      "fetch",
      fetchMock,
    );

    const client =
      new ResearchAPIClient("/api");

    await client.deleteDocument(
      DOCUMENT_ID,
    );

    expect(fetchMock).toHaveBeenCalledWith(
      `/api/documents/${DOCUMENT_ID}`,
      {
        method: "DELETE",
      },
    );
  });

  it("sends semantic search requests as JSON", async () => {
    const fetchMock = vi.fn<typeof fetch>();

    fetchMock.mockResolvedValue(
      jsonResponse({
        query: "attention",
        results: [],
      }),
    );

    vi.stubGlobal(
      "fetch",
      fetchMock,
    );

    const client =
      new ResearchAPIClient("/api");

    const response =
      await client.search({
        query: "attention",
        top_k: 3,
        document_id: DOCUMENT_ID,
      });

    expect(response).toEqual({
      query: "attention",
      results: [],
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/search",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: "attention",
          top_k: 3,
          document_id: DOCUMENT_ID,
        }),
      },
    );
  });

  it("returns cited RAG responses", async () => {
    const fetchMock = vi.fn<typeof fetch>();

    fetchMock.mockResolvedValue(
      jsonResponse({
        rag: {
          question: "What is attention?",
          answer:
            "The indexed documents do not contain enough evidence.",
          evidence: [],
          retrieved_count: 0,
          used_evidence_count: 0,
          estimated_prompt_tokens: 0,
          actual_prompt_tokens: 0,
          context_truncated: false,
          insufficient_evidence: true,
          generation: null,
        },
        citations: {
          references: [],
          sources: [],
        },
      }),
    );

    vi.stubGlobal(
      "fetch",
      fetchMock,
    );

    const client =
      new ResearchAPIClient("/api");

    const response =
      await client.query({
        question: "What is attention?",
        top_k: 5,
      });

    expect(
      response.rag.question,
    ).toBe(
      "What is attention?",
    );

    expect(
      response.rag.insufficient_evidence,
    ).toBe(true);

    expect(
      response.citations.references,
    ).toEqual([]);
  });

  it("translates API errors into ResearchAPIError", async () => {
    const fetchMock = vi.fn<typeof fetch>();

    fetchMock.mockResolvedValue(
      jsonResponse(
        {
          detail: "Document not found",
        },
        404,
      ),
    );

    vi.stubGlobal(
      "fetch",
      fetchMock,
    );

    const client =
      new ResearchAPIClient("/api");

    try {
      await client.deleteDocument(
        DOCUMENT_ID,
      );

      throw new Error(
        "Expected request to fail",
      );
    } catch (error) {
      expect(error).toBeInstanceOf(
        ResearchAPIError,
      );

      expect(error).toMatchObject({
        status: 404,
        message: "Document not found",
      });
    }
  });

  it("extracts validation messages from FastAPI errors", async () => {
    const fetchMock = vi.fn<typeof fetch>();

    fetchMock.mockResolvedValue(
      jsonResponse(
        {
          detail: [
            {
              type: "greater_than_equal",
              loc: [
                "body",
                "top_k",
              ],
              msg:
                "Input should be greater than or equal to 1",
            },
          ],
        },
        422,
      ),
    );

    vi.stubGlobal(
      "fetch",
      fetchMock,
    );

    const client =
      new ResearchAPIClient("/api");

    await expect(
      client.search({
        query: "attention",
        top_k: 0,
      }),
    ).rejects.toMatchObject({
      status: 422,
      message:
        "Input should be greater than or equal to 1",
    });
  });
});