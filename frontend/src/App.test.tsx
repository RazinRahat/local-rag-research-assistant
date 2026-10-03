import {
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";

import userEvent from "@testing-library/user-event";
import {
  describe,
  expect,
  it,
  vi,
} from "vitest";

import type {
  CitedRAGResponse,
  SearchResponse,
  StoredDocument,
} from "./api";

import App from "./App";

import type {
  DocumentClient,
} from "./documents";

import type {
  RAGQueryClient,
} from "./query";

import type {
  SearchClient,
} from "./search";

const DOCUMENT_ID =
  "a".repeat(64);

function makeDocumentClient(): DocumentClient {
  const document: StoredDocument = {
    document_id: DOCUMENT_ID,
    file_name: "attention.pdf",
    chunk_count: 8,
  };

  return {
    listDocuments:
      async () => [document],

    uploadDocument:
      async (file) => ({
        document_id:
          "b".repeat(64),
        file_name: file.name,
        chunk_count: 4,
      }),

    deleteDocument:
      async () => undefined,
  };
}

function makeSearchResponse(): SearchResponse {
  return {
    query:
      "attention mechanisms",
    results: [
      {
        rank: 1,
        score: 0.913,
        chunk: {
          chunk_id:
            "chunk-1",
          document_id:
            DOCUMENT_ID,
          file_name:
            "attention.pdf",
          page_number: 2,
          chunk_index: 0,
          page_chunk_index: 0,
          text:
            "Transformers use attention mechanisms to model relationships between tokens.",
          token_count: 12,
          char_count: 75,
          token_start: 0,
          token_end: 12,
        },
      },
    ],
  };
}

function makeQueryResponse():
  CitedRAGResponse {
  const evidence = {
    source_id: "S1",
    retrieval_rank: 1,
    score: 0.913,
    chunk: {
      chunk_id:
        "chunk-1",
      document_id:
        DOCUMENT_ID,
      file_name:
        "attention.pdf",
      page_number: 2,
      chunk_index: 0,
      page_chunk_index: 0,
      text:
        "Transformers use attention mechanisms rather than recurrent computation.",
      token_count: 10,
      char_count: 72,
      token_start: 0,
      token_end: 10,
    },
  };

  const answer =
    "Transformers use attention instead of recurrent computation [S1].";

  const citationStart =
    Array.from(
      "Transformers use attention instead of recurrent computation ",
    ).length;

  return {
    rag: {
      question:
        "How do Transformers avoid recurrence?",
      answer,
      evidence: [
        evidence,
      ],
      retrieved_count: 1,
      used_evidence_count: 1,
      estimated_prompt_tokens: 120,
      actual_prompt_tokens: 123,
      context_truncated: false,
      insufficient_evidence: false,
      generation: null,
    },
    citations: {
      references: [
        {
          source_id: "S1",
          start_index:
            citationStart,
          end_index:
            citationStart + 4,
        },
      ],
      sources: [
        {
          source_id: "S1",
          evidence,
        },
      ],
    },
  };
}

describe("App", () => {
  it("renders the research workspace", async () => {
    render(
      <App
        documentClient={
          makeDocumentClient()
        }
      />,
    );

    expect(
      screen.getByRole(
        "heading",
        {
          name:
            "Research Assistant",
        },
      ),
    ).toBeInTheDocument();

    expect(
      screen.getByRole(
        "heading",
        {
          name:
            "Ask your research corpus.",
        },
      ),
    ).toBeInTheDocument();

    expect(
      screen.getByRole(
        "button",
        {
          name:
            "Add document",
        },
      ),
    ).toBeInTheDocument();

    expect(
      await screen.findByText(
        "attention.pdf",
      ),
    ).toBeInTheDocument();
  });

  it("uploads a document and displays it", async () => {
    const user =
      userEvent.setup();

    render(
      <App
        documentClient={
          makeDocumentClient()
        }
      />,
    );

    const input =
      screen.getByLabelText(
        "Upload PDF",
      );

    const file =
      new File(
        ["pdf-data"],
        "new-paper.pdf",
        {
          type:
            "application/pdf",
        },
      );

    await user.upload(
      input,
      file,
    );

    const sidebar =
      screen.getByRole(
        "complementary",
      );

    await waitFor(() => {
      expect(
        within(
          sidebar,
        ).getByText(
          "new-paper.pdf",
        ),
      ).toBeInTheDocument();
    });

    expect(
      within(
        sidebar,
      ).getByText(
        "4 chunks",
      ),
    ).toBeInTheDocument();

    expect(
      screen.getByTitle(
        "new-paper.pdf",
      ),
    ).toHaveTextContent(
      "new-paper.pdf",
    );
  });

  it("performs semantic search within the selected document", async () => {
    const user =
      userEvent.setup();

    const searchMock =
      vi.fn<
        SearchClient["search"]
      >();

    searchMock.mockResolvedValue(
      makeSearchResponse(),
    );

    render(
      <App
        documentClient={
          makeDocumentClient()
        }
        searchClient={{
          search: searchMock,
        }}
      />,
    );

    await user.click(
      await screen.findByRole(
        "button",
        {
          name:
            "Select attention.pdf",
        },
      ),
    );

    await user.click(
      screen.getByRole(
        "button",
        {
          name: "Search",
        },
      ),
    );

    const input =
      screen.getByRole(
        "textbox",
        {
          name:
            "Semantic search",
        },
      );

    await user.type(
      input,
      "attention mechanisms",
    );

    await user.click(
      screen.getByRole(
        "button",
        {
          name:
            "Search evidence",
        },
      ),
    );

    await waitFor(() => {
      expect(
        searchMock,
      ).toHaveBeenCalledWith({
        query:
          "attention mechanisms",
        top_k: 5,
        document_id:
          DOCUMENT_ID,
      });
    });

    expect(
      await screen.findByText(
        "Transformers use attention mechanisms to model relationships between tokens.",
      ),
    ).toBeInTheDocument();

    expect(
      screen.getByText(
        "Page 2",
      ),
    ).toBeInTheDocument();
  });

  it("performs a cited RAG query within the selected document", async () => {
    const user =
      userEvent.setup();

    const queryMock =
      vi.fn<
        RAGQueryClient["query"]
      >();

    queryMock.mockResolvedValue(
      makeQueryResponse(),
    );

    render(
      <App
        documentClient={
          makeDocumentClient()
        }
        queryClient={{
          query: queryMock,
        }}
      />,
    );

    await user.click(
      await screen.findByRole(
        "button",
        {
          name:
            "Select attention.pdf",
        },
      ),
    );

    const input =
      screen.getByRole(
        "textbox",
        {
          name:
            "Research question",
        },
      );

    await user.type(
      input,
      "How do Transformers avoid recurrence?",
    );

    await user.click(
      screen.getByRole(
        "button",
        {
          name:
            "Ask research",
        },
      ),
    );

    await waitFor(() => {
      expect(
        queryMock,
      ).toHaveBeenCalledWith({
        question:
          "How do Transformers avoid recurrence?",
        top_k: 5,
        document_id:
          DOCUMENT_ID,
      });
    });

    const answer =
      await screen.findByRole(
        "region",
        {
          name:
            "Cited research answer",
        },
      );

    expect(
      answer,
    ).toHaveTextContent(
      "Transformers use attention instead of recurrent computation [S1].",
    );

    expect(
      answer,
    ).toHaveTextContent(
      "Page 2",
    );

    const citation =
      within(answer).getByRole(
        "button",
        {
          name:
            "Inspect source S1",
        },
      );

    expect(
      citation,
    ).toHaveTextContent(
      "[S1]",
    );

    await user.click(
      citation,
    );

    expect(
      within(answer).getByRole(
        "region",
        {
          name:
            "Evidence for S1",
        },
      ),
    ).toHaveTextContent(
      "Transformers use attention mechanisms rather than recurrent computation.",
    );
  });

  it("submits a research question with the keyboard shortcut", async () => {
    const user =
      userEvent.setup();

    const queryMock =
      vi.fn<
        RAGQueryClient["query"]
      >();

    queryMock.mockResolvedValue(
      makeQueryResponse(),
    );

    render(
      <App
        documentClient={
          makeDocumentClient()
        }
        queryClient={{
          query: queryMock,
        }}
      />,
    );

    const input =
      screen.getByRole(
        "textbox",
        {
          name:
            "Research question",
        },
      );

    await user.type(
      input,
      "How do Transformers avoid recurrence?",
    );

    await user.keyboard(
      "{Meta>}{Enter}{/Meta}",
    );

    await waitFor(() => {
      expect(
        queryMock,
      ).toHaveBeenCalledWith({
        question:
          "How do Transformers avoid recurrence?",
        top_k: 5,
      });
    });
  });

  it("disables research actions when the library is empty", async () => {
    const emptyClient:
      DocumentClient = {
      listDocuments:
        async () => [],

      uploadDocument:
        async () => {
          throw new Error(
            "Not used",
          );
        },

      deleteDocument:
        async () => undefined,
    };

    render(
      <App
        documentClient={
          emptyClient
        }
      />,
    );

    await waitFor(() => {
      expect(
        screen.getByText(
          "No documents indexed yet.",
        ),
      ).toBeInTheDocument();
    });

    const input =
      screen.getByRole(
        "textbox",
        {
          name:
            "Research question",
        },
      );

    await userEvent.type(
      input,
      "What is attention?",
    );

    expect(
      screen.getByRole(
        "button",
        {
          name:
            "Ask research",
        },
      ),
    ).toBeDisabled();

    expect(
      screen.getByText(
        /Index at least one PDF before/,
      ),
    ).toBeInTheDocument();
  });

  it("returns to the full corpus when the selected document is deleted", async () => {
    const user =
      userEvent.setup();

    const client =
      makeDocumentClient();

    const deleteMock =
      vi.fn(
        async () => undefined,
      );

    client.deleteDocument =
      deleteMock;

    const confirm =
      vi.spyOn(
        window,
        "confirm",
      );

    confirm.mockReturnValue(
      true,
    );

    render(
      <App
        documentClient={
          client
        }
      />,
    );

    await user.click(
      await screen.findByRole(
        "button",
        {
          name:
            "Select attention.pdf",
        },
      ),
    );

    expect(
      screen.getByTitle(
        "attention.pdf",
      ),
    ).toHaveTextContent(
      "attention.pdf",
    );

    await user.click(
      screen.getByRole(
        "button",
        {
          name:
            "Delete attention.pdf",
        },
      ),
    );

    await waitFor(() => {
      expect(
        deleteMock,
      ).toHaveBeenCalledWith(
        DOCUMENT_ID,
      );
    });

    await waitFor(() => {
      expect(
        screen.getByTitle(
          "All indexed documents",
        ),
      ).toHaveTextContent(
        "All indexed documents",
      );
    });

    expect(
      screen.queryByRole(
        "button",
        {
          name:
            "Select attention.pdf",
        },
      ),
    ).not.toBeInTheDocument();

    confirm.mockRestore();
  });

  it("performs hybrid search when hybrid retrieval is selected", async () => {
    const user =
      userEvent.setup();

    const searchMock =
      vi.fn<
        SearchClient["search"]
      >();

    searchMock.mockResolvedValue(
      makeSearchResponse(),
    );

    render(
      <App
        documentClient={
          makeDocumentClient()
        }
        searchClient={{
          search: searchMock,
        }}
      />,
    );

    await user.click(
      screen.getByRole(
        "button",
        {
          name: "Search",
        },
      ),
    );

    await user.click(
      screen.getByRole(
        "button",
        {
          name: "Hybrid",
        },
      ),
    );

    const input =
      screen.getByRole(
        "textbox",
        {
          name:
            "Hybrid search",
        },
      );

    await user.type(
      input,
      "attention mechanisms",
    );

    await user.click(
      screen.getByRole(
        "button",
        {
          name:
            "Search evidence",
        },
      ),
    );

    await waitFor(() => {
      expect(
        searchMock,
      ).toHaveBeenCalledWith({
        query:
          "attention mechanisms",
        top_k: 5,
        mode: "hybrid",
      });
    });
  });
});