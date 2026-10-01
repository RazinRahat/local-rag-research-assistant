import {
  render,
  screen,
} from "@testing-library/react";

import type {
  RetrievalResult,
} from "../api";

import {
  SearchResults,
} from "./SearchResults";
import { describe, expect, it } from "vitest";

const RESULT: RetrievalResult = {
  rank: 1,
  score: 0.9134,
  chunk: {
    chunk_id: "chunk-1",
    document_id:
      "a".repeat(64),
    file_name: "attention.pdf",
    page_number: 3,
    chunk_index: 2,
    page_chunk_index: 0,
    text:
      "Transformers use self-attention to model relationships between tokens.",
    token_count: 11,
    char_count: 68,
    token_start: 0,
    token_end: 11,
  },
};

describe("SearchResults", () => {
  it("renders ranked evidence", () => {
    render(
      <SearchResults
        query="attention"
        results={[RESULT]}
        isSearching={false}
        error={null}
        onDismissError={() =>
          undefined
        }
      />,
    );

    expect(
      screen.getByText(
        "attention.pdf",
      ),
    ).toBeInTheDocument();

    expect(
      screen.getByText(
        "Page 3",
      ),
    ).toBeInTheDocument();

    expect(
      screen.getByText(
        "0.913",
      ),
    ).toBeInTheDocument();

    expect(
      screen.getByText(
        RESULT.chunk.text,
      ),
    ).toBeInTheDocument();
  });

  it("renders empty search state", () => {
    render(
      <SearchResults
        query="missing concept"
        results={[]}
        isSearching={false}
        error={null}
        onDismissError={() =>
          undefined
        }
      />,
    );

    expect(
      screen.getByRole(
        "heading",
        {
          name:
            "No matching evidence",
        },
      ),
    ).toBeInTheDocument();
  });

  it("renders loading state", () => {
    render(
      <SearchResults
        query={null}
        results={[]}
        isSearching
        error={null}
        onDismissError={() =>
          undefined
        }
      />,
    );

    expect(
      screen.getByText(
        "Searching the index",
      ),
    ).toBeInTheDocument();
  });

  it("renders search failures", () => {
    render(
      <SearchResults
        query="attention"
        results={[]}
        isSearching={false}
        error="Search unavailable"
        onDismissError={() =>
          undefined
        }
      />,
    );

    expect(
      screen.getByRole(
        "alert",
      ),
    ).toHaveTextContent(
      "Search unavailable",
    );

    expect(
      screen.getByRole(
        "button",
        {
          name: "Dismiss",
        },
      ),
    ).toBeInTheDocument();
  });
});