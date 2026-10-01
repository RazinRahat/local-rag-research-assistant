import {
  render,
  screen,
} from "@testing-library/react";

import type {
  CitationSource,
} from "../api";

import {
  EvidenceInspector,
} from "./EvidenceInspector";
import { describe, expect, it } from "vitest";

const SOURCE:
  CitationSource = {
  source_id: "S1",
  evidence: {
    source_id: "S1",
    retrieval_rank: 2,
    score: 0.8123,
    chunk: {
      chunk_id:
        "chunk-abc",
      document_id:
        "a".repeat(64),
      file_name:
        "transformer.pdf",
      page_number: 7,
      chunk_index: 4,
      page_chunk_index: 1,
      text:
        "The model relies entirely on attention mechanisms.",
      token_count: 8,
      char_count: 49,
      token_start: 320,
      token_end: 328,
    },
  },
};

describe("EvidenceInspector", () => {
  it("renders trusted evidence and provenance", () => {
    render(
      <EvidenceInspector
        source={SOURCE}
      />,
    );

    const inspector =
      screen.getByRole(
        "region",
        {
          name:
            "Evidence for S1",
        },
      );

    expect(
      inspector,
    ).toHaveTextContent(
      "transformer.pdf",
    );

    expect(
      inspector,
    ).toHaveTextContent(
      "The model relies entirely on attention mechanisms.",
    );

    expect(
      inspector,
    ).toHaveTextContent(
      "0.812",
    );

    expect(
      inspector,
    ).toHaveTextContent(
      "Retrieval rank",
    );
  });

  it("exposes technical provenance", () => {
    render(
      <EvidenceInspector
        source={SOURCE}
      />,
    );

    expect(
      screen.getByText(
        "Technical provenance",
      ),
    ).toBeInTheDocument();

    expect(
      screen.getByText(
        "chunk-abc",
      ),
    ).toBeInTheDocument();

    expect(
      screen.getByText(
        "320–328",
      ),
    ).toBeInTheDocument();
  });
});