import {
  render,
  screen,
} from "@testing-library/react";

import userEvent from "@testing-library/user-event";
import {
    describe,
    expect,
  it,
  vi,
} from "vitest";

import type {
  CitationReference,
  CitationSource,
} from "../api";

import {
  AnswerWithCitations,
} from "./AnswerWithCitations";

const DOCUMENT_ID =
  "a".repeat(64);

function makeSource(
  sourceId = "S1",
): CitationSource {
  return {
    source_id: sourceId,
    evidence: {
      source_id: sourceId,
      retrieval_rank: 1,
      score: 0.91,
      chunk: {
        chunk_id:
          "chunk-1",
        document_id:
          DOCUMENT_ID,
        file_name:
          "paper.pdf",
        page_number: 3,
        chunk_index: 0,
        page_chunk_index: 0,
        text:
          "Attention replaces recurrence.",
        token_count: 4,
        char_count: 30,
        token_start: 0,
        token_end: 4,
      },
    },
  };
}

describe("AnswerWithCitations", () => {
  it("renders validated citations as interactive controls", async () => {
    const user =
      userEvent.setup();

    const onSelect =
      vi.fn();

    const answer =
      "Attention replaces recurrence [S1].";

    const startIndex =
      Array.from(
        "Attention replaces recurrence ",
      ).length;

    const references:
      CitationReference[] = [
        {
          source_id: "S1",
          start_index:
            startIndex,
          end_index:
            startIndex + 4,
        },
      ];

    render(
      <AnswerWithCitations
        answer={answer}
        references={
          references
        }
        sources={[
          makeSource(),
        ]}
        activeSourceId={
          null
        }
        onSelectSource={
          onSelect
        }
      />,
    );

    const citation =
      screen.getByRole(
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
      onSelect,
    ).toHaveBeenCalledWith(
      "S1",
    );
  });

  it("handles Python-style Unicode code-point offsets", () => {
    const prefix =
      "🙂 Attention works ";

    const answer =
      `${prefix}[S1].`;

    const startIndex =
      Array.from(prefix).length;

    render(
      <AnswerWithCitations
        answer={answer}
        references={[
          {
            source_id: "S1",
            start_index:
              startIndex,
            end_index:
              startIndex + 4,
          },
        ]}
        sources={[
          makeSource(),
        ]}
        activeSourceId={
          "S1"
        }
        onSelectSource={() =>
          undefined
        }
      />,
    );

    expect(
      screen.getByRole(
        "button",
        {
          name:
            "Inspect source S1",
        },
      ),
    ).toHaveTextContent(
      "[S1]",
    );

    expect(
      screen.getByText(
        /🙂 Attention works/,
      ),
    ).toBeInTheDocument();
  });

  it("does not create controls for unknown sources", () => {
    render(
      <AnswerWithCitations
        answer="Unsupported [S99]."
        references={[
          {
            source_id: "S99",
            start_index: 12,
            end_index: 17,
          },
        ]}
        sources={[
          makeSource("S1"),
        ]}
        activeSourceId={
          null
        }
        onSelectSource={() =>
          undefined
        }
      />,
    );

    expect(
      screen.queryByRole(
        "button",
        {
          name:
            "Inspect source S99",
        },
      ),
    ).not.toBeInTheDocument();
  });
});