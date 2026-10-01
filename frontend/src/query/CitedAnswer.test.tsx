import {
  render,
  screen,
} from "@testing-library/react";

import type {
  CitedRAGResponse,
} from "../api";

import {
  CitedAnswer,
} from "./CitedAnswer";
import { describe, expect, it } from "vitest";
import { userEvent } from "@testing-library/user-event/dist/cjs/setup/index.js";

const DOCUMENT_ID =
  "a".repeat(64);

function makeResponse():
  CitedRAGResponse {
  const evidence = {
    source_id: "S1",
    retrieval_rank: 1,
    score: 0.91,
    chunk: {
      chunk_id:
        "chunk-1",
      document_id:
        DOCUMENT_ID,
      file_name:
        "transformer.pdf",
      page_number: 3,
      chunk_index: 0,
      page_chunk_index: 0,
      text:
        "The Transformer uses attention instead of recurrence.",
      token_count: 9,
      char_count: 54,
      token_start: 0,
      token_end: 9,
    },
  };

  return {
    rag: {
      question:
        "How does the Transformer avoid recurrence?",
      answer:
        "The Transformer relies on attention instead of recurrent computation [S1].",
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
          start_index: 66,
          end_index: 70,
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

describe("CitedAnswer", () => {
  it("renders a grounded cited answer", () => {
    const response =
      makeResponse();

    render(
      <CitedAnswer
        response={response}
        isQuerying={false}
        error={null}
        onDismissError={() =>
          undefined
        }
      />,
    );

    const answer =
      screen.getByRole(
        "region",
        {
          name:
            "Cited research answer",
        },
      );

    expect(
      answer,
    ).toHaveTextContent(
      response.rag.answer,
    );

    expect(
      answer,
    ).toHaveTextContent(
      "S1",
    );

    expect(
      answer,
    ).toHaveTextContent(
      "transformer.pdf",
    );

    expect(
      answer,
    ).toHaveTextContent(
      "Page 3",
    );
  });

  it("renders insufficient-evidence responses", () => {
    const response =
      makeResponse();

    response.rag = {
      ...response.rag,
      answer:
        "I couldn't find sufficient evidence in the indexed documents to answer that question.",
      evidence: [],
      retrieved_count: 0,
      used_evidence_count: 0,
      insufficient_evidence: true,
      generation: null,
    };

    response.citations = {
      references: [],
      sources: [],
    };

    render(
      <CitedAnswer
        response={response}
        isQuerying={false}
        error={null}
        onDismissError={() =>
          undefined
        }
      />,
    );

    expect(
      screen.getByRole(
        "region",
        {
          name:
            "Insufficient evidence response",
        },
      ),
    ).toHaveTextContent(
      "I couldn't find sufficient evidence",
    );
  });

  it("renders query progress", () => {
    render(
      <CitedAnswer
        response={null}
        isQuerying
        error={null}
        onDismissError={() =>
          undefined
        }
      />,
    );

    expect(
      screen.getByText(
        "Researching your question",
      ),
    ).toBeInTheDocument();
  });

  it("renders query failures", () => {
    render(
      <CitedAnswer
        response={null}
        isQuerying={false}
        error="Local language model is unavailable"
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
      "Local language model is unavailable",
    );
  });

  it("renders generation diagnostics", async () => {
    const user =
      userEvent.setup();

    const response =
      makeResponse();

    response.rag.generation = {
      content:
        response.rag.answer,
      model_name:
        "qwen3.5:9b",
      thinking: null,
      done_reason: "stop",
      prompt_tokens: 123,
      completion_tokens: 37,
      total_duration_ns:
        2_500_000_000,
      load_duration_ns:
        100_000_000,
      prompt_eval_duration_ns:
        500_000_000,
      generation_duration_ns:
        1_900_000_000,
    };

    render(
      <CitedAnswer
        response={response}
        isQuerying={false}
        error={null}
        onDismissError={() =>
          undefined
        }
      />,
    );

    await user.click(
      screen.getByText(
        "Generation diagnostics",
      ),
    );

    expect(
      screen.getByText(
        "qwen3.5:9b",
      ),
    ).toBeInTheDocument();

    expect(
      screen.getByText(
        "37",
      ),
    ).toBeInTheDocument();

    expect(
      screen.getByText(
        "2.50 s",
      ),
    ).toBeInTheDocument();

    expect(
      screen.getByText(
        "stop",
      ),
    ).toBeInTheDocument();
  });
});