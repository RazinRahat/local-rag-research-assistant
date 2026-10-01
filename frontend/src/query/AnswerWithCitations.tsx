import type {
  ReactNode,
} from "react";

import type {
  CitationReference,
  CitationSource,
} from "../api";

interface AnswerWithCitationsProps {
  answer: string;
  references: CitationReference[];
  sources: CitationSource[];
  activeSourceId: string | null;
  onSelectSource: (
    sourceId: string,
  ) => void;
}

export function AnswerWithCitations({
  answer,
  references,
  sources,
  activeSourceId,
  onSelectSource,
}: AnswerWithCitationsProps) {
  const knownSourceIds =
    new Set(
      sources.map(
        (source) =>
          source.source_id,
      ),
    );

  const answerCodePoints =
    Array.from(answer);

  const sortedReferences =
    [...references].sort(
      (left, right) =>
        left.start_index -
        right.start_index,
    );

  const content: ReactNode[] = [];

  let cursor = 0;

  sortedReferences.forEach(
    (reference, index) => {
      const {
        source_id,
        start_index,
        end_index,
      } = reference;

      const validRange =
        start_index >= cursor &&
        end_index > start_index &&
        end_index <=
          answerCodePoints.length;

      const knownSource =
        knownSourceIds.has(
          source_id,
        );

      if (
        !validRange ||
        !knownSource
      ) {
        return;
      }

      if (
        start_index > cursor
      ) {
        content.push(
          answerCodePoints
            .slice(
              cursor,
              start_index,
            )
            .join(""),
        );
      }

      const citationText =
        answerCodePoints
          .slice(
            start_index,
            end_index,
          )
          .join("");

      content.push(
        <button
          key={`${source_id}-${start_index}-${index}`}
          className={`inline-citation ${
            activeSourceId ===
            source_id
              ? "active"
              : ""
          }`}
          type="button"
          aria-label={`Inspect source ${source_id}`}
          aria-pressed={
            activeSourceId ===
            source_id
          }
          onClick={() =>
            onSelectSource(
              source_id,
            )
          }
        >
          {citationText}
        </button>,
      );

      cursor = end_index;
    },
  );

  if (
    cursor <
    answerCodePoints.length
  ) {
    content.push(
      answerCodePoints
        .slice(cursor)
        .join(""),
    );
  }

  return (
    <p className="answer-text">
      {content}
    </p>
  );
}