import type {
  ComponentProps,
} from "react";

import {
  render,
  screen,
} from "@testing-library/react";

import userEvent from "@testing-library/user-event";
import {
  afterEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";

import type {
  StoredDocument,
} from "../api";

import {
  DocumentSidebar,
} from "./DocumentSidebar";

const DOCUMENT_ID =
  "a".repeat(64);

const DOCUMENT:
  StoredDocument = {
  document_id: DOCUMENT_ID,
  file_name:
    "attention.pdf",
  chunk_count: 8,
};

type SidebarProps =
  ComponentProps<
    typeof DocumentSidebar
  >;

function makeProps(
  overrides:
    Partial<SidebarProps> = {},
): SidebarProps {
  return {
    documents: [
      DOCUMENT,
    ],
    selectedDocumentId: null,
    isLoading: false,
    isUploading: false,
    deletingDocumentId: null,
    error: null,

    onSelectDocument:
      vi.fn(),

    onUpload:
      vi.fn(
        async () => undefined,
      ),

    onDelete:
      vi.fn(
        async () => undefined,
      ),

    onRetry:
      vi.fn(
        async () => undefined,
      ),

    onDismissError:
      vi.fn(),

    ...overrides,
  };
}

describe(
  "DocumentSidebar",
  () => {
    afterEach(() => {
      vi.restoreAllMocks();
    });

    it("selects a document", async () => {
      const user =
        userEvent.setup();

      const onSelectDocument =
        vi.fn();

      render(
        <DocumentSidebar
          {...makeProps({
            onSelectDocument,
          })}
        />,
      );

      await user.click(
        screen.getByRole(
          "button",
          {
            name:
              "Select attention.pdf",
          },
        ),
      );

      expect(
        onSelectDocument,
      ).toHaveBeenCalledWith(
        DOCUMENT_ID,
      );
    });

    it("confirms before deleting a document", async () => {
      const user =
        userEvent.setup();

      const onDelete =
        vi.fn(
          async () =>
            undefined,
        );

      const confirm =
        vi.spyOn(
          window,
          "confirm",
        );

      confirm.mockReturnValue(
        true,
      );

      render(
        <DocumentSidebar
          {...makeProps({
            onDelete,
          })}
        />,
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

      expect(
        confirm,
      ).toHaveBeenCalledWith(
        'Delete "attention.pdf" from the index?',
      );

      expect(
        onDelete,
      ).toHaveBeenCalledWith(
        DOCUMENT_ID,
      );
    });

    it("does not delete when confirmation is cancelled", async () => {
      const user =
        userEvent.setup();

      const onDelete =
        vi.fn(
          async () =>
            undefined,
        );

      vi.spyOn(
        window,
        "confirm",
      ).mockReturnValue(false);

      render(
        <DocumentSidebar
          {...makeProps({
            onDelete,
          })}
        />,
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

      expect(
        onDelete,
      ).not.toHaveBeenCalled();
    });
  },
);