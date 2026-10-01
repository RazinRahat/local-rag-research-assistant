import {
  type ChangeEvent,
  useRef,
} from "react";

import type {
  StoredDocument,
} from "../api";

interface DocumentSidebarProps {
  documents: StoredDocument[];
  selectedDocumentId: string | null;
  isLoading: boolean;
  isUploading: boolean;
  deletingDocumentId: string | null;
  error: string | null;
  onSelectDocument: (
    documentId: string | null,
  ) => void;
  onUpload: (
    file: File,
  ) => Promise<void>;
  onDelete: (
    documentId: string,
  ) => Promise<void>;
  onRetry: () => Promise<void>;
  onDismissError: () => void;
}

export function DocumentSidebar({
  documents,
  selectedDocumentId,
  isLoading,
  isUploading,
  deletingDocumentId,
  error,
  onSelectDocument,
  onUpload,
  onDelete,
  onRetry,
  onDismissError,
}: DocumentSidebarProps) {
  const fileInputRef =
    useRef<HTMLInputElement>(null);

  async function handleFileChange(
    event: ChangeEvent<HTMLInputElement>,
  ): Promise<void> {
    const file =
      event.target.files?.[0];

    event.target.value = "";

    if (!file) {
      return;
    }

    await onUpload(file);
  }

  async function handleDelete(
    document: StoredDocument,
  ): Promise<void> {
    const confirmed =
      window.confirm(
        `Delete "${document.file_name}" from the index?`,
      );

    if (!confirmed) {
      return;
    }

    await onDelete(
      document.document_id,
    );
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-heading">
        <div>
          <p className="eyebrow">
            LIBRARY
          </p>

          <h2>Documents</h2>
        </div>

        <span
          className="document-count"
          aria-label={`${documents.length} indexed ${
            documents.length === 1
              ? "document"
              : "documents"
          }`}
        >
          {documents.length}
        </span>
      </div>

      <input
        ref={fileInputRef}
        className="visually-hidden"
        type="file"
        accept=".pdf,application/pdf"
        aria-label="Upload PDF"
        onChange={(event) => {
          void handleFileChange(event);
        }}
      />

      <button
        className="upload-button"
        type="button"
        disabled={isUploading}
        onClick={() =>
          fileInputRef.current?.click()
        }
      >
        <span aria-hidden="true">
          +
        </span>{" "}
        {isUploading
          ? "Indexing document..."
          : "Add document"}
      </button>

      {error && (
        <div
          className="sidebar-error"
          role="alert"
        >
          <p>{error}</p>

          <div className="error-actions">
            <button
              type="button"
              onClick={() => {
                void onRetry();
              }}
            >
              Retry
            </button>

            <button
              type="button"
              onClick={onDismissError}
            >
              Dismiss
            </button>
          </div>
        </div>
      )}

      <div className="document-list">
        <button
          className={`document-scope ${
            selectedDocumentId === null
              ? "selected"
              : ""
          }`}
          type="button"
          aria-pressed={
            selectedDocumentId === null
          }
          onClick={() =>
            onSelectDocument(null)
          }
        >
          <span className="document-icon">
            ◫
          </span>

          <span>
            <strong>
              All documents
            </strong>

            <small>
              Search the complete corpus
            </small>
          </span>
        </button>

        {isLoading ? (
          <div
            className="library-status"
            aria-live="polite"
          >
            Loading documents...
          </div>
        ) : documents.length === 0 ? (
          <div className="empty-library">
            <p>
              No documents indexed yet.
            </p>

            <span>
              Add a research PDF to
              start building your local
              knowledge base.
            </span>
          </div>
        ) : (
          documents.map(
            (document) => {
              const selected =
                selectedDocumentId ===
                document.document_id;

              const deleting =
                deletingDocumentId ===
                document.document_id;

              return (
                <div
                  className={`document-row ${
                    selected
                      ? "selected"
                      : ""
                  }`}
                  key={
                    document.document_id
                  }
                >
                  <button
                    className="document-select"
                    type="button"
                    aria-label={`Select ${document.file_name}`}
                    aria-pressed={selected}
                    onClick={() =>
                      onSelectDocument(
                        document.document_id,
                      )
                    }
                  >
                    <span className="document-name">
                      {
                        document.file_name
                      }
                    </span>

                    <span className="document-meta">
                      {
                        document.chunk_count
                      }{" "}
                      {document.chunk_count ===
                      1
                        ? "chunk"
                        : "chunks"}
                    </span>
                  </button>

                  <button
                    className="delete-document"
                    type="button"
                    aria-label={`Delete ${document.file_name}`}
                    disabled={deleting}
                    onClick={() => {
                      void handleDelete(
                        document,
                      );
                    }}
                  >
                    {deleting
                      ? "…"
                      : "×"}
                  </button>
                </div>
              );
            },
          )
        )}
      </div>

      <div className="sidebar-footer">
        <span className="status-dot" />

        Local index
      </div>
    </aside>
  );
}