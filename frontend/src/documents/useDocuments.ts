import {
  useCallback,
  useEffect,
  useState,
} from "react";

import {
  ResearchAPIError,
  researchApi,
  type StoredDocument,
} from "../api";

export interface DocumentClient {
  listDocuments(): Promise<StoredDocument[]>;

  uploadDocument(
    file: File,
  ): Promise<StoredDocument>;

  deleteDocument(
    documentId: string,
  ): Promise<void>;
}

interface UseDocumentsResult {
  documents: StoredDocument[];
  isLoading: boolean;
  isUploading: boolean;
  deletingDocumentId: string | null;
  error: string | null;
  refreshDocuments: () => Promise<void>;
  uploadDocument: (
    file: File,
  ) => Promise<StoredDocument | null>;
  deleteDocument: (
    documentId: string,
  ) => Promise<boolean>;
  clearError: () => void;
}

function sortDocuments(
  documents: StoredDocument[],
): StoredDocument[] {
  return [...documents].sort(
    (left, right) =>
      left.file_name.localeCompare(
        right.file_name,
      ),
  );
}

function getErrorMessage(
  error: unknown,
): string {
  if (error instanceof ResearchAPIError) {
    return error.message;
  }

  if (error instanceof Error) {
    return error.message;
  }

  return "An unexpected error occurred.";
}

export function useDocuments(
  client: DocumentClient = researchApi,
): UseDocumentsResult {
  const [
    documents,
    setDocuments,
  ] = useState<StoredDocument[]>([]);

  const [
    isLoading,
    setIsLoading,
  ] = useState(true);

  const [
    isUploading,
    setIsUploading,
  ] = useState(false);

  const [
    deletingDocumentId,
    setDeletingDocumentId,
  ] = useState<string | null>(null);

  const [
    error,
    setError,
  ] = useState<string | null>(null);

  const refreshDocuments =
    useCallback(async () => {
      setIsLoading(true);
      setError(null);

      try {
        const response =
          await client.listDocuments();

        setDocuments(
          sortDocuments(response),
        );
      } catch (caughtError) {
        setError(
          getErrorMessage(caughtError),
        );
      } finally {
        setIsLoading(false);
      }
    }, [client]);

  const uploadDocument =
    useCallback(
      async (
        file: File,
      ): Promise<StoredDocument | null> => {
        setIsUploading(true);
        setError(null);

        try {
          const uploaded =
            await client.uploadDocument(file);

          setDocuments(
            (current) =>
              sortDocuments([
                ...current.filter(
                  (document) =>
                    document.document_id !==
                    uploaded.document_id,
                ),
                uploaded,
              ]),
          );

          return uploaded;
        } catch (caughtError) {
          setError(
            getErrorMessage(caughtError),
          );

          return null;
        } finally {
          setIsUploading(false);
        }
      },
      [client],
    );

  const deleteDocument =
    useCallback(
      async (
        documentId: string,
      ): Promise<boolean> => {
        setDeletingDocumentId(
          documentId,
        );

        setError(null);

        try {
          await client.deleteDocument(
            documentId,
          );

          setDocuments(
            (current) =>
              current.filter(
                (document) =>
                  document.document_id !==
                  documentId,
              ),
          );

          return true;
        } catch (caughtError) {
          setError(
            getErrorMessage(caughtError),
          );

          return false;
        } finally {
          setDeletingDocumentId(null);
        }
      },
      [client],
    );

  const clearError =
    useCallback(() => {
      setError(null);
    }, []);

  useEffect(() => {
    let cancelled = false;

    client
      .listDocuments()
      .then((response) => {
        if (cancelled) {
          return;
        }

        setDocuments(
          sortDocuments(response),
        );

        setError(null);
      })
      .catch((caughtError: unknown) => {
        if (cancelled) {
          return;
        }

        setError(
          getErrorMessage(caughtError),
        );
      })
      .finally(() => {
        if (cancelled) {
          return;
        }

        setIsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [client]);

  return {
    documents,
    isLoading,
    isUploading,
    deletingDocumentId,
    error,
    refreshDocuments,
    uploadDocument,
    deleteDocument,
    clearError,
  };
}