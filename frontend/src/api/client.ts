import type {
  CitedRAGResponse,
  HealthResponse,
  QueryRequest,
  SearchRequest,
  SearchResponse,
  StoredDocument,
} from "./models";

interface ErrorPayload {
  detail?: unknown;
}

export class ResearchAPIError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);

    this.name = "ResearchAPIError";
    this.status = status;
  }
}

function formatErrorDetail(
  detail: unknown,
  fallback: string,
): string {
  if (typeof detail === "string") {
    return detail;
  }

  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => {
        if (
          typeof item === "object" &&
          item !== null &&
          "msg" in item &&
          typeof item.msg === "string"
        ) {
          return item.msg;
        }

        return null;
      })
      .filter((message): message is string => message !== null);

    if (messages.length > 0) {
      return messages.join("; ");
    }
  }

  return fallback;
}

async function assertResponseOk(
  response: Response,
): Promise<void> {
  if (response.ok) {
    return;
  }

  const fallback =
    response.statusText ||
    `Request failed with status ${response.status}`;

  let message = fallback;

  try {
    const payload =
      (await response.json()) as ErrorPayload;

    message = formatErrorDetail(
      payload.detail,
      fallback,
    );
  } catch {
    // The response did not contain JSON error details.
  }

  throw new ResearchAPIError(
    response.status,
    message,
  );
}

export class ResearchAPIClient {
  private readonly baseUrl: string;

  constructor(baseUrl = "/api") {
    this.baseUrl = baseUrl.replace(/\/$/, "");
  }

  private async requestJson<T>(
    path: string,
    init?: RequestInit,
  ): Promise<T> {
    const response = await fetch(
      `${this.baseUrl}${path}`,
      init,
    );

    await assertResponseOk(response);

    return (await response.json()) as T;
  }

  async health(): Promise<HealthResponse> {
    return this.requestJson<HealthResponse>(
      "/health",
    );
  }

  async listDocuments(): Promise<StoredDocument[]> {
    return this.requestJson<StoredDocument[]>(
      "/documents",
    );
  }

  async uploadDocument(
    file: File,
  ): Promise<StoredDocument> {
    const body = new FormData();

    body.append(
      "file",
      file,
      file.name,
    );

    return this.requestJson<StoredDocument>(
      "/documents",
      {
        method: "POST",
        body,
      },
    );
  }

  async deleteDocument(
    documentId: string,
  ): Promise<void> {
    const response = await fetch(
      `${this.baseUrl}/documents/${encodeURIComponent(documentId)}`,
      {
        method: "DELETE",
      },
    );

    await assertResponseOk(response);
  }

  async search(
    request: SearchRequest,
  ): Promise<SearchResponse> {
    return this.requestJson<SearchResponse>(
      "/search",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(request),
      },
    );
  }

  async query(
    request: QueryRequest,
  ): Promise<CitedRAGResponse> {
    return this.requestJson<CitedRAGResponse>(
      "/query",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(request),
      },
    );
  }
}

export const researchApi =
  new ResearchAPIClient();