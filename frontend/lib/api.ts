import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add response interceptor for better error handling
api.interceptors.response.use(
  response => response,
  error => {
    console.error('API Error Details:', {
      message: error.message,
      code: error.code,
      baseURL: error.config?.baseURL,
      url: error.config?.url,
      status: error.response?.status,
      data: error.response?.data,
    });
    return Promise.reject(error);
  }
);

// Types
export interface ChatResponse {
  response: string;
  sources: string[];
  confidence?: number;
  confidenceLabel?: 'low' | 'medium' | 'high' | string;
  whyThisAnswer?: string;
  highlights?: { source: string; snippet: string }[];
  schemes?: {
    name: string;
    score: number;
    reason: string;
    eligibility: string;
    nextStep: string;
    missingCriteria?: string[];
  }[];
  comparison?: {
    schemeA: {
      name: string;
      score: number;
      pros: string[];
      cons: string[];
      fit: string;
    };
    schemeB: {
      name: string;
      score: number;
      pros: string[];
      cons: string[];
      fit: string;
    };
    recommendedFit: string;
  } | null;
  documentInsights?: {
    field: string;
    value: string;
    source: string;
  }[];
  actionPlan?: {
    title: string;
    steps: string[];
    reminders: {
      title: string;
      dueDate: string;
      frequency: string;
      category: string;
    }[];
  } | null;
  cached?: boolean;
}

export interface UserProfile {
  age?: number;
  gender?: string;
  income: string;
  employmentStatus: string;
  taxRegime: string;
  homeownerStatus: string;
  children?: string;
  childrenAges?: string;
  parentsAge?: string;
  investmentCapacity?: string;
  riskAppetite?: string;
  financialGoals?: string[];
  existingInvestments?: string[];
}

export interface User {
  id: string;
  email: string;
  username: string;
  age?: number;
  gender?: string;
  income?: string;
  employmentStatus?: string;
  taxRegime?: string;
  homeownerStatus?: string;
  children?: string;
  childrenAges?: string;
  parentsAge?: string;
  investmentCapacity?: string;
  riskAppetite?: string;
  financialGoals?: string[];
  existingInvestments?: string[];
  createdAt?: string;
  lastLogin?: string;
}

export interface ChatSession {
  id: string;
  userId: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  isActive: boolean;
  messageCount: number;
}

export interface ChatMessage {
  id: string;
  sessionId: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: string[];
  responseTime?: number;
  cached?: boolean;
  createdAt: string;
}

export interface SavedMessage {
  id: string;
  userId: string;
  messageId: string;
  content: string;
  note?: string;
  tags?: string[];
  savedAt: string;
}

export interface AnalyticsSummary {
  totalQueries: number;
  totalUploads: number;
  activeUsers: number;
  totalTaxSaved: number;
  avgResponseTime: number;
  cacheHitRate: number;
  topEvents: { type: string; count: number }[];
  period?: string;
}

export interface ChatHistoryMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface UploadResponse {
  status: string;
  message: string;
  document_id?: string;
}

export interface StatusResponse {
  initialized: boolean;
  documents_indexed: number;
  model: string | null;
}

// ===========================
// Authentication APIs
// ===========================

export async function register(
  email: string,
  username: string,
  password: string
): Promise<{ status: string; user: User; message?: string }> {
  const { data } = await api.post('/api/users/register', {
    email,
    username,
    password,
  });
  return data;
}

export async function login(
  email: string,
  password: string
): Promise<{ status: string; user: User; message?: string }> {
  const { data } = await api.post('/api/users/login', {
    email,
    password,
  });
  return data;
}

// ===========================
// Profile Management APIs
// ===========================

export async function getProfile(userId: string): Promise<User> {
  const { data } = await api.get(`/api/users/${userId}/profile`);
  return data;
}

export async function updateProfile(
  userId: string,
  profile: Partial<UserProfile>
): Promise<{ status: string;[key: string]: any }> {
  const { data } = await api.put(`/api/users/${userId}/profile`, profile);
  return data;
}

// ===========================
// Chat Session APIs
// ===========================

export async function createChatSession(
  userId: string,
  title?: string
): Promise<ChatSession> {
  const { data } = await api.post(`/api/users/${userId}/sessions`, { title });
  return data.session;
}

export async function getChatSessions(userId: string): Promise<ChatSession[]> {
  const { data } = await api.get(`/api/users/${userId}/sessions`);
  return data.sessions || [];
}

export async function getChatMessages(sessionId: string): Promise<ChatMessage[]> {
  const { data } = await api.get(`/api/sessions/${sessionId}/messages`);
  return data.messages || [];
}

export async function deleteChatSession(sessionId: string): Promise<{ status: string }> {
  const { data } = await api.delete(`/api/sessions/${sessionId}`);
  return data;
}

/**
 * Persist a user + assistant message pair to an existing chat session.
 * Used by the voice assistant to log conversational exchanges.
 */
export async function addSessionMessages(
  sessionId: string,
  userMessage: string,
  assistantMessage: string,
  sources?: string[],
  responseTime?: number
): Promise<{ status: string; userMessage: ChatMessage; assistantMessage: ChatMessage }> {
  const { data } = await api.post(`/api/sessions/${sessionId}/messages`, {
    userMessage,
    assistantMessage,
    sources,
    response_time: responseTime,
  });
  return data;
}

export async function updateChatSessionTitle(
  sessionId: string,
  title: string
): Promise<{ status: string; session: ChatSession }> {
  const { data } = await api.put(`/api/sessions/${sessionId}/title`, { title });
  return data;
}

// ===========================
// Saved Messages APIs
// ===========================

export async function saveMessage(
  userId: string,
  messageId: string,
  note?: string,
  tags?: string[]
): Promise<{ status: string; saved: SavedMessage }> {
  const { data } = await api.post(`/api/users/${userId}/saved-messages`, {
    message_id: messageId,
    note,
    tags,
  });
  return data;
}

export async function getSavedMessages(userId: string): Promise<SavedMessage[]> {
  const { data } = await api.get(`/api/users/${userId}/saved-messages`);
  return data.messages || [];
}

// ===========================
// Analytics APIs
// ===========================

export async function getAnalyticsSummary(days: number = 30): Promise<AnalyticsSummary> {
  const { data } = await api.get(`/api/analytics/summary`, {
    params: { days },
  });
  return data;
}

export async function getQueryDistribution(
  days: number = 30
): Promise<{ date: string; count: number }[]> {
  const { data } = await api.get(`/api/analytics/query-distribution`, {
    params: { days },
  });
  return data;
}

// ===========================
// Document Management APIs
// ===========================

export async function getUserDocuments(userId: string) {
  const { data } = await api.get(`/api/users/${userId}/documents`);
  return data.documents || [];
}

export async function deleteUserDocument(documentId: string): Promise<{ status: string; message: string }> {
  const { data } = await api.delete(`/api/documents/${documentId}`);
  return data;
}

// ===========================
// Chat APIs (Updated)
// ===========================

export async function sendMessage(
  message: string,
  profile?: UserProfile,
  history?: ChatHistoryMessage[],
  userId?: string,
  sessionId?: string,
  sourceFilter?: string
): Promise<ChatResponse> {
  const { data } = await api.post<ChatResponse>('/api/chat', {
    message,
    profile,
    history,
    userId,
    sessionId,
    sourceFilter,
  });

  // Handle response format (may have nested structure from Gemini)
  if (Array.isArray(data.response)) {
    // Extract text from Gemini's response format
    const textContent = data.response.find((item: any) => item.type === 'text');
    return {
      response: textContent?.text || 'No response',
      sources: data.sources,
    };
  }

  return data;
}

export async function sendMessageStream(
  message: string,
  profile: UserProfile | undefined,
  history: ChatHistoryMessage[],
  onToken: (token: string) => void,
  onSources: (sources: string[]) => void,
  onMeta?: (meta: Omit<ChatResponse, 'response' | 'sources'>) => void,
  userId?: string,
  sessionId?: string,
  sourceFilter?: string
): Promise<void> {
  let streamResponseOk = false;
  try {
    const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message,
        profile,
        history,
        userId,
        sessionId,
        sourceFilter,
      }),
    });

    streamResponseOk = response.ok;

    if (!response.ok || !response.body) {
      throw new Error(`Streaming failed with status ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split('\n\n');
      buffer = parts.pop() || '';

      for (const part of parts) {
        const lines = part.split('\n');
        let event = 'message';
        let data = '';

        for (const line of lines) {
          if (line.startsWith('event:')) {
            event = line.replace('event:', '').trim();
          } else if (line.startsWith('data:')) {
            data += line.replace('data:', '').trim();
          }
        }

        if (!data) continue;

        if (event === 'token') {
          try {
            const parsed = JSON.parse(data);
            onToken(typeof parsed === 'string' ? parsed : String(parsed ?? ''));
          } catch {
            // Be permissive to avoid dropping the whole stream on a single malformed chunk.
            onToken(data);
          }
        } else if (event === 'sources') {
          try {
            const parsed = JSON.parse(data);
            onSources(Array.isArray(parsed) ? parsed : []);
          } catch {
            onSources([]);
          }
        } else if (event === 'meta') {
          try {
            const parsed = JSON.parse(data);
            if (parsed && typeof parsed === 'object') {
              onMeta?.(parsed);
            }
          } catch {
            // Ignore malformed metadata packets and keep token streaming alive.
          }
        } else if (event === 'error') {
          let message = data;
          try {
            message = JSON.parse(data);
          } catch {
            // Keep raw payload if not JSON.
          }
          throw new Error(String(message));
        } else if (event === 'done') {
          return;
        }
      }
    }
  } catch (error) {
    // If stream endpoint accepted the request, do not also call /api/chat.
    // This avoids duplicate requests and backend 500s from fallback path.
    if (streamResponseOk) {
      throw error;
    }

    const fallback = await sendMessage(message, profile, history, userId, sessionId, sourceFilter);
    onToken(fallback.response);
    onSources(fallback.sources || []);
    onMeta?.({
      confidence: fallback.confidence,
      confidenceLabel: fallback.confidenceLabel,
      whyThisAnswer: fallback.whyThisAnswer,
      highlights: fallback.highlights,
      schemes: fallback.schemes,
      comparison: fallback.comparison,
      documentInsights: fallback.documentInsights,
      actionPlan: fallback.actionPlan,
      cached: fallback.cached,
    });
  }
}

export async function uploadDocument(file: File, userId?: string): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);
  if (userId) {
    formData.append('user_id', userId);
  }

  // Create a separate axios instance for FormData without default headers
  const uploadApi = axios.create({
    baseURL: API_BASE_URL,
  });

  uploadApi.interceptors.response.use(
    response => response,
    error => {
      console.error('Upload Error Details:', {
        message: error.message,
        code: error.code,
        baseURL: error.config?.baseURL,
        url: error.config?.url,
        status: error.response?.status,
        data: error.response?.data,
      });
      return Promise.reject(error);
    }
  );

  const { data } = await uploadApi.post<UploadResponse>('/api/upload', formData);

  return data;
}

export async function getStatus(): Promise<StatusResponse> {
  const { data } = await api.get<StatusResponse>('/api/status');
  return data;
}

export async function healthCheck(): Promise<boolean> {
  try {
    const { data } = await api.get('/ping');
    return data.status === 'ok';
  } catch {
    return false;
  }
}

export default api;
