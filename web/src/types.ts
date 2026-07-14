export type AgentEventName =
  | "run_started"
  | "planner_action"
  | "tool_result"
  | "answer"
  | "completed"
  | "failed";

export interface AgentBudget {
  max_steps: number;
  max_searches: number;
  max_graph_expansions: number;
  max_reads: number;
  max_evidence_chunks: number;
  max_graph_hops: number;
  evidence_token_budget: number;
}

export interface AgentRunRequest {
  question: string;
  workspace_id?: string;
  profile_id?: string;
  budget: AgentBudget;
  database_url?: string;
  embedding_provider?: "flagembedding" | "hash";
  embedding_model?: string;
  embedding_dimensions?: number;
  use_deepseek: boolean;
  top_k?: number;
  context_token_budget?: number;
  graph_hops?: number;
  reranker_enabled?: boolean;
  rerank_candidate_multiplier?: number;
}

export interface Workspace {
  id: string;
  name: string;
  created_at: string;
  uploads: string[];
}

export interface IngestReport {
  discovered: number;
  inserted: number;
  updated: number;
  skipped: number;
  chunks_written: number;
}

export interface GraphBuildReport {
  run_id: string;
  status: string;
  chunks: {
    total: number;
    cached: number;
    scheduled: number;
    succeeded: number;
    failed: number;
  };
  graph: { nodes: number; edges: number };
}

export interface AgentEvent {
  event: AgentEventName;
  run_id: string;
  step: number;
  data: Record<string, unknown>;
}

export interface GroundedAnswer {
  answer: string;
  citations: string[];
  insufficient_evidence: boolean;
}

export interface EvidenceItem {
  citation_id: string;
  chunk_id: string;
  document_id: string;
  document_title: string;
  section_path: string[];
  page_start: number | null;
  page_end: number | null;
  token_count: number;
  score: number;
  text: string;
  source_entity_ids?: string[];
  source_relation_ids?: string[];
}

export interface RuntimeDefaults {
  database_url?: string;
  embedding_provider?: "flagembedding" | "hash";
  embedding_model?: string;
  embedding_dimensions?: number;
  agent_budget?: Partial<AgentBudget>;
  retrieval?: {
    top_k?: number;
    context_token_budget?: number;
    graph_hops?: number;
  };
}

export interface IndexBuildReport {
  profile_id: string;
  chunks: number;
  entities: number;
  relations: number;
  reused: boolean;
}
