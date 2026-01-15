export type SimilarityResultOut = {
  backend: string;
  score: number;
  interpretation: string;
};

export type SemanticScoreResponse = {
  prompt_en: string;
  target_es: string;
  gloss_en: string;
  scores: Record<string, SimilarityResultOut>;
  nearest: Record<string, Array<[string, number]>>;
};

export type SemanticScoreRequest = {
  lesson_id: number;
  item_id: number;
  learner_es: string;
};

export type LabelScore = {
  label: string;
  score: number;
};

export type TutorAction = {
  code: string;
  message: string;
  severity: "info" | "warn" | "block";
};

export type EntityOut = {
  text: string;
  label: string;
  start_char: number;
  end_char: number;
  explanation?: string | null;
};

export type NerOut = {
  entities: EntityOut[];
  noun_phrases: string[];
};

export type FluencyOut = { perplexity: number; band: string };
export type FluencyField = FluencyOut | null;

export type TutorEvaluateRequest = {
  lesson_id: number;
  item_id: number;
  learner_text: string;
  expected_lang: string;
  allow_mixed?: boolean;
  prompt_en?: string;
  target_text?: string;
  gloss_en?: string;
};

export type TutorEvaluateResponse = {
  prompt_en: string;
  target_es: string;
  gloss_en: string;
  expected_lang: string;

  detected_lang: string;
  detected_top_k: LabelScore[];

  syntax_issues: any[];
  fluency: FluencyField;

  ner: NerOut;

  semantics: Record<string, SimilarityResultOut>;
  nearest: Record<string, Array<[string, number]>>;

  action: TutorAction;
};

export type SyntaxIssue = { code?: string; message?: string } & Record<string, any>;

export type NerEntityOut = {
  text: string;
  label: string;
  start_char: number;
  end_char: number;
  explanation?: string | null;
};

export type NerResponse = {
  entities: NerEntityOut[];
  noun_phrases: string[];
};

