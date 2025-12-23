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
