export type AnalyzeResponse = {
  ok: boolean;
  file?: {
    size: number;
    mimetype: string;
  };
predict?: {
    best_label: string;
    best_confidence: number;
    topk?: [{
       label: string; 
       confidence: number;
      }];
  };
  meta?: {
    userId?: string;
  };
  bytes?: number;
};
