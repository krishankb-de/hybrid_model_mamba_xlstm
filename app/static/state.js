// app/static/state.js — pure reducers: the stored event log rebuilds exactly the live view.
//
// applyEvent folds one {event, data} (from the stream or from polling) into a new view and never mutates what it is
// given; replay folds a stored log, so a reload shows what was streamed. Stage keys are inserted in arrival order, so
// Object.keys(view.stages) follows the contract order. A stage keeps the state its last event gave it: a turn that ends
// aborted or in error never sends the stage_end of the stage it was in, so that stage stays 'running'. What to show
// for it is decided from view.status as well.
export function initialView(messageId) {
  return { id: messageId, status: 'running', lastSeq: 0, mode: null, provenance: null, options: null,
           image: null, stages: {}, report: '', displayReport: '', provisional: false, truncated: false,
           neighbors: [], matches: [], trueRank: null, labels: null, agreement: null, score: null,
           notices: [], error: null, totalMs: null };
}

export function applyEvent(view, { event, data }) {
  if (data.seq <= view.lastSeq) return view;            // polling can resend what the stream delivered
  const v = { ...view, lastSeq: data.seq };
  switch (event) {
    case 'message_start':
      return { ...v, id: data.message_id, mode: data.mode, provenance: data.model, options: data.options, image: data.image };
    case 'stage_start':
      return { ...v, stages: { ...v.stages, [data.stage]: { state: 'running' } } };
    case 'stage_end': {
      const st = data.skipped ? { state: 'skipped', skipped: data.skipped }
                              : { state: 'done', ms: data.ms, detail: data.detail };
      const next = { ...v, stages: { ...v.stages, [data.stage]: st } };
      const d = data.detail || {};
      if (data.stage === 'retrieve') return { ...next, neighbors: d.image_neighbors || [], matches: d.report_matches || [], trueRank: d.true_report_rank || null };
      if (data.stage === 'label') return { ...next, labels: d.chexbert_14 || null, agreement: d.neighbor_agreement || null };
      if (data.stage === 'score') return { ...next, score: d };
      return next;
    }
    case 'content_block_delta':
      return { ...v, report: data.delta.text, provisional: true };
    case 'content_block_stop':
      return { ...v, provisional: false };
    case 'warning':
      return { ...v, notices: [...v.notices, data] };
    case 'error':
      return { ...v, error: data.error };
    case 'message_stop':
      return { ...v, status: data.status, report: data.report ?? v.report, displayReport: data.display_report ?? v.report,
               truncated: !!data.truncated_mid_sentence, provisional: false, totalMs: data.total_ms };
    default:
      return v;
  }
}

export const replay = (events) => events.reduce(applyEvent, initialView(events.length ? events[0].data.message_id : null));
