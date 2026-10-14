// app/static/glossary.js — the plain explanations of the report card (CHAT_UI_PLAN.md P6-H): static data, nothing else.
//
// Each row is the words a report uses for a thing (the first is the one the Terms list names it by, the others are the same thing said another
// way) and what it says in plain language. render.js marks these words in a report and explains each once, and its summary says the CheXbert
// labels in the same words. The explanations are the user's own table, word for word, so a change here is a change to what the page tells a
// reader. The tables are frozen, so that no script can change them at run time.

const row = (terms, explanation) => Object.freeze({ terms: Object.freeze(terms), explanation });

// The 14 labels first (the label "Pleural Other" is the row "pleural thickening"), then the words of a report that a reader meets most.
export const GLOSSARY = Object.freeze([
  row(['no finding'], 'no abnormality is reported'),
  row(['enlarged cardiomediastinum'], 'the heart and the central chest look wider than normal'),
  row(['cardiomegaly'], 'an enlarged heart'),
  row(['lung opacity', 'opacity'], 'an area of the lung that looks whiter, or denser, than normal'),
  row(['lung lesion', 'lesion'], 'a spot or mass in the lung'),
  row(['edema'], 'fluid in the lung tissue'),
  row(['consolidation'], 'part of the lung filled with fluid or inflammation instead of air'),
  row(['pneumonia'], 'a lung infection'),
  row(['atelectasis'], 'part of a lung is partly collapsed or not fully inflated'),
  row(['pneumothorax'], 'air between the lung and the chest wall, which can make the lung collapse'),
  row(['pleural effusion', 'effusion'], 'fluid between the lung and the chest wall'),
  row(['pleural thickening'], 'a thickened lining around the lung'),
  row(['fracture'], 'a broken bone, for example a rib'),
  row(['support devices', 'device'], 'tubes, lines, wires or devices placed in the body'),
  row(['vascular congestion'], 'the blood vessels in the lungs look fuller than normal'),
  row(['hilar', 'hila'], 'around the roots of the lungs, where vessels and airways enter'),
  row(['basilar', 'base', 'bases'], 'the lower parts of the lungs'),
  row(['costophrenic angle'], 'the lower outer corner of the lung on the image'),
  row(['mediastinum', 'mediastinal'], 'the central chest between the lungs'),
  row(['interstitial'], 'in the tissue between the air sacs'),
  row(['nodule'], 'a small round spot'),
  row(['granuloma'], 'a small healed spot, often from an old infection'),
  row(['hemidiaphragm'], 'one half of the breathing muscle under the lungs'),
  row(['emphysema'], 'damaged air sacs that trap air'),
  row(['sternotomy wires'], 'wires that close the breastbone after chest surgery'),
  row(['endotracheal tube'], 'a breathing tube in the windpipe'),
  row(['nasogastric tube'], 'a tube through the nose into the stomach'),
  row(['picc', 'picc line'], 'a thin tube in an arm vein that reaches near the heart'),
  row(['pacemaker'], 'a device that helps the heart keep a regular beat'),
  row(['scoliosis'], 'a sideways curve of the spine'),
  row(['osseous'], 'about the bones'),
  row(['focal'], 'in one small area'),
  row(['bilateral'], 'on both sides'),
  row(['acute'], 'new or sudden'),
  row(['chronic'], 'long-standing'),
]);

// The CheXbert label names of the summary, in the order it says them (the order of CHEXBERT_14, with Support Devices last), each with the row of
// the table it is said by. "No Finding" is not here: a report with it says no abnormal finding, in a sentence of its own (render.js summaryText).
export const SUMMARY_LABELS = Object.freeze([
  ['Enlarged Cardiomediastinum', 'enlarged cardiomediastinum'],
  ['Cardiomegaly', 'cardiomegaly'],
  ['Lung Opacity', 'lung opacity'],
  ['Lung Lesion', 'lung lesion'],
  ['Edema', 'edema'],
  ['Consolidation', 'consolidation'],
  ['Pneumonia', 'pneumonia'],
  ['Atelectasis', 'atelectasis'],
  ['Pneumothorax', 'pneumothorax'],
  ['Pleural Effusion', 'pleural effusion'],
  ['Pleural Other', 'pleural thickening'],
  ['Fracture', 'fracture'],
  ['Support Devices', 'support devices'],
].map((pair) => Object.freeze(pair)));
